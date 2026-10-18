"""SM121 warp-MMA FP8 linear training primitive.

Unlike Hopper's TMA/WGMMA kernels, uses Triton pointwise quantization and
PyTorch's architecture-dispatched FP8 GEMM. It MUST pass on-device preflight
before GB10_FP8_LINEAR=1. Forward is E4M3 x E4M3 -> BF16; backward defaults
to BF16 GEMMs to protect optimizer convergence. Quantization uses per-tensor
FP32 amax and explicit +/-448 saturation. No silent BF16 fallback.

GB10 has warp MMA but not SM100's tcgen05/TMEM. Performance depends on the
installed sm_121-enabled cuBLASLt/torch stack: benchmark on both GB10s.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(src, dst, scale, N:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(src+i, i<N, other=0).to(tl.float32)
    s=tl.load(scale)
    v=tl.minimum(448.,tl.maximum(-448., x / s))
    tl.store(dst+i,v.to(tl.float8e4nv),i<N)


def quantize_e4m3(x):
    if not x.is_cuda:
        raise RuntimeError("FP8 Triton quantization needs CUDA")
    x=x.contiguous()
    with torch.no_grad():
        # Avoid NaNs in scale and guard tiny/zero magnitude tensors.
        max_abs=torch.nan_to_num(x.float().abs().amax(),nan=0.0,posinf=448,neginf=448)
        scale=(max_abs.float()*1.05/448.).clamp(min=1e-8).contiguous()
        out=torch.empty(x.shape,device=x.device,dtype=torch.float8_e4m3fn)
        from gb10.triton_tiles import quant_block
        block = quant_block()
        _quantize[(triton.cdiv(x.numel(),block),)](x,out,scale,x.numel(),block,num_warps=4)
    return out,scale


class _Fp8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        # weight is [out,in] like torch.nn.functional.linear.
        assert x.dtype == weight.dtype == torch.bfloat16
        assert x.size(-1)==weight.shape[-1] and x.size(-1)%16==0
        x2=x.reshape(-1,x.shape[-1]).contiguous()
        w=weight.contiguous()
        xq,sx=quantize_e4m3(x2)
        wq,sw=quantize_e4m3(w)
        try:
            output=torch._scaled_mm(
                xq,wq.T.contiguous(),out_dtype=torch.bfloat16,
                scale_a=sx,scale_b=sw,use_fast_accum=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "sm_121 FP8 GEMM unavailable; disable GB10_FP8_LINEAR or "
                "install a GB10-enabled PyTorch/cuBLASLt container."
            ) from exc
        ctx.save_for_backward(x,weight)
        return output.reshape(*x.shape[:-1],weight.shape[0])

    @staticmethod
    def backward(ctx,grad_out):
        x,w=ctx.saved_tensors
        g=grad_out.reshape(-1,grad_out.shape[-1]).contiguous()
        x2=x.reshape(-1,x.shape[-1])
        # Keep gradients in BF16 until GB10 FP8 dgrad and wgrad are validated
        # against ANVIL convergence over the full #92 schedule.
        dx=(g@w).reshape_as(x) if ctx.needs_input_grad[0] else None
        dw=(g.T@x2).reshape_as(w) if ctx.needs_input_grad[1] else None
        return dx,dw


def linear(x,weight):
    return _Fp8Linear.apply(x,weight)


def enabled():
    import os
    return os.environ.get("GB10_FP8_LINEAR","0") == "1"


def dispatch(x,weight):
    if enabled():
        return linear(x,weight)
    return torch.nn.functional.linear(x,weight)
