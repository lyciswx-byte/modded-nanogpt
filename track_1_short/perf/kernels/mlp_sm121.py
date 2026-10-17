"""SM121 FP8 MLP: portable FP8 tensor-core GEMMs with fused Triton epilogues.

No Hopper TensorDescriptor/TMA kernel is launched here. GB10 uses cuBLASLt FP8
GEMMs through torch._scaled_mm and shape-tunable Triton epilogues which fuse
relu-square / dpre, FP8 saturation, both storage layouts, and delayed amax.

The parameter/cache/gradient contract intentionally matches the record #92
FusedLinearReLUSquareFunction. This *is not* the original Hopper fused GEMM,
and is not a throughput improvement until measured on a real SM121 device.

Tune independently of torch.compile / CUDA-graph capture with environment:
  GB10_FP8_MLP_TILE_M=32  (16, 32, 64)
  GB10_FP8_MLP_TILE_N=128 (64, 128)
Values are process-static: never change them during a compiled run.
"""
import os

import torch
import triton
import triton.language as tl

# Import registers the four existing custom FP8 GEMMs and glue_fp8_t.
from track_1_short.perf.kernels.mlp import quantize_dual_layout_fused


def sm121_mlp_tiles() -> tuple[int, int]:
    m = int(os.environ.get("GB10_FP8_MLP_TILE_M", "32"))
    n = int(os.environ.get("GB10_FP8_MLP_TILE_N", "128"))
    if m not in (16, 32, 64) or n not in (64, 128):
        raise ValueError("SM121 FP8 MLP tiles must use M=16/32/64, N=64/128")
    return m, n


# Read once, before torch.compile. Tile sweeps must run in separate processes.
TILE_M, TILE_N = sm121_mlp_tiles()


@triton.jit
def _sm121_post_quant(PRE, POST, POST_T, SCALE, AMAX,
                      M: tl.constexpr, N: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (m[:, None] < M) & (n[None, :] < N)
    pre = tl.load(PRE + m[:, None] * N + n[None, :], mask, 0).to(tl.float32)
    # Original Hopper epilogue rounds pre to BF16, and relu(pre)^2 to BF16
    # before quantization. _scaled_mm already writes the rounded BF16 pre.
    pos = tl.maximum(pre, 0.0)
    post = (pos * pos).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.maximum(tl.load(SCALE), 1.0e-12)
    q = tl.minimum(post * inv, 448.0).to(tl.float8e4nv)
    tl.store(POST + m[:, None] * N + n[None, :], q, mask)
    tl.store(POST_T + n[:, None] * M + m[None, :], tl.trans(q), tl.trans(mask))
    peak = tl.max(tl.max(tl.where(mask, post, 0.0), 1), 0)
    tl.atomic_max(AMAX, peak, sem="relaxed")


@triton.jit
def _sm121_dpre_quant(DPOST, POST, DPRE, DPRE_T, POST_SCALE, DPRE_SCALE, AMAX,
                      M: tl.constexpr, N: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (m[:, None] < M) & (n[None, :] < N)
    idx = m[:, None] * N + n[None, :]
    dpost = tl.load(DPOST + idx, mask, 0).to(tl.float32)
    post = tl.load(POST + idx, mask, 0).to(tl.float32)
    # d(relu(pre)^2)/dpre = 2 * relu(pre); reconstruction follows the
    # original FP8 path's sqrt(quantized_post * delayed_post_scale).
    p_scale = tl.load(POST_SCALE)
    dpre = (2.0 * dpost * tl.sqrt(tl.maximum(post * p_scale, 0.0))).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.maximum(tl.load(DPRE_SCALE), 1.0e-12)
    quant = tl.maximum(tl.minimum(dpre * inv, 57344.0), -57344.0).to(tl.float8e5)
    tl.store(DPRE + idx, quant, mask)
    tl.store(DPRE_T + n[:, None] * M + m[None, :], tl.trans(quant), tl.trans(mask))
    peak = tl.max(tl.max(tl.where(mask, tl.abs(dpre), 0.0), 1), 0)
    tl.atomic_max(AMAX, peak, sem="relaxed")


def post_quant_sm121(pre: torch.Tensor, scale: torch.Tensor, amax: torch.Tensor,
                     tile_m: int = TILE_M, tile_n: int = TILE_N):
    """BF16 GEMM output -> E4M3 post in both GEMM layouts, recording amax."""
    assert pre.ndim == 2 and pre.dtype == torch.bfloat16 and pre.is_contiguous()
    m, n = pre.shape
    post = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=pre.device)
    post_t = torch.empty((n, m), dtype=torch.float8_e4m3fn, device=pre.device)
    _sm121_post_quant[(triton.cdiv(m, tile_m), triton.cdiv(n, tile_n))](
        pre, post, post_t, scale, amax, m, n, BM=tile_m, BN=tile_n,
        num_warps=4, num_stages=2)
    return post, post_t


def dpre_quant_sm121(dpost: torch.Tensor, post: torch.Tensor, post_scale: torch.Tensor,
                     dpre_scale: torch.Tensor, amax: torch.Tensor,
                     tile_m: int = TILE_M, tile_n: int = TILE_N):
    """BF16 GEMM backward -> E5M2 dpre in both layouts, recording amax."""
    assert dpost.ndim == 2 and dpost.dtype == torch.bfloat16 and dpost.is_contiguous()
    m, n = dpost.shape
    dpre = torch.empty((m, n), dtype=torch.float8_e5m2, device=dpost.device)
    dpre_t = torch.empty((n, m), dtype=torch.float8_e5m2, device=dpost.device)
    _sm121_dpre_quant[(triton.cdiv(m, tile_m), triton.cdiv(n, tile_n))](
        dpost, post, dpre, dpre_t, post_scale, dpre_scale, amax,
        m, n, BM=tile_m, BN=tile_n, num_warps=4, num_stages=2)
    return dpre, dpre_t


class SM121FP8MLPFunction(torch.autograd.Function):
    """Record #92 FP8 MLP argument/gradient contract with SM121-safe epilogues.

    Unlike the Hopper path, GEMMs and epilogues are separate launches.
    The delay-scaled activation and dpre amax and the optional folded residual
    post-lambda are preserved, including their custom gradients.
    """

    @staticmethod
    def forward(ctx, x, W1, W2, W1_f8=None, dequant_scale=None, x_f8=None, W2_f8=None,
                w2_scale=None, post_scale=None, post_amax=None, w2_f8_row=None,
                dq_bwd=None, x_f8_t=None, x_scale=None, w1_f8_col=None,
                w1_scale=None, g_scale=None, dpre_scale=None, dpre_amax=None, p_fold=None):
        if W1_f8 is None:
            # Evaluation never uses FP8 MLP caches.
            return torch.nn.functional.linear(
                torch.relu(torch.nn.functional.linear(x, W1)).square(), W2.T)
        x_flat = x.reshape(-1, x.shape[-1])
        x8 = x_f8.reshape(-1, x_f8.shape[-1])
        pre = torch._scaled_mm(x8, W1_f8.T, out_dtype=torch.bfloat16,
                               scale_a=x_scale, scale_b=w1_scale, use_fast_accum=True)
        post_f8, post_t = post_quant_sm121(pre, post_scale, post_amax)
        out = torch.ops.nanogpt.dp_f8(post_f8, W2_f8, post_scale, w2_scale)
        ctx.save_for_backward(*((x, W1, W2) + ((p_fold,) if p_fold is not None else ())))
        ctx.fold_p = p_fold is not None
        ctx.post_f8, ctx.post_t, ctx.post_scale = post_f8, post_t, post_scale
        ctx.w2_f8_row, ctx.w2_scale, ctx.x_f8_t, ctx.x_scale = w2_f8_row, w2_scale, x_f8_t, x_scale
        ctx.w1_f8_col, ctx.w1_scale, ctx.g_scale = w1_f8_col, w1_scale, g_scale
        ctx.dpre_scale, ctx.dpre_amax = dpre_scale, dpre_amax
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        st = ctx.saved_tensors
        x, W2 = st[0], st[2]
        p_fold = st[3] if ctx.fold_p else None
        g = grad_output.reshape(-1, grad_output.shape[-1])
        # E5M2 grad and both layouts; original record #92 uses STE quantization.
        g8, gt8 = quantize_dual_layout_fused(g, ctx.g_scale, fmt=torch.float8_e5m2)
        dW2 = torch.ops.nanogpt.wg2_f8(ctx.post_t, gt8, ctx.post_scale, ctx.g_scale)
        grad_p = None
        if ctx.fold_p:
            grad_p = (dW2.float() * W2.float()).sum().to(p_fold.dtype)
            dW2 = dW2 * p_fold
        dpost = torch._scaled_mm(g8, ctx.w2_f8_row.T, out_dtype=torch.bfloat16,
                                 scale_a=ctx.g_scale, scale_b=ctx.w2_scale,
                                 use_fast_accum=False)
        dpre8, dpret8 = dpre_quant_sm121(dpost, ctx.post_f8, ctx.post_scale,
                                         ctx.dpre_scale, ctx.dpre_amax)
        dW1 = torch.ops.nanogpt.wg1_f8(dpret8, ctx.x_f8_t, ctx.dpre_scale, ctx.x_scale)
        dx = torch.ops.nanogpt.dx_f8(dpre8, ctx.w1_f8_col, ctx.dpre_scale, ctx.w1_scale)
        # Three differentiable inputs + sixteen cache/scale args + p_fold.
        return (dx.reshape(x.shape), dW1, dW2) + (None,) * 16 + (grad_p,)
