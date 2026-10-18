"""GB10 SM121 fused FP8-logit softcapped multi-target cross entropy.

Consumes FP8 E4M3 logits with a per-slab FP32 scale. Triton kernels:
  1) tiled softcap/sum(exp(softcap-23)) per row;
  2) loss + row normalization over tile sums;
  3) backward directly creates the BF16 dLogits, including MTP duplicates
     and prefix targets, without ever constructing a probability tensor.

Uses fixed softcap 23*sigmoid((logits+5)/7.5), as #92. Quantization is
deliberate and introduces a new approximation vs BF16 CE; NOT benchmark-
equivalent until loss/gradient parity and convergence tests pass.

The LM projection can independently use sm_121 FP8 matmul from fp8_linear.py.
"""
import torch
import triton
import triton.language as tl

from gb10.fp8_linear import quantize_e4m3


@triton.jit
def _z(raw):
    p=1.0/(1.0+tl.exp(-(raw+5.0)/7.5))
    return 23.0*p,p


@triton.jit
def _partials(X,Scale,Out,V:tl.constexpr,CHUNKS:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(0)
    c=tl.program_id(1)
    offs=c*BK+tl.arange(0,BK)
    scale=tl.load(Scale)
    raw=tl.load(X+r*V+offs,offs<V,other=0).to(tl.float32)*scale
    z,_=_z(raw)
    e=tl.exp(z-23.)
    tl.store(Out+r*CHUNKS+c,tl.sum(tl.where(offs<V,e,0.0),0))


@triton.jit
def _finalize(X,Scale,Part,Logz,Loss,Targets,Prefix,Weights,PrefixWeight,
              V:tl.constexpr,CHUNKS:tl.constexpr,CPOW:tl.constexpr,
              BASE:tl.constexpr,TOTAL:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0)
    global_row=BASE+row
    offs=tl.arange(0,CPOW)
    sums=tl.load(Part+row*CHUNKS+offs,offs<CHUNKS,other=0.0)
    logz=23.+tl.log(tl.sum(sums,0))
    tl.store(Logz+row,logz)
    scale=tl.load(Scale)
    loss=tl.full((),0.,tl.float32)
    for k in tl.static_range(K):
        valid=global_row+k<TOTAL
        idx=tl.load(Targets+global_row+k,valid,other=0)
        raw=tl.load(X+row*V+idx,valid,other=0).to(tl.float32)*scale
        z,_=_z(raw)
        w=tl.load(Weights+k)
        loss+=tl.where(valid,w*(logz-z),0.)
    pfx=tl.load(Prefix+global_row)
    pw=tl.load(PrefixWeight)
    rawp=tl.load(X+row*V+tl.maximum(pfx,0),pfx>=0,other=0).to(tl.float32)*scale
    zp,_=_z(rawp)
    loss+=tl.where(pfx>=0,pw*(logz-zp),0.)
    tl.store(Loss+row,loss)


@triton.jit
def _backward(X,Scale,Logz,Targets,Prefix,Weights,PrefixWeight,G,Out,
              V:tl.constexpr,BASE:tl.constexpr,TOTAL:tl.constexpr,
              K:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(0)
    c=tl.program_id(1)
    off=c*BK+tl.arange(0,BK)
    mask=off<V
    row=BASE+r
    raw=tl.load(X+r*V+off,mask,other=0).to(tl.float32)*tl.load(Scale)
    z,p=_z(raw)
    norm=tl.load(Logz+r)
    weight_sum=tl.full((),0.,tl.float32)
    correction=tl.full((BK,),0.,tl.float32)
    for k in tl.static_range(K):
        valid=row+k<TOTAL
        target=tl.load(Targets+row+k,valid,other=0)
        wk=tl.load(Weights+k)
        weight_sum+=tl.where(valid,wk,0.)
        correction+=tl.where(valid & (off==target),wk,0.)
    pref=tl.load(Prefix+row)
    pref_w=tl.load(PrefixWeight)
    weight_sum+=tl.where(pref>=0,pref_w,0.)
    correction+=tl.where((pref>=0) & (off==pref),pref_w,0.)
    prob=tl.exp(z-norm)
    dz=(23.0/7.5)*p*(1.0-p)
    upstream=tl.load(G+r)
    dx=upstream*(weight_sum*prob-correction)*dz
    tl.store(Out+r*V+off,dx,mask)


class _FusedFP8CE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,targets,prefix_positions,mtp_weights,prefix_weight,start,total):
        assert logits.is_cuda and logits.ndim==2 and logits.dtype==torch.bfloat16
        assert logits.is_contiguous()
        rows,vocab=logits.shape
        assert mtp_weights.numel() in (1,2,3)
        assert 0 <= start < total and start + rows <= total
        assert prefix_positions.numel() == targets.numel() == total
        assert targets.dtype==prefix_positions.dtype==torch.int64
        q,s=quantize_e4m3(logits)
        from gb10.triton_tiles import ce_block
        BK=ce_block(vocab)
        chunks=triton.cdiv(vocab,BK)
        partial=torch.empty((rows,chunks),device=logits.device,dtype=torch.float32)
        logz=torch.empty((rows,),device=logits.device,dtype=torch.float32)
        loss=torch.empty_like(logz)
        _partials[(rows,chunks)](q,s,partial,vocab,chunks,BK,num_warps=4)
        _finalize[(rows,)](q,s,partial,logz,loss,targets,prefix_positions,
                           mtp_weights,prefix_weight,vocab,chunks,
                           triton.next_power_of_2(chunks),start,total,mtp_weights.numel(),
                           num_warps=4)
        ctx.save_for_backward(q,s,logz,targets,prefix_positions,mtp_weights,prefix_weight)
        ctx.shape=logits.shape
        ctx.start=start
        ctx.total=total
        return loss

    @staticmethod
    def backward(ctx,grad_loss):
        q,s,logz,targets,pref,weights,pref_w=ctx.saved_tensors
        rows,vocab=ctx.shape
        dx=torch.empty((rows,vocab),dtype=torch.bfloat16,device=q.device)
        from gb10.triton_tiles import ce_block
        bk=ce_block(vocab)
        _backward[(rows,triton.cdiv(vocab,bk))](
            q,s,logz,targets,pref,weights,pref_w,grad_loss.contiguous(),dx,
            vocab,ctx.start,ctx.total,weights.numel(),bk,num_warps=4)
        return dx,None,None,None,None,None,None


def fused_loss(logits,targets,prefix_positions,mtp_weights,prefix_weight,start,total):
    return _FusedFP8CE.apply(logits,targets,prefix_positions,mtp_weights,prefix_weight,start,total)
