"""GPU-only FA4 backward fault isolation on GB10; never enables training.

Usage in the existing SM121 CUDA container, AFTER merging/pushing 4044024:
    python -m gb10.diagnose_fa4_backward --json gb10/results/fa4_backward.json

Progression: single-segment full noncausal -> causal -> multiple packed
segments -> sliding window -> mixed QK/V.  Report loss-free forward parity,
per-input gradients, repeatability and error location. An error of 1544 is a
hard FAIL; no tolerances are silently relaxed.

This tool accesses the *real* bundle through gb10.fa4.load_fa4(). In contrast
to gb10.test_accelerated, it can isolate the failing masking combination.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _stats(actual, reference):
    a = actual.detach().float()
    b = reference.detach().float()
    d = a - b
    norm = b.square().mean().sqrt().clamp_min(1e-8)
    abs_err = d.abs()
    flat = abs_err.flatten()
    worst = int(flat.argmax().item())
    return {
        "finite": bool(torch.isfinite(a).all().item()),
        "max_abs": float(abs_err.max().item()),
        "p99_abs": float(torch.quantile(flat, .99).item()),
        "rms_relative_to_ref": float((d.square().mean().sqrt() / norm).item()),
        "ref_rms": float(norm.item()),
        "ref_max_abs": float(b.abs().max().item()),
        "worst_flat_index": worst,
        "actual_at_worst": float(a.flatten()[worst].item()),
        "ref_at_worst": float(b.flatten()[worst].item()),
    }


def _reference(q, k, v, ends, *, causal, window, scale):
    pieces = []
    for lo, hi in zip(ends[:-1], ends[1:]):
        if hi <= lo:
            continue
        L = hi - lo
        qt = q[lo:hi].transpose(0,1)[None]
        kt = k[lo:hi].transpose(0,1)[None]
        vt = v[lo:hi].transpose(0,1)[None]
        if window is None:
            out = F.scaled_dot_product_attention(qt, kt, vt,
                is_causal=causal, dropout_p=0., scale=scale)
        else:
            index = torch.arange(L,device=q.device)
            delta = index[:,None] - index[None,:]
            mask = (delta >= 0) & (delta <= window) if causal else delta.abs() <= window
            out = F.scaled_dot_product_attention(qt, kt, vt,
                attn_mask=mask, dropout_p=0., scale=scale)
        pieces.append(out[0].transpose(0,1))
    return torch.cat(pieces,0)


def _real_fa4(q,k,v,ends,*,causal,window,scale):
    from gb10.fa4 import load_fa4, pad_mixed_qkv_for_fa4
    q,k,v = pad_mixed_qkv_for_fa4(q,k,v)
    offsets = torch.tensor(ends,device=q.device,dtype=torch.int32)
    max_len = max(b-a for a,b in zip(ends[:-1],ends[1:]))
    args = dict(
        cu_seqlens_q=offsets, cu_seqlens_k=offsets,
        max_seqlen_q=max_len, max_seqlen_k=max_len,
        causal=causal, softmax_scale=scale,
        window_size=(-1,-1) if window is None else (window,0 if causal else window),
    )
    out=load_fa4()(q.contiguous(),k.contiguous(),v.contiguous(),**args)
    if isinstance(out,(tuple,list)):
        out=out[0]
    return out


def run_case(name, lens, d_qk, d_v, causal, window, scale, *, repeats=2):
    device=torch.device("cuda")
    total=sum(lens)
    ends=[0]
    for n in lens:
        ends.append(ends[-1]+n)
    # Different random seeds and scalar magnitudes can reveal uninitialized
    # atomics and invalid masking; same data for forward/reference/replays.
    torch.manual_seed(total+17*d_qk+3*d_v)
    q=torch.randn(total,6,d_qk,device=device,dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    v=torch.randn(total,6,d_v,device=device,dtype=torch.bfloat16,requires_grad=True)
    upstream=torch.randn(total,6,d_v,device=device,dtype=torch.bfloat16)
    result={"case":name,"lens":lens,"d_qk":d_qk,"d_v":d_v,
            "causal":causal,"window":window,"scale":scale}
    try:
        reference=_reference(q,k,v,ends,causal=causal,window=window,scale=scale)
        ref_grads=torch.autograd.grad(reference,(q,k,v),upstream,retain_graph=False)
        saved=[]
        for i in range(repeats):
            out=_real_fa4(q,k,v,ends,causal=causal,window=window,scale=scale)
            grads=torch.autograd.grad(out,(q,k,v),upstream)
            torch.cuda.synchronize()
            per={"out":_stats(out,reference),
                 "dq":_stats(grads[0],ref_grads[0]),
                 "dk":_stats(grads[1],ref_grads[1]),
                 "dv":_stats(grads[2],ref_grads[2])}
            saved.append(([t.detach().clone() for t in grads],per))
        result["runs"]=[item[1] for item in saved]
        if len(saved)>1:
            result["repeat_grads"]={
                key:_stats(saved[1][0][j],saved[0][0][j])
                for j,key in enumerate(("dq","dk","dv"))
            }
        # A precise allclose rejection is intentional; downstream training
        # must remain on native SDPA when any of these gates fails.
        result["pass"]=all(
            m["finite"] and m["max_abs"] <= .12 + .12*m["ref_max_abs"]
            and m["rms_relative_to_ref"] <= .12
            for item in saved for key,m in item[1].items()
            if key in ("out","dq","dk","dv")
        )
    except Exception as exc:
        result["pass"]=False
        result["exception"]=f"{type(exc).__name__}: {exc}"
        import traceback
        result["traceback"]=traceback.format_exc()
    print(f"[FA4 isolate] {name} {'PASS' if result['pass'] else 'FAIL'} "
          f"({d_qk},{d_v}) lengths={lens} window={window}",flush=True)
    if "runs" in result:
        d=result["runs"][0]
        print("  "+", ".join(
            f"{key}: max={d[key]['max_abs']:.4g} "
            f"nrms={d[key]['rms_relative_to_ref']:.4g} "
            f"refmax={d[key]['ref_max_abs']:.4g}"
            for key in ("out","dq","dk","dv")
        ),flush=True)
    if "exception" in result:
        print(f"  error={result['exception']}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--json",default=None,
                        help="Optional output artifact path; never includes tensor contents")
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--scale",type=float,default=.13)
    parser.add_argument("--case",default="all",
                        help="Substring filter for case name, or 'all'")
    opts=parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(12,1):
        raise RuntimeError("The FA4 backward diagnostic needs a real sm_121 GB10")
    if opts.repeats < 2:
        raise ValueError("Use at least two repeats to detect gradient races")
    from gb10.cute_compat import cutlass_version
    cases=[
        ("dense-noncausal-64", (192,), 64,64, False,None),
        ("dense-causal-64", (192,),64,64,True,None),
        ("varlen-causal-64", (96,160,64,192),64,64,True,None),
        ("single-window-64", (192,),64,64,True,128),
        ("varlen-window-64", (96,160,64,192),64,64,True,128),
        ("varlen-window-mixed", (96,160,64,192),64,128,True,128),
        ("varlen-window-128", (96,160,64,192),128,128,True,128),
    ]
    selected=[spec for spec in cases if opts.case=="all" or opts.case in spec[0]]
    if not selected:
        raise ValueError(f"no FA4 diagnostic case matched {opts.case!r}")
    result={"torch":torch.__version__,"cuda":torch.version.cuda,
            "device":torch.cuda.get_device_name(0),
            "cutlass_dsl":cutlass_version(),
            "cases":[run_case(*case,opts.scale,repeats=opts.repeats) for case in selected]}
    if opts.json:
        dest=Path(opts.json)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    if not all(x["pass"] for x in result["cases"]):
        raise SystemExit("FA4 backward remains INVALID; do not train with this kernel")


if __name__=="__main__":
    main()
