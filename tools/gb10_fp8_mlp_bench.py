"""SM121 FP8 MLP numerical checks and BF16-vs-FP8 training-operator timings.

Run ONLY on a real GB10 after building Dockerfile.gb10:

  docker run --rm --gpus all --ipc=host \
    -e GB10_MLP_BACKEND=fp8 \
    -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
    modded-nanogpt:92-gb10 python tools/gb10_fp8_mlp_bench.py --benchmark

This calls the actual FusedLinearReLUSquareFunction, including its existing
FP8 downstream GEMMs and custom backward, rather than timing only the new fused
up GEMM. BF16 is the independent F.linear / relu-square reference. Timing is
single-operator eager GPU-event timing, NOT an end-to-end training speedup.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from track_1_short.perf.kernels.mlp import (
    ACTIVE_FP8_MLP_IMPL, FusedLinearReLUSquareFunction, quantize_dual_layout_fused,
)


def _bundle(m: int, n: int, k: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16).requires_grad_()
    w1 = (torch.randn((n, k), device=device, dtype=torch.bfloat16) / k ** 0.5).requires_grad_()
    w2 = (torch.randn((n, k), device=device, dtype=torch.bfloat16) / n ** 0.5).requires_grad_()
    dy = (torch.randn((m, k), device=device, dtype=torch.bfloat16) * 0.08).contiguous()

    # Same cache layout contract as GPT.init_mlp_fp8, but refresh weights eagerly
    # with exact scales so each correctness run is self-contained.
    e4 = torch.float8_e4m3fn
    one = lambda v: torch.tensor([v], dtype=torch.float32, device=device)
    xs, gs = one(1 / 16), one(1 / 64)
    w1s = w1.detach().float().abs().amax().mul(1.12 / 448).clamp_min(1e-9).reshape(1)
    w2s = w2.detach().float().abs().amax().mul(1.12 / 448).clamp_min(1e-9).reshape(1)
    w1q = (w1.detach().float() / w1s).clamp(-448, 448).to(e4).contiguous()
    w2q = (w2.detach().float() / w2s).clamp(-448, 448).to(e4).contiguous()
    w1col = w1q.T.contiguous().T
    w2col = w2q.T.contiguous().T
    xq, xt = quantize_dual_layout_fused(x.detach(), xs, e4)
    # Use a real exact bootstrap scale for this reference, rather than an
    # arbitrary 0.01 that would saturate legitimate post-ReLU-square tails.
    pre_ref = F.linear(x.detach(), w1.detach())
    post_ref = F.relu(pre_ref).square()
    ps = (post_ref.float().amax() * (1.03 / 448.0)).clamp_min(1e-6).reshape(1)
    dpre_ref = (F.linear(dy.detach(), w2.detach()) * 2 * F.relu(pre_ref)).float()
    ds = (dpre_ref.abs().amax() * (1.25 / 57344.0)).clamp_min(1e-7).reshape(1)
    pa, da = torch.zeros(1, device=device), torch.zeros(1, device=device)
    args = (w1q, xs * w1s, xq, w2col, w2s, ps, pa, w2q,
            gs * w2s, xt, xs, w1col, w1s, gs, ds, da)
    return x, w1, w2, dy, args, (pa, da)


def _run_fp8(x, w1, w2, dy, args):
    y = FusedLinearReLUSquareFunction.apply(x, w1, w2, *args)
    dx, dw1, dw2 = torch.autograd.grad(y, (x, w1, w2), dy)
    return y, (dx, dw1, dw2)


def _run_bf16(x, w1, w2, dy):
    # The project weight layout stores down projection [H,K]; the original
    # projection is post @ W2 (NOT F.linear(post, W2) with an extra transpose).
    y = F.linear(F.relu(F.linear(x, w1)).square(), w2.T)
    dx, dw1, dw2 = torch.autograd.grad(y, (x, w1, w2), dy)
    return y, (dx, dw1, dw2)


def _relative(actual, reference):
    actual, reference = actual.float(), reference.float()
    return float((actual - reference).norm() / reference.norm().clamp_min(1e-6))


def correctness():
    # Small and large-ish GEMMs exercise masked tails and both tensor layouts.
    for m, n, k in ((128, 256, 128), (192, 320, 128), (256, 2816, 768)):
        x, w1, w2, dy, args, (pa, da) = _bundle(m, n, k, seed=1234)
        y8, grads8 = _run_fp8(x, w1, w2, dy, args)
        y16, grads16 = _run_bf16(x, w1, w2, dy)
        out_error = _relative(y8, y16)
        grad_errors = tuple(_relative(a, b) for a, b in zip(grads8, grads16))
        assert torch.isfinite(y8).all().item()
        assert all(torch.isfinite(g).all().item() for g in grads8)
        assert pa.item() > 0 and da.item() > 0, "missing activation/dpre amax writes"
        # FP8 (especially e5m2 gradients) is deliberately approximate. Numeric
        # tolerances here are bring-up gates; tighter tolerances require
        # independent quantized-reference checks on the actual GB10.
        assert out_error < 0.30, f"FP8 output relative L2 {out_error:.4f} is too high"
        assert all(e < 0.45 for e in grad_errors), f"FP8 gradient relative L2 {grad_errors}"
        print(f"[ok] SM121 FP8 MLP M={m} N={n} K={k}: "
              f"output relL2={out_error:.4f}, gradients relL2={grad_errors}")
    print("Operator-level FP8 MLP numerical checks passed; NOT a training convergence result.")


def _time(op, reps: int = 12):
    for _ in range(4):
        op()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        op()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return sorted(samples)[len(samples) // 2]


def benchmark(tokens: int):
    from track_1_short.perf.kernels.mlp import prime_stage_cache
    # Auto-tuning MUST precede torch.compile/CUDA graph capture.
    prime_stage_cache()
    x, w1, w2, dy, args, _ = _bundle(tokens, 2816, 768, seed=5678)
    bf16 = _time(lambda: _run_bf16(x, w1, w2, dy))
    fp8 = _time(lambda: _run_fp8(x, w1, w2, dy, args))
    print(f"GB10 FP8 MLP ({ACTIVE_FP8_MLP_IMPL}) M={tokens} N=2816 K=768:")
    print(f"  BF16 F.linear + autograd:  {bf16:.3f} ms / MLP forward+backward")
    print(f"  SM121 FP8 fused + GEMMs:   {fp8:.3f} ms / MLP forward+backward")
    print(f"  BF16 / FP8 ratio:         {bf16 / fp8:.3f}x")
    print("Eager operator-level comparison, not full training; validate full-model CUDA "
          "graphs, loss stability, n-gram updates and final CE before enabling by default.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="also time BF16 vs FP8 full MLP fwd/bwd")
    parser.add_argument("--tokens", type=int, default=8192, help="training-shape batch tokens to benchmark")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("Requires one SM121 NVIDIA GB10 GPU")
    if os.environ.get("GB10_MLP_BACKEND") != "fp8" or ACTIVE_FP8_MLP_IMPL != "sm121":
        raise RuntimeError("Set GB10_MLP_BACKEND=fp8 and GB10_FP8_MLP_IMPL=sm121")
    correctness()
    if args.benchmark:
        benchmark(args.tokens)


if __name__ == "__main__":
    main()
