"""SM121 FP8 MLP verification and honest, standalone BF16-vs-FP8 microbenchmark.

On physical GB10, in Docker:
  python tools/gb10_fp8_mlp_probe.py --check-compile
  python tools/gb10_fp8_mlp_probe.py --bench --tokens 8192 --iters 10
  GB10_FP8_MLP_TILE_M=64 GB10_FP8_MLP_TILE_N=64 python tools/gb10_fp8_mlp_probe.py --bench

Benchmark timings exclude model context, FP8 weight-cache refresh, optimizer,
data loading and CE. They are NOT whole-training speedup predictions.
"""
import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from track_1_short.perf.kernels.mlp_sm121 import SM121FP8MLPFunction, TILE_M, TILE_N


def require_sm121():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("The SM121 backend must be validated on a physical GB10 (capability 12.1)")
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Use native ARM64 GB10 rather than x86 CUDA emulation")


def fixture(tokens, dim, hidden, seed=1234, folded=True):
    """Actual cache layout and all 16 scale/cache inputs expected by GPT._mlp_fp8_args."""
    torch.manual_seed(seed)
    dev = "cuda"
    x = (torch.randn(tokens, dim, device=dev, dtype=torch.bfloat16) * 0.4).requires_grad_()
    w1 = (torch.randn(hidden, dim, device=dev, dtype=torch.bfloat16) * 0.025).requires_grad_()
    w2 = (torch.randn(hidden, dim, device=dev, dtype=torch.bfloat16) * 0.025).requires_grad_()
    s_x = torch.tensor(2**-4, device=dev)
    s_g = torch.tensor(2**-6, device=dev)
    s_w1 = (w1.detach().abs().amax().float() / 448).clamp_min(1e-12).reshape(1)
    s_w2 = (w2.detach().abs().amax().float() / 448).clamp_min(1e-12).reshape(1)
    e4 = torch.float8_e4m3fn
    def q4(a, s):
        return (a.detach().float() / s).clamp(-448, 448).to(e4).contiguous()
    x8, a8, b8 = q4(x, s_x), q4(w1, s_w1), q4(w2, s_w2)
    p = torch.tensor(0.87, device=dev, requires_grad=True) if folded else None
    sw2 = s_w2 * p.detach() if folded else s_w2
    post_scale = torch.tensor(0.025, device=dev)
    dpre_scale = torch.tensor(0.025, device=dev)
    post_max = torch.zeros(1, device=dev)
    dpre_max = torch.zeros(1, device=dev)
    # GPT caches row & col-major views of identically quantized weight values.
    a8_col = a8.T.contiguous().T
    b8_col = b8.T.contiguous().T
    args = (a8, s_x * s_w1, x8, b8_col, sw2, post_scale, post_max,
            b8, sw2 * s_g, x8.T.contiguous(), s_x, a8_col, s_w1, s_g,
            dpre_scale, dpre_max) + ((p,) if folded else ())
    return x, w1, w2, p, args


def compare_one(tokens, dim, hidden, folded, mlp_function=SM121FP8MLPFunction):
    x, w1, w2, p, args = fixture(tokens, dim, hidden, folded=folded)
    y = mlp_function.apply(x, w1, w2, *args)
    g = (torch.randn_like(y) * 0.05).contiguous()
    variables = (x, w1, w2) + ((p,) if folded else ())
    actual = torch.autograd.grad(y, variables, g)
    (a8, _, x8, b8_col, sw2, sp, post_max, b8, _, x8t, sx, a8_col,
     sw1, sg, sdpre, dpre_max) = args[:16]

    # Independently reconstruct the exact FP8 tensorwise GEMMs, activation
    # rounding, delayed scale, e4m3/e5m2 clamps, and STE gradient contract.
    pre = torch._scaled_mm(x8, a8.T, out_dtype=torch.bfloat16,
                           scale_a=sx, scale_b=sw1, use_fast_accum=True)
    post = F.relu(pre).float().square().bfloat16()
    post8 = (post.float() / sp).clamp(0, 448).to(torch.float8_e4m3fn)
    expected = torch._scaled_mm(post8, b8_col, out_dtype=torch.bfloat16,
                                scale_a=sp, scale_b=sw2, use_fast_accum=True)
    torch.testing.assert_close(y, expected, atol=0.025, rtol=0.01)
    g8 = (g.float() / sg).clamp(-57344, 57344).to(torch.float8_e5m2)
    dw2 = torch._scaled_mm(post8.T.contiguous(), g8, out_dtype=torch.bfloat16,
                           scale_a=sp, scale_b=sg, use_fast_accum=False)
    dpost = torch._scaled_mm(g8, b8.T, out_dtype=torch.bfloat16,
                             scale_a=sg, scale_b=sw2, use_fast_accum=False)
    dpre = (2 * dpost.float() * torch.sqrt(post8.float() * sp)).bfloat16()
    dpre8 = (dpre.float() / sdpre).clamp(-57344, 57344).to(torch.float8_e5m2)
    dw1 = torch._scaled_mm(dpre8.T.contiguous(), x8, out_dtype=torch.bfloat16,
                           scale_a=sdpre, scale_b=sx, use_fast_accum=False)
    dx = torch._scaled_mm(dpre8, a8, out_dtype=torch.bfloat16,
                          scale_a=sdpre, scale_b=sw1, use_fast_accum=False)
    dp = (dw2.float() * w2.detach().float()).sum().to(p.dtype) if folded else None
    if folded:
        dw2 = dw2 * p.detach()
    expected_grads = (dx, dw1, dw2) + ((dp,) if folded else ())
    for label, got, want in zip(("dx", "dW1", "dW2", "dp"), actual, expected_grads):
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-8)
        assert torch.isfinite(got).all().item(), f"{label}: nonfinite gradients"
        assert error.item() < 0.045, f"{label}: FP8 contract mismatch: {error.item():.4f}"
    assert post_max.item() > 0 and dpre_max.item() > 0, "delayed amax must update"
    print(f"[ok] SM121 FP8 MLP {tokens=} {dim=} {hidden=} {folded=}: "
          "output, dx/dW1/dW2/dp, delayed amax")


def check_compiled(mlp_function=SM121FP8MLPFunction):
    x, w1, w2, p, args = fixture(128, 768, 2816)
    compiled = torch.compile(
        lambda x, w1, w2: mlp_function.apply(x, w1, w2, *args),
        fullgraph=True, dynamic=False)
    for _ in range(2):
        y = compiled(x, w1, w2)
        grads = torch.autograd.grad(y.sum(), (x, w1, w2))
        assert all(torch.isfinite(t).all().item() for t in grads)
    print("[ok] SM121 FP8 MLP torch.compile(fullgraph=True) forward/backward")


def bench(tokens, iters):
    x, w1, w2, p, args = fixture(tokens, 768, 2816)
    g = torch.randn_like(x)
    def bf16_step():
        y = F.linear(F.relu(F.linear(x, w1)).square(), w2.T)
        return torch.autograd.grad(y, (x, w1, w2), g)
    def fp8_step():
        y = SM121FP8MLPFunction.apply(x, w1, w2, *args)
        return torch.autograd.grad(y, (x, w1, w2), g)
    for label, step in (("bf16", bf16_step), ("fp8_sm121", fp8_step)):
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            start.record()
            step()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        print(f"{label}: median={times[len(times)//2]:.3f} ms/MLP fwd+bwd "
              f"(M={tokens},D=768,H=2816; no optimizer/refresh)")


def sweep(tokens: int, iters: int):
    """Run 6 candidate tiled kernels in isolated interpreters (process-static configs)."""
    variants = [(m, n) for m in (16, 32, 64) for n in (64, 128)]
    results = []
    for m, n in variants:
        env = os.environ.copy()
        env["GB10_FP8_MLP_TILE_M"] = str(m)
        env["GB10_FP8_MLP_TILE_N"] = str(n)
        command = [sys.executable, str(Path(__file__).resolve()), "--bench",
                   "--tokens", str(tokens), "--iters", str(iters)]
        p = subprocess.run(command, env=env, text=True, capture_output=True)
        print(f"\n--- tile M={m} N={n} exit={p.returncode} ---")
        print(p.stdout[-2000:])
        if p.returncode != 0:
            print(p.stderr[-3000:])
            results.append((float("inf"), m, n))
            continue
        match = re.search(r"fp8_sm121: median=([\d.]+) ms", p.stdout)
        if not match:
            raise RuntimeError(f"Missing benchmark result for tile M={m} N={n}")
        results.append((float(match.group(1)), m, n))
    print("\nSM121 FP8 MLP benchmark ranking (median ms, smaller is better):")
    for t, m, n in sorted(results):
        print(f"tile M={m:2} N={n:3}: {t:.3f} ms")
    finite = [r for r in results if r[0] != float("inf")]
    if not finite:
        raise RuntimeError("All candidate tiles failed; do not enable fp8_sm121 training")
    best = min(finite)
    print(f"Best tested tile: GB10_FP8_MLP_TILE_M={best[1]} "
          f"GB10_FP8_MLP_TILE_N={best[2]}; repeat for all training-stage sizes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-compile", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="six tile candidates in isolated processes; validates each before timing")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()
    require_sm121()
    if args.sweep:
        sweep(args.tokens, args.iters)
        return
    print(f"SM121 tiles M={TILE_M} N={TILE_N}; torch={torch.__version__}")
    compare_one(128, 128, 256, folded=False)
    compare_one(128, 128, 256, folded=True)
    compare_one(256, 768, 2816, folded=True)
    if args.check_compile:
        check_compiled()
    if args.bench:
        bench(args.tokens, args.iters)


if __name__ == "__main__":
    main()
