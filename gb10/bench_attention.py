"""Microbenchmark the GB10-specific ragged SDPA path on a real sm_121.

Time the forward AND backward, not import-time/kernel warmup, and report
allocated unified GPU memory. These results must be measured on EACH GB10;
this script has not been validated on real sm_121 hardware here.
"""
import argparse
import statistics
import torch

from gb10.attention import segmented_sdpa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="128,320,768,1536")
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--qk", type=int, default=64)
    ap.add_argument("--v", type=int, default=128)
    ap.add_argument("--window", type=int, default=640)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark must run inside a CUDA-enabled GB10 container")
    cap = torch.cuda.get_device_capability()
    if cap != (12, 1):
        raise RuntimeError(f"Expected GB10 sm_121, found {cap}")
    lens = [int(v) for v in args.lengths.split(",")]
    ends = [0]
    for length in lens:
        if length < 1:
            raise ValueError("length must be positive")
        ends.append(ends[-1] + length)
    cum = torch.tensor(ends, dtype=torch.int32)  # CPU: no GPU synchronization at boundaries
    n = ends[-1]
    q = torch.randn(n, args.heads, args.qk, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(n, args.heads, args.v, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def iteration():
        y = segmented_sdpa(q, k, v, cum, args.window, 0.13)
        y.float().square().mean().backward()
        for t in (q, k, v):
            t.grad = None

    for _ in range(4):
        iteration()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.repeats):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        iteration()
        en.record()
        en.synchronize()
        times.append(st.elapsed_time(en))
    print(f"GPU={torch.cuda.get_device_name()}, CC={cap}, segments={lens}, "
          f"window={args.window}, head_dims=({args.qk},{args.v}), "
          f"mean_ms={statistics.mean(times):.3f}, p50_ms={statistics.median(times):.3f}, "
          f"allocated_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}")


if __name__ == "__main__":
    main()
