"""Native SM121 checks for the adapted Hopper TensorDescriptor/TMA MLP.

Run with NANOGPT_TARGET=gb10 GB10_MLP_BACKEND=fp8 GB10_FP8_MLP_IMPL=hopper.
Optional --bench reports isolated eager operator timings, not training speedup.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from gb10_fp8_mlp_probe import compare_one, fixture, require_sm121
from gb10_training_probe import check_compiled_eval_mlp
from track_1_short.perf.kernels.mlp import (
    ACTIVE_FP8_MLP_IMPL, FusedLinearReLUSquareFunction as TmaMLP,
    linear_relu_square_kernel, prime_stage_cache,
)


def check_graph():
    x, w1, w2, p, args = fixture(144, 768, 2816, folded=True)
    g = torch.randn_like(x) * 0.05
    variables = (x, w1, w2, p)
    compiled = torch.compile(lambda x, w1, w2: TmaMLP.apply(x, w1, w2, *args),
                             dynamic=False, fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Compile before eager priming: the GB10 default must already be safe.
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = compiled(x, w1, w2)
            grads = torch.autograd.grad(y, variables, g)
        del y, grads
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = compiled(x, w1, w2)
        gradients = torch.autograd.grad(output, variables, g)
    previous = None
    for _ in range(2):
        with torch.no_grad():
            x.normal_(std=0.4)
            x8 = (x.float() / args[10]).clamp(-448, 448).to(torch.float8_e4m3fn)
            args[2].copy_(x8)
            args[9].copy_(x8.T)
            g.normal_(std=0.05)
            args[6].zero_()
            args[15].zero_()
        graph.replay()
        torch.cuda.synchronize()
        graph_amax = (args[6].clone(), args[15].clone())
        assert all(t.item() > 0 for t in graph_amax), "graph did not update delayed amax"
        args[6].zero_()
        args[15].zero_()
        expected = TmaMLP.apply(x, w1, w2, *args)
        expected_grads = torch.autograd.grad(expected, variables, g)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=0.025, rtol=0.01)
        for got, want in zip(gradients, expected_grads):
            torch.testing.assert_close(got, want, atol=0.025, rtol=0.01)
        for got, want in zip(graph_amax, (args[6], args[15])):
            torch.testing.assert_close(got, want)
        if previous is not None:
            assert not torch.equal(previous, output), "graph did not read live inputs"
        previous = output.detach().clone()
    print("[ok] TMA compiled forward/backward + CUDA graph replay: live FP8 inputs, gradients, folded gain, amax")


def run_checks():
    require_sm121()
    if (os.environ.get("NANOGPT_TARGET", "").lower() != "gb10"
            or os.environ.get("GB10_MLP_BACKEND") != "fp8" or ACTIVE_FP8_MLP_IMPL != "hopper"):
        raise RuntimeError("Set NANOGPT_TARGET=gb10 GB10_MLP_BACKEND=fp8 GB10_FP8_MLP_IMPL=hopper")
    torch.manual_seed(1234)
    check_graph()
    for tokens, hidden in ((128, 256), (144, 320), (256, 2816)):
        for folded in (False, True):
            compare_one(tokens, 768, hidden, folded, mlp_function=TmaMLP)
    for tokens in (8192, 24576):
        compare_one(tokens, 768, 2816, folded=True, mlp_function=TmaMLP)
    check_compiled_eval_mlp()
    prime_stage_cache()
    # Inspect the actual eager compiled variants, including BF16 forward, FP8
    # forward and FP8 backward. This is compiler metadata, not an SMEM estimate.
    kernels = linear_relu_square_kernel.device_caches[torch.cuda.current_device()][0].values()
    shared = sorted({kernel.metadata.shared for kernel in kernels})
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    assert shared and max(shared) <= limit, (shared, limit)
    print(f"[ok] TMA compiled shared memory bytes={shared}; hardware limit={limit}")


def benchmark(tokens, iters):
    from gb10_fp8_mlp_bench import _time
    x, w1, w2, _, args = fixture(tokens, 768, 2816, folded=False)
    g = torch.randn_like(x) * 0.05

    def tma_step():
        y = TmaMLP.apply(x, w1, w2, *args)
        return torch.autograd.grad(y, (x, w1, w2), g)

    def bf16_step():
        y = F.linear(F.relu(F.linear(x, w1)).square(), w2.T)
        return torch.autograd.grad(y, (x, w1, w2), g)

    bf16 = _time(bf16_step, reps=iters)
    tma = _time(tma_step, reps=iters)
    print(f"Isolated eager MLP forward+backward {tokens=} dim=768 hidden=2816: "
          f"BF16={bf16:.3f} ms, TMA FP8={tma:.3f} ms, ratio={bf16 / tma:.3f}x")
    print("Excludes model context, cache refresh, optimizer and data; NOT a whole-training speedup.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()
    if args.tokens <= 0 or args.tokens % 16 or args.iters <= 0:
        parser.error("tokens must be a positive multiple of 16; iters must be positive")
    run_checks()
    if args.bench:
        benchmark(args.tokens, args.iters)


if __name__ == "__main__":
    main()
