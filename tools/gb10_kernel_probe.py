"""GPU-only GB10 operator bring-up; runs before any FineWeb training.

Usage:
  NANOGPT_TARGET=gb10 NANOGPT_ATTENTION_BACKEND=fa2 python tools/gb10_kernel_probe.py
  NANOGPT_ATTENTION_BACKEND=fa4 python tools/gb10_kernel_probe.py --attention-only
  GB10_MLP_BACKEND=fp8 python tools/gb10_kernel_probe.py --check-fp8-mlp

Failures are fatal. Kernel import success alone does not prove that SM121 can
launch its forward/backward kernels, so each check executes real CUDA work.
"""
import argparse
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from track_1_short.perf.attention_backend import configured_backend, prime_attention_backend, varlen_attention
from track_1_short.perf.gb10_config import gb10_mlp_backend


def require_gb10():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible inside Docker; launch with --gpus all")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU={props.name} compute={cap}, RAM={props.total_memory / 2**30:.1f} GiB")
    print(f"host={platform.machine()}, torch={torch.__version__}, torch_cuda={torch.version.cuda}")
    print(f"torch arch list={torch.cuda.get_arch_list()}")
    print(f"attention={configured_backend()}, MLP={gb10_mlp_backend()}")
    if cap != (12, 1):
        raise RuntimeError(f"Expected GB10 sm121, got sm{cap[0]}{cap[1]}")
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("GB10 is a Grace ARM64 platform; use native linux/arm64 Docker")
    if torch.version.cuda is None or not torch.version.cuda.startswith("13."):
        raise RuntimeError("Expected NGC PyTorch with CUDA 13.x (do not pip install cu128)")
    return props


def check_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x, y, out, N: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(x + off, off < N, 0)
        b = tl.load(y + off, off < N, 0)
        tl.store(out + off, a + b, off < N)

    a = torch.arange(1024, device="cuda", dtype=torch.float32)
    b = torch.ones_like(a)
    out = torch.empty_like(a)
    add_kernel[(4,)](a, b, out, a.numel(), 256, num_warps=4)
    torch.testing.assert_close(out, a + b)
    print("[ok] Triton SM121 JIT + launch")


def check_fp8_matmul():
    a_bf = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16) / 4
    b_bf = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16) / 4
    a = a_bf.to(torch.float8_e4m3fn)
    b = b_bf.to(torch.float8_e4m3fn)
    one = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    out = torch._scaled_mm(a, b, out_dtype=torch.bfloat16,
                           scale_a=one, scale_b=one, use_fast_accum=True)
    reference = a.float() @ b.float()
    torch.testing.assert_close(out.float(), reference, atol=0.1, rtol=0.1)
    print("[ok] cuBLAS scaled FP8 GEMM")


def check_attention():
    prime_attention_backend()
    for qk_dim, v_dim in ((64, 128), (128, 64), (64, 64)):
        counts = (5, 7)
        cu = torch.tensor((0, 5, 12), device="cuda", dtype=torch.int32)
        q = torch.randn((12, 2, qk_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn((12, 2, qk_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn((12, 2, v_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        scale = qk_dim ** -0.5
        y = varlen_attention(q, k, v, cu, max_len=7, scale=scale, left_window=3)
        # Exact small reference: separate documents, causal + 3-token left window.
        refs = []
        start = 0
        for length in counts:
            qs = q[start:start + length].float().transpose(0, 1)
            ks = k[start:start + length].float().transpose(0, 1)
            vs = v[start:start + length].float().transpose(0, 1)
            logits = torch.matmul(qs, ks.transpose(-2, -1)) * scale
            row = torch.arange(length, device="cuda")
            allowed = (row[:, None] >= row[None, :]) & (row[:, None] - row[None, :] <= 3)
            logits = logits.masked_fill(~allowed.unsqueeze(0), -torch.inf)
            refs.append((logits.softmax(dim=-1) @ vs).transpose(0, 1))
            start += length
        ref = torch.cat(refs).to(y.dtype)
        torch.testing.assert_close(y, ref, atol=0.09, rtol=0.09)
        dq, dk, dv = torch.autograd.grad(y.float().square().mean(), (q, k, v))
        assert all(g.isfinite().all().item() for g in (dq, dk, dv)), "nonfinite attention gradients"
        torch.cuda.synchronize()
        print(f"[ok] varlen causal window attention QK={qk_dim} V={v_dim} forward/backward")


def check_compiled_attention(capture=False):
    """Test the same fullgraph compilation path used by the trainer."""
    cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
    q = torch.randn((12, 2, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((12, 2, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((12, 2, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(varlen_attention, dynamic=False, fullgraph=True)
    for _ in range(2):
        y = compiled(q, k, v, cu, max_len=7, scale=0.125, left_window=3)
        gradients = torch.autograd.grad(y.float().square().mean(), (q, k, v))
        assert all(g.isfinite().all().item() for g in gradients)
    torch.cuda.synchronize()
    print("[ok] torch.compile fullgraph attention forward/backward")
    if capture:
        graph = torch.cuda.CUDAGraph()
        # Real CUDA-graph capture: tests allocation/launch restrictions separately
        # from compile. All inputs are static and remain alive across replay.
        with torch.cuda.graph(graph):
            output = compiled(q, k, v, cu, max_len=7, scale=0.125, left_window=3)
            grads = torch.autograd.grad(output.float().square().mean(), (q, k, v))
        graph.replay()
        torch.cuda.synchronize()
        assert all(t.isfinite().all().item() for t in grads)
        print("[ok] attention forward/backward CUDA graph replay")


def check_bf16_mlp():
    x = torch.randn((128, 768), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn((2816, 768), device="cuda", dtype=torch.bfloat16, requires_grad=True) * 0.01
    w2 = torch.randn((768, 2816), device="cuda", dtype=torch.bfloat16, requires_grad=True) * 0.01
    y = F.linear(F.relu(F.linear(x, w1)).square(), w2)
    grads = torch.autograd.grad(y.float().square().mean(), (x, w1, w2))
    assert all(g.isfinite().all().item() for g in grads)
    print("[ok] portable BF16 MLP forward/backward")


def check_ngram_embedding():
    from track_1_short.perf.kernels.ngram_embed import ngram_embedding
    tokens = 128
    cache = torch.randn((256, 768), device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(2 * tokens, device="cuda", dtype=torch.int32)
    signs = torch.ones((8192, 768), device="cuda", dtype=torch.bfloat16)
    inputs = torch.arange(tokens, device="cuda", dtype=torch.int32)
    sink = torch.zeros((2 * tokens, 768), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = ngram_embedding(cache, ids, signs, inputs, sink)
    (grad,) = torch.autograd.grad(output.float().sum(), (sink,))
    assert torch.isfinite(output).all().item() and torch.isfinite(grad).all().item()
    print("[ok] n-gram custom Triton gather + grad sink")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--check-fp8-mlp", action="store_true")
    parser.add_argument("--skip-compiled-attention", action="store_true",
                        help="debug eager operators only (not representative of trainer)")
    parser.add_argument("--capture-attention", action="store_true",
                        help="also capture/replay the compiled attention backward graph")
    args = parser.parse_args()
    require_gb10()
    check_attention()
    if not args.skip_compiled_attention:
        check_compiled_attention(capture=args.capture_attention)
    if not args.attention_only:
        check_triton()
        check_fp8_matmul()
        check_bf16_mlp()
        check_ngram_embedding()
        if not hasattr(torch.cuda, "_compile_kernel"):
            raise RuntimeError("NGC torch lacks torch.cuda._compile_kernel used by record #92 CE; requires port")
        print("[ok] CUDA cross-entropy compile API present (not full CE execution)")
    if args.check_fp8_mlp:
        from track_1_short.perf.kernels.mlp import prime_stage_cache
        prime_stage_cache()
        print("[ok] FP8 TMA MLP stage kernel launches")
    torch.cuda.synchronize()
    print("GB10 operator probe passed; this is NOT an end-to-end training/convergence result.")


if __name__ == "__main__":
    main()
