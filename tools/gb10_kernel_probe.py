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
    import triton
    print(f"triton={triton.__version__}, visible GPUs={torch.cuda.device_count()}")
    print(f"TRITON_PTXAS_PATH={os.environ.get('TRITON_PTXAS_PATH')}")
    print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}")
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


def attention_reference(q, k, v, counts, scale, window):
    """Independent FP32, document-by-document causal local attention."""
    refs, start = [], 0
    for length in counts:
        qs, ks, vs = [t[start:start + length].float().transpose(0, 1) for t in (q, k, v)]
        logits = qs @ ks.transpose(-2, -1) * scale
        row = torch.arange(length, device=q.device)
        allowed = (row[:, None] >= row[None, :]) & (row[:, None] - row[None, :] <= window)
        refs.append((logits.masked_fill(~allowed, -torch.inf).softmax(-1) @ vs).transpose(0, 1))
        start += length
    return torch.cat(refs)


def assert_attention(y, grads, q, k, v, counts, scale, window):
    ref = attention_reference(q, k, v, counts, scale, window)
    expected = torch.autograd.grad(ref.square().mean(), (q, k, v))
    torch.testing.assert_close(y.float(), ref, atol=0.04, rtol=0.04)
    for actual, reference in zip(grads, expected):
        torch.testing.assert_close(actual, reference, atol=2e-4, rtol=0.06)


def check_attention():
    prime_attention_backend()
    for qk_dim, v_dim in ((64, 128), (128, 64), (64, 64), (128, 128)):
        for paired in (False, True):
            counts = (10, 14) if paired else (5, 7)
            tokens, heads = sum(counts), 1 if paired else 2
            cu = torch.tensor((0, counts[0], tokens), device="cuda", dtype=torch.int32)
            q, k = [torch.randn((tokens, heads, qk_dim), device="cuda", dtype=torch.bfloat16,
                                requires_grad=True) for _ in range(2)]
            v = torch.randn((tokens, heads, v_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            scale = qk_dim ** -0.5
            y = varlen_attention(q, k, v, cu, max_len=max(counts), scale=scale, left_window=3)
            grads = torch.autograd.grad(y.float().square().mean(), (q, k, v))
            assert_attention(y, grads, q, k, v, counts, scale, 3)
            print(f"[ok] attention QK={qk_dim} V={v_dim} paired={paired}, output + gradient reference")


def check_compiled_attention(capture=False):
    """Fullgraph + replay must read live document boundaries and live batch values."""
    cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)
    q, k = [torch.randn((12, 2, 64), device="cuda", dtype=torch.bfloat16,
                        requires_grad=True) for _ in range(2)]
    v = torch.randn((12, 2, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(varlen_attention, dynamic=False, fullgraph=True)
    # Warm autograd on the capture stream: default-stream AccumulateGrad nodes
    # retained by the compiler cannot be imported into a capturing side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = compiled(q, k, v, cu, max_len=9, scale=0.125, left_window=3)
            gradients = torch.autograd.grad(y.float().square().mean(), (q, k, v))
        del y, gradients
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = compiled(q, k, v, cu, max_len=9, scale=0.125, left_window=3)
            grads = torch.autograd.grad(output.float().square().mean(), (q, k, v))
    for counts in ((5, 7), (7, 5), (3, 9)):
        with torch.no_grad():
            cu.copy_(torch.tensor([0, counts[0], 12], device="cuda", dtype=torch.int32))
            for t in (q, k, v):
                t.normal_()
        if graph is not None:
            graph.replay()
        else:
            output = compiled(q, k, v, cu, max_len=9, scale=0.125, left_window=3)
            grads = torch.autograd.grad(output.float().square().mean(), (q, k, v))
        torch.cuda.synchronize()
        assert_attention(output, grads, q, k, v, counts, 0.125, 3)
    print(f"[ok] compiled attention live batches + document boundaries, CUDA graph={capture}")


def check_bf16_mlp():
    from track_1_short.model.gpt import GPT
    x = torch.randn((128, 768), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1, w2 = [(torch.randn((2816, 768), device="cuda", dtype=torch.bfloat16) * 0.01).requires_grad_()
              for _ in range(2)]
    y = GPT._mlp(None, x, w1, w2, None)
    grads = torch.autograd.grad(y.float().square().mean(), (x, w1, w2))
    rx, rw1, rw2 = [t.detach().float().requires_grad_() for t in (x, w1, w2)]
    reference = F.relu(rx @ rw1.T).square() @ rw2
    rgrads = torch.autograd.grad(reference.square().mean(), (rx, rw1, rw2))
    from gb10_training_probe import relative_error
    relative_error(y, reference, 0.02)
    for g, ref in zip(grads, rgrads):
        relative_error(g, ref, 0.03)
    print("[ok] actual BF16 MLP vs FP32 reference: output L2 < 0.02, gradients L2 < 0.03")


def check_ngram_embedding():
    from track_1_short.perf.kernels.ngram_embed import ngram_embedding
    from track_1_short.ngram_table import BIGRAM_SIGN_MULS, TRIGRAM_SIGN_MULS
    tokens = 128
    cache = torch.randn((256, 768), device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(2 * tokens, device="cuda", dtype=torch.int32)
    ids[::2] = 0  # repeats and both valid lookup bounds
    signs = torch.randint(0, 2, (8192, 768), device="cuda").mul(2).sub(1).bfloat16()
    inputs = torch.randint(0, 50257, (tokens,), device="cuda", dtype=torch.int32)
    sink = torch.zeros((2 * tokens, 768), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = ngram_embedding(cache, ids, signs, inputs, sink)
    current = inputs.long()
    prev = F.pad(current[:-1], (1, 0))
    prev2 = F.pad(current[:-2], (2, 0))
    bigram = ((BIGRAM_SIGN_MULS[0] * current) ^ (BIGRAM_SIGN_MULS[1] * prev)) & 8191
    trigram = ((TRIGRAM_SIGN_MULS[0] * current) ^ (TRIGRAM_SIGN_MULS[1] * prev)
               ^ (TRIGRAM_SIGN_MULS[2] * prev2)) & 8191
    bigram[0] = 0
    trigram[:2] = 0
    sa, sb = signs[bigram], signs[trigram]
    reference = (cache[ids[:tokens].long()].float() * sa.float()
                 + cache[ids[tokens:].long()].float() * sb.float()).bfloat16()
    incoming = torch.randn_like(output)
    (grad,) = torch.autograd.grad(output, (sink,), incoming)
    expected_grad = torch.cat((incoming * sa, incoming * sb))
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)
    print("[ok] n-gram signed gather + gradient sink: repeated/boundary IDs, bit-exact reference")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--check-fp8-mlp", action="store_true")
    parser.add_argument("--skip-compiled-attention", action="store_true",
                        help="debug eager operators only (not representative of trainer)")
    parser.add_argument("--capture-attention", action="store_true",
                        help="also capture/replay the compiled attention backward graph")
    parser.add_argument("--benchmark-refresh", action="store_true",
                        help="measure isolated pre/post-change default-sized FP8 refresh work")
    args = parser.parse_args()
    require_gb10()
    torch.manual_seed(123)
    check_attention()
    if not args.skip_compiled_attention:
        check_compiled_attention(capture=args.capture_attention)
    if not args.attention_only:
        check_triton()
        check_fp8_matmul()
        check_bf16_mlp()
        check_ngram_embedding()
        from gb10_training_probe import check_ce, check_sparse_update, check_bf16_refresh, check_packed_qkv, check_anvil, check_ce_mtp_boundaries
        check_packed_qkv()
        check_ce()
        check_ce_mtp_boundaries()
        check_sparse_update()
        check_anvil()
        if gb10_mlp_backend() == "bf16":
            check_bf16_refresh()
    if args.check_fp8_mlp:
        from track_1_short.perf.kernels.mlp import prime_stage_cache
        prime_stage_cache()
        print("[ok] FP8 TMA MLP stage kernel launches")
    if args.benchmark_refresh:
        from gb10_training_probe import benchmark_refresh
        benchmark_refresh()
    torch.cuda.synchronize()
    print("GB10 operator probe passed; this is NOT an end-to-end training/convergence result.")


if __name__ == "__main__":
    main()
