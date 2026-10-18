"""SM121 FA4 adapter, not the H100's patched FA3 binary.

The GB10 does not have SM100 tcgen05/TMEM. Use the downstream FA4/CuTe
SM120 warp-mma path *only* after smoke testing this workload's unequal
QK/V widths, varlen metadata, sliding windows and backward gradients.
FA4 is opt-in via GB10_ATTN_BACKEND=fa4; baseline remains native SDPA.
"""
import functools
import inspect

import torch


FA4_MODEL = "SecondNatureComputing/flash-attn-4-sm120"


@functools.lru_cache(maxsize=1)
def load_fa4():
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("SM121 FA4 adapter only supports the GB10 compute capability (12, 1)")
    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise RuntimeError("FA4 requires 'kernels', nvidia-cutlass-dsl and apache-tvm-ffi; "
                           "see gb10/README.md") from exc
    # The SM120 bundle contains an alternative warp-MMA implementation,
    # not an SM100 tcgen05/TMEM binary or SM90 WGMMA.
    kernel = get_kernel(FA4_MODEL, version=1)
    fn = kernel.flash_attn_varlen_func
    try:
        sig = inspect.signature(fn)
        parameters = sig.parameters
        needed = {"cu_seqlens_q", "cu_seqlens_k", "window_size", "softmax_scale"}
        if not needed.issubset(parameters) and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            raise RuntimeError(f"FA4 missing required varlen/sliding-window API: {sig}")
    except (TypeError, ValueError):
        # A pybind interface may not publish a Python signature; the smoke
        # checker in test_fa4.py will call the actual function.
        pass
    return fn


def packed_fa4(q, k, v, cum_seqlens, left_window, scale):
    """Varlen exact causal sliding-window FA4, with Dqk!=Dv preserved."""
    assert q.device.type == "cuda"
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[:2] == k.shape[:2] == v.shape[:2]
    assert q.shape[-1] in (64, 128) and v.shape[-1] in (64, 128)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    # The loader emits padded terminal endpoints. FA4 requires strictly
    # increasing offsets, so remove trailing duplicates on the CPU.
    cpu = cum_seqlens.cpu() if cum_seqlens.device.type != "cpu" else cum_seqlens
    # The validation loader may also begin with duplicate zero offsets
    # when the first token is BOS, not only trailing padding at T.
    ends = []
    for endpoint in cpu.tolist():
        if not ends or endpoint > ends[-1]:
            ends.append(endpoint)
    assert ends[0] == 0 and ends[-1] == q.shape[0], (ends[-1], q.shape[0])
    max_seqlen = max(b - a for a, b in zip(ends, ends[1:]))
    cu = torch.tensor(ends, dtype=torch.int32, device=q.device)
    fn = load_fa4()
    y = fn(q.contiguous(), k.contiguous(), v.contiguous(),
           cu_seqlens_q=cu, cu_seqlens_k=cu,
           max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
           causal=True, softmax_scale=float(scale),
           window_size=(int(left_window), 0))
    if isinstance(y, (tuple, list)):
        y = y[0]
    assert y.shape == v.shape, (y.shape, v.shape)
    return y


def dispatch(q, k, v, seqlens, window, scale):
    import os
    backend = os.environ.get("GB10_ATTN_BACKEND", "sdpa").lower()
    if backend == "graph":
        # GPU-resident dynamic document masks: same captured graph supports
        # different BOS boundaries at fixed token and cu_seqlens shapes.
        from gb10.graph_attention import graph_sdpa
        return graph_sdpa(q, k, v, seqlens, window, scale)
    if backend == "fa4":
        return packed_fa4(q, k, v, seqlens, window, scale)
    if backend == "hybrid":
        # FA4's launch overhead is measurable for tiny segments on SM121.
        # The threshold must be selected from actual GB10 forward+backward
        # timings, not Hopper benchmarks. No automatic fallback on FA4 error.
        cpu = seqlens.cpu() if seqlens.device.type != "cpu" else seqlens
        raw = cpu.tolist()
        positive = [b-a for a,b in zip(raw, raw[1:]) if b>a]
        threshold = int(os.environ.get("GB10_FA4_MIN_SEQLEN", "384"))
        if positive and max(positive) >= threshold:
            return packed_fa4(q, k, v, seqlens, window, scale)
    if backend in ("sdpa", "hybrid"):
        from gb10.attention import segmented_sdpa
        return segmented_sdpa(q, k, v, seqlens, window, scale)
    raise ValueError("GB10_ATTN_BACKEND must be 'sdpa', 'fa4', 'hybrid' or 'graph'")
