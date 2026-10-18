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
    kernel = get_kernel(FA4_MODEL)
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
    ends = cpu.tolist()
    valid = 1
    while valid < len(ends) and ends[valid] > ends[valid-1]:
        valid += 1
    ends = ends[:valid]
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
    if backend == "fa4":
        return packed_fa4(q, k, v, seqlens, window, scale)
    if backend == "sdpa":
        from gb10.attention import segmented_sdpa
        return segmented_sdpa(q, k, v, seqlens, window, scale)
    raise ValueError("GB10_ATTN_BACKEND must be 'sdpa' or 'fa4' (no silent kernel fallback)")
