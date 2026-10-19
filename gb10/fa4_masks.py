"""Shared FA4/CuTe mask argument semantics for native call and isolator.

CUTLASS/FA4 canonicalizes finite causal left-window masks to local attention.
Use typed CuTe Int32 for both window endpoints in the legacy SM121 bundle
so the real path and diagnostic path exercise the SAME JIT specialization.
No CuTe dependency is imported for native SDPA or CPU reference tests.
"""


def _cute_int32(value):
    try:
        from cutlass.base_dsl.typing import Int32
    except ImportError:
        return int(value)
    return Int32(int(value))


def fa4_mask_kwargs(*, causal, window):
    """Map full causal/noncausal and local windows to FA4 kwargs.

    Finite causal window k is key indices [q-k, q] inclusive, exactly the
    convention used in gb10.attention.segmented_sdpa. For an explicit
    finite window, canonical FA4 uses local (causal=False, right=0), because
    upstream _resolve_causal_local_window canonicalizes causal+left similarly.
    """
    if window is None:
        return dict(causal=bool(causal),
                    window_size=(_cute_int32(-1), _cute_int32(-1)))
    if not isinstance(window, int) or window < 0:
        raise ValueError("FA4 local window must be a nonnegative integer or None")
    return dict(
        causal=False,
        window_size=(_cute_int32(window),
                     _cute_int32(0 if causal else window)),
    )
