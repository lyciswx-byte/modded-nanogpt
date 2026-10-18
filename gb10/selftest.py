"""Fail-closed on-device verification for GB10 accelerator feature flags.

Runs BEFORE allocating the large distributed n-gram table; failures reject
training rather than corrupting a real run or silently using a slow fallback.
"""
import os
import unittest


def run_selected_tests():
    selected = []
    if os.environ.get("GB10_ATTN_BACKEND", "sdpa") == "fa4":
        selected.append("test_fa4_varlen_mixed_heads_forward_backward")
    if os.environ.get("GB10_FP8_LINEAR", "0") == "1":
        selected.append("test_fp8_projection_and_grads")
    if os.environ.get("GB10_FUSED_FP8_CE", "0") == "1":
        selected.append("test_fused_fp8_ce_value_gradient")
    if not selected:
        return
    from gb10.test_accelerated import AcceleratedGB10
    suite = unittest.TestSuite(AcceleratedGB10(name) for name in selected)
    print(f"[GB10 accelerator] verifying {selected} on this SM121 before training",flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("GB10 accelerated kernel verification failed/skipped; "
                           "disable the relevant GB10_* accelerator switch or fix the kernel")
