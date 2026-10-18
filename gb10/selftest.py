"""Fail-closed on-device verification for GB10 accelerator feature flags.

Runs BEFORE allocating the large distributed n-gram table; failures reject
training rather than corrupting a real run or silently using a slow fallback.
"""
import os
import unittest


def run_selected_tests():
    selected = []
    if os.environ.get("GB10_ATTN_BACKEND", "sdpa") in ("fa4", "hybrid"):
        selected.append("test_fa4_varlen_mixed_heads_forward_backward")
    if os.environ.get("GB10_FP8_LINEAR", "0") == "1":
        selected.append("test_fp8_projection_and_grads")
    if os.environ.get("GB10_FUSED_FP8_CE", "0") == "1":
        selected.append("test_fused_fp8_ce_value_gradient")
    suite = unittest.TestSuite()
    if selected:
        from gb10.test_accelerated import AcceleratedGB10
        for name in selected:
            suite.addTest(AcceleratedGB10(name))
    if os.environ.get("GB10_CUDA_GRAPHS", "0") == "1":
        from gb10.test_graph_attention import GraphAttentionReference
        suite.addTest(GraphAttentionReference("test_cuda_graph_replay_reads_live_cu_seqlens"))
        selected.append("test_cuda_graph_replay_reads_live_cu_seqlens")
    if os.environ.get("GB10_GRAM_BACKEND", "aten") == "triton":
        from gb10.test_tiles import TritonTileGB10
        for name in ("test_tiled_gram_xxt_xtx_and_poly", "test_tile_graph_replay_uses_live_input"):
            suite.addTest(TritonTileGB10(name))
            selected.append(name)
    if not selected:
        return
    print(f"[GB10 accelerator] verifying {selected} on this SM121 before training",flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("GB10 accelerated kernel verification failed/skipped; "
                           "disable the relevant GB10_* accelerator switch or fix the kernel")
