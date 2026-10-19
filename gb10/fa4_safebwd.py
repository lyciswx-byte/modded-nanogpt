"""Opt-in diagnostic bridge: FA4 forward, mathematically checked SDPA backward.

The third-party SM120 CuTe bundle can JIT-compile but has shown dQ/dK/dV
divergence in the #92 mixed-QK/V ragged sliding-window workload. This bridge
NEVER invokes the CuTe backward kernel: it validates the FA4 forward against
native SDPA and recomputes native SDPA under autograd in backward.

This is intentionally SLOW: it computes an extra forward during both forward
verification and backward. It is useful for fault isolation and small training
smokes, NOT an optimized production FA4 implementation. It does not prove that
CuTe backward has been fixed and is excluded from CUDA Graph capture.
"""
import torch

from gb10.attention import segmented_sdpa


class _FA4ForwardSDPABackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, left_window, scale):
        from gb10.fa4 import packed_fa4
        # Function.forward is intentionally autograd-disabled. Forward kernels
        # still execute; the bundled third-party backward will never be called.
        fa4_output = packed_fa4(q, k, v, cu_seqlens, left_window, scale)
        reference = segmented_sdpa(q, k, v, cu_seqlens, left_window, scale)
        torch.testing.assert_close(
            fa4_output.float(), reference.float(), atol=.085, rtol=.06,
            msg="FA4 forward is not equivalent to native SDPA; refusing safe-backward mode",
        )
        ctx.save_for_backward(q, k, v, cu_seqlens)
        ctx.left_window = int(left_window)
        ctx.scale = float(scale)
        return fa4_output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, cu_seqlens = ctx.saved_tensors
        with torch.enable_grad():
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            out = segmented_sdpa(qq, kk, vv, cu_seqlens,
                                 ctx.left_window, ctx.scale)
            dq, dk, dv = torch.autograd.grad(
                out, (qq, kk, vv), grad_output.contiguous(),
            )
        return dq, dk, dv, None, None, None


def forward_fa4_backward_sdpa(q, k, v, cu_seqlens, left_window, scale):
    """Debug only. Default and production continue to use native SDPA."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FA4+SDPA diagnostic backward cannot enter CUDA graphs")
    return _FA4ForwardSDPABackward.apply(
        q, k, v, cu_seqlens, int(left_window), float(scale)
    )
