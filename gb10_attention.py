"""Experimental GB10 Blackwell SM121 replacement for Hopper-only FA3.

Implements the subset of flash_attn_varlen_func used by Record #89:
unbatched [T,H,D] Q/K/V, cumulative document boundaries, causal left-window,
explicit attention scale. Uses PyTorch FlexAttention + a directly constructed
block-sparse mask so mask creation is O(T * window_blocks), not O(T**2).

This backend is NOT assumed numerically equivalent to FA3. Run the GPU
correctness/gradient tests and the full FineWeb validation before comparison.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

BLOCK_SIZE = 128


def window_block_plan(tokens: int, left_window: int, block_size: int = BLOCK_SIZE
                      ) -> tuple[int, int]:
    """Return (#query-blocks, max candidate key blocks per query block)."""
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if left_window < 0:
        raise ValueError("GB10 backend requires a finite causal left window")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return math.ceil(tokens / block_size), math.ceil(left_window / block_size) + 1


def make_varlen_block_mask(
    cu_seqlens: Tensor, tokens: int, left_window: int
) -> BlockMask:
    """Block-sparse candidate lists plus exact per-token document/causal mask."""
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be int32, matching the original loader")
    q_blocks, max_kv_blocks = window_block_plan(tokens, left_window)
    dev = cu_seqlens.device
    q_ids = torch.arange(q_blocks, device=dev, dtype=torch.int32)
    # Each query block can attend to current block and at most the blocks
    # reaching q_start-left_window; never allocate a quadratic dense mask.
    starts = (q_ids - (max_kv_blocks - 1)).clamp_min(0)
    counts = (q_ids + 1).clamp(max=max_kv_blocks)
    idx = starts[:, None] + torch.arange(max_kv_blocks, device=dev, dtype=torch.int32)[None, :]
    idx = idx[None, None]                 # [B=1,H=1,Q_BLOCKS,K_BLOCKS]
    counts = counts[None, None]           # [1,1,Q_BLOCKS]

    # Padding in cu_seqlens is at the final token count. torch.searchsorted
    # handles variable-length boundaries without a host synchronization.
    position = torch.arange(tokens, device=dev, dtype=torch.int32)
    doc_ids = torch.searchsorted(cu_seqlens[1:].contiguous(), position, right=True)

    def mask_mod(b: Tensor, h: Tensor, q: Tensor, k: Tensor) -> Tensor:
        # FlexAttention may evaluate mask_mod for padded tile indices.
        q_valid = q < tokens
        k_valid = k < tokens
        qi = q.clamp(min=0, max=tokens - 1)
        ki = k.clamp(min=0, max=tokens - 1)
        dist = q - k
        return (
            q_valid & k_valid
            & (doc_ids[qi] == doc_ids[ki])
            & (dist >= 0) & (dist <= left_window)
        )

    return BlockMask.from_kv_blocks(
        kv_num_blocks=counts,
        kv_indices=idx,
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(tokens, tokens),
    )


def _forward(q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor,
             left_window: int, scale: float) -> Tensor:
    length = q.shape[0]
    mask = make_varlen_block_mask(cu_seqlens, length, left_window)
    # flash_attn_varlen_func has [T,H,D], FlexAttention takes [B,H,T,D].
    y = flex_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        block_mask=mask,
        scale=scale,
    )
    return y.squeeze(0).transpose(0, 1).contiguous()


# Compiled entrypoint also works when called from an eager validation model.
# When already inside the Record #89 compiled trainer, call _forward directly
# rather than nesting torch.compile().
_compiled_forward = torch.compile(_forward, dynamic=False, fullgraph=False)


def flash_attn_varlen_func(
    q: Tensor, k: Tensor, v: Tensor,
    cu_seqlens_q: Tensor, cu_seqlens_k: Tensor,
    max_seqlen_q: int, max_seqlen_k: int,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    **kwargs,
) -> Tensor:
    """FA3 call-site compatible interface (limited to #89's causal self-attention)."""
    if kwargs:
        raise ValueError(f"Unsupported GB10 attention arguments: {sorted(kwargs)}")
    if not causal:
        raise ValueError("Only causal attention is implemented")
    if cu_seqlens_q is not cu_seqlens_k:
        raise ValueError("This backend supports self-attention only (shared cu_seqlens)")
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Expected q/k/v with identical [T,H,D] shapes")
    if window_size[1] != 0 or window_size[0] < 0:
        raise ValueError("GB10 attention needs (finite_left_window, 0)")
    # max_seqlen_* are FA3 metadata, not the packed token count; actual
    # document boundaries come from cu_seqlens.
    if max_seqlen_q != max_seqlen_k:
        raise ValueError("Self-attention max-seq lengths must match")
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    if torch.compiler.is_compiling():
        return _forward(q, k, v, cu_seqlens_q, window_size[0], scale)
    return _compiled_forward(q, k, v, cu_seqlens_q, window_size[0], scale)
