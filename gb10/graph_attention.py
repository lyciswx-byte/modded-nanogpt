"""CUDA-graph-compatible packed variable-length sliding attention for GB10.

No .item(), .tolist(), D2H, dynamic Python segments, FlashAttention SM90 .so,
or NCCL inside a captured forward/backward. All document boundaries remain
GPU tensors. Uses query tiles (128) and grouped SDPA over only the overlapping
key window, not a quadratic [T,T] full-batch attention mask. This is an
intermediate graph-safe implementation; actual SM121 latency must be measured.

Input Q/K [T,H,Dq], V [T,H,Dv], with qk=64 or 128 and v=64 or 128.
The paired-head path already doubles T and offsets before arriving here.
"""
import os
import torch
import torch.nn.functional as F


def graph_sdpa(q, k, v, cum_seqlens, left_window, scale):
    if q.is_cuda and cum_seqlens.device.type != "cuda":
        raise RuntimeError("GB10 CUDA Graph requires device-resident cu_seqlens, "
                           "not host-side segmented SDPA")
    if q.device != cum_seqlens.device:
        raise RuntimeError("cu_seqlens and Q must reside on the same GB10 GPU")
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[:2] == k.shape[:2] == v.shape[:2]
    assert q.shape[-1] in (64, 128) and v.shape[-1] in (64, 128)
    total, heads, dk = q.shape
    tile = int(os.environ.get("GB10_GRAPH_QUERY_TILE", "128"))
    group = int(os.environ.get("GB10_GRAPH_TILES_PER_GROUP", "4"))
    if tile not in (64, 128, 256) or not 1 <= group <= 16:
        raise ValueError("query tile must be 64/128/256, 1 <= group <= 16")
    # Each tile's K span is [tile_start-window, tile_end]; Q-specific
    # causality/window/document mask is computed on device every replay.
    keys_per_tile = int(left_window) + tile
    offsets = torch.arange(keys_per_tile, device=q.device, dtype=torch.long)
    query_offsets = torch.arange(tile, device=q.device, dtype=torch.long)
    chunks = []
    for base in range(0, total, group * tile):
        tile_count = min(group, (total - base + tile - 1) // tile)
        tile_starts = torch.arange(tile_count, device=q.device, dtype=torch.long) * tile + base
        q_pos = tile_starts[:, None] + query_offsets[None, :]
        k_pos = tile_starts[:, None] - int(left_window) + offsets[None, :]
        valid_q = q_pos < total
        valid_k = (k_pos >= 0) & (k_pos < total)
        q_index = q_pos.clamp(max=total-1).reshape(-1)
        k_index = k_pos.clamp(min=0, max=total-1).reshape(-1)

        # Dynamic segment IDs from *live* cu_seqlens. The same captured graph
        # can replay with different document boundaries within the same shape.
        # searchsorted(..., right=True) makes position exactly at a boundary
        # belong to the new segment, including duplicated padded endpoints.
        query_doc = torch.searchsorted(cum_seqlens.contiguous(), q_pos.contiguous(), right=True)
        key_doc = torch.searchsorted(cum_seqlens.contiguous(), k_pos.contiguous(), right=True)
        delta = q_pos[:, :, None] - k_pos[:, None, :]
        allowed = (
            valid_q[:, :, None] & valid_k[:, None, :]
            & (query_doc[:, :, None] == key_doc[:, None, :])
            & (delta >= 0) & (delta <= int(left_window))
        )
        # Shape [tiles,H,Q,Dq] and [tiles,H,K,Dq/Dv]
        q_tile = q.index_select(0, q_index).reshape(tile_count, tile, heads, dk).permute(0, 2, 1, 3)
        k_tile = k.index_select(0, k_index).reshape(tile_count, keys_per_tile, heads, dk).permute(0, 2, 1, 3)
        v_tile = v.index_select(0, k_index).reshape(tile_count, keys_per_tile, heads, v.shape[-1]).permute(0, 2, 1, 3)
        y = F.scaled_dot_product_attention(
            q_tile, k_tile, v_tile, attn_mask=allowed[:, None],
            dropout_p=0.0, is_causal=False, scale=float(scale),
        )
        y = y.permute(0, 2, 1, 3).reshape(tile_count * tile, heads, v.shape[-1])
        chunks.append(y[: min(total - base, tile_count * tile)])
    return torch.cat(chunks, dim=0)
