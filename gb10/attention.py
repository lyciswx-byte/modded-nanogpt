"""GB10 native attention baseline. No Hopper FA3 binary is imported.

Each segment has its own causal sliding-window SDPA operation. This avoids a
batch-wide NxN mask, supports qk_dim != v_dim, and defers GPU kernel choice to
the PyTorch/cuDNN build installed for sm_121. This is a functional baseline:
the native kernel, masks and sequence packing still need on-device profiling.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=32)
def _mask(length: int, left: int, device_type: str, device_index: int):
    p = torch.arange(length, device=torch.device(device_type, device_index))
    distance = p[:, None] - p[None, :]
    return (distance >= 0) & (distance <= left)


def qk_norm_rope_reference(qk, factor1, factor2, heads, rotary_dim, paired, key_offset):
    """Reference QK Norm/rotary, including the packed paired-head layout."""
    T, two_heads, D = qk.shape
    assert two_heads == 2 * heads and D % 2 == 0
    x = qk.reshape(T, 2, heads, D).float()
    rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1.1920928955078125e-7)
    flip = x.reshape(T, 2, heads, D // 2, 2).flip(-1).reshape_as(x)
    if paired:
        head_rot = torch.arange(heads, device=x.device) % 2
        cos = factor1.float().reshape(T, 2, D)[:, head_rot, :]
        sin = factor2.float().reshape(T, 2, D)[:, head_rot, :]
    else:
        cos = factor1.float()[:, None, :]
        sin = factor2.float()[:, None, :]
    rotated = (x * rstd) * cos[:, None] + (flip * rstd) * sin[:, None]
    if key_offset:
        assert not paired and D > rotary_dim
        normalized_keys = (x * rstd)[:, 1]
        prev = torch.cat([normalized_keys[:1], normalized_keys[:-1]], dim=0)
        keys = torch.cat([rotated[:, 1, :, :rotary_dim], prev[:, :, rotary_dim:]], dim=-1)
        rotated = torch.stack((rotated[:, 0], keys), dim=1)
    q, k = rotated.to(qk.dtype)[:, 0], rotated.to(qk.dtype)[:, 1]
    if paired:
        assert heads % 2 == 0
        q = q.reshape(T, 2, heads // 2, D).reshape(2 * T, heads // 2, D)
        k = k.reshape(T, 2, heads // 2, D).reshape(2 * T, heads // 2, D)
    return q, k


def segmented_sdpa(q, k, v, cum_seqlens, left_window: int, scale: float):
    """Exact document boundaries; never construct a global batch NxN mask.

    Padded cum_seqlens begins at zero and ends with duplicated terminal T.
    GB10's loader retains boundaries on the CPU so seven attention
    layers do not each pay GPU->CPU synchronization.
    """
    assert q.ndim == k.ndim == v.ndim == 3
    ends = cum_seqlens.tolist() if cum_seqlens.device.type == "cpu" else cum_seqlens.cpu().tolist()
    chunks = []
    start = 0
    for raw_end in ends[1:]:
        end = int(raw_end)
        if end <= start:
            continue
        L = end - start
        qq = q[start:end].transpose(0, 1).unsqueeze(0)
        kk = k[start:end].transpose(0, 1).unsqueeze(0)
        vv = v[start:end].transpose(0, 1).unsqueeze(0)
        mask = _mask(L, int(left_window), q.device.type, q.device.index or 0)
        out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask,
                                              dropout_p=0., is_causal=False, scale=float(scale))
        chunks.append(out.squeeze(0).transpose(0, 1))
        start = end
        if start == q.shape[0]:
            break
    assert start == q.shape[0], f"Packed sequence ended at {start}, not {q.shape[0]}"
    return torch.cat(chunks, dim=0)
