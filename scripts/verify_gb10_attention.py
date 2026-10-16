#!/usr/bin/env python3
"""GB10 hardware acceptance test: compare FlexAttention varlen forward AND backward.

This test intentionally runs GPU kernels. Run it before the Record #89
2-step smoke test. It checks document isolation, causal left-window behavior,
BF16 forward output and Q/K/V gradients against a dense FP32 reference.
"""
from __future__ import annotations

import math
import sys

import torch

from gb10_attention import flash_attn_varlen_func


def reference(q, k, v, cuts: list[int], window: int, scale: float):
    t = q.shape[0]
    p = torch.arange(t, device=q.device)
    split = torch.tensor(cuts[1:], device=q.device)
    doc = torch.searchsorted(split, p, right=True)
    rows = p[:, None]
    cols = p[None, :]
    allow = ((doc[:, None] == doc[None, :])
             & (rows >= cols) & (rows - cols <= window))
    # [heads, query, key] scores
    score = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    score = score.masked_fill(~allow[None], float("-inf"))
    prob = torch.softmax(score, -1)
    out = torch.einsum("hts,shd->thd", prob, v.float())
    return out


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1):
        print("Requires NVIDIA GB10 SM121 CUDA device.", file=sys.stderr)
        return 2
    torch.manual_seed(123)
    cuts = [0, 37, 81, 147, 192]
    t, h, d = cuts[-1], 2, 64
    seq = torch.tensor(cuts + [192] * 6, dtype=torch.int32, device="cuda")
    scale = 1 / math.sqrt(d)
    for window in (32, 128, 384):
        inputs = [
            torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        ]
        y = flash_attn_varlen_func(
            *inputs,
            cu_seqlens_q=seq, cu_seqlens_k=seq,
            max_seqlen_q=192, max_seqlen_k=192,
            causal=True, softmax_scale=scale, window_size=(window, 0)
        )
        y_ref = reference(*inputs, cuts, window, scale)
        err = (y.float() - y_ref).abs().max().item()
        print(f"window={window} forward max_abs={err:.6f}")
        if not torch.isfinite(y).all() or err > 0.10:
            raise AssertionError(f"GB10 FlexAttention forward mismatch for window {window}")
        weight = torch.randn_like(y)
        grad = torch.autograd.grad(y, inputs, weight)
        grad_ref = torch.autograd.grad(y_ref, inputs, weight.float())
        for label, a, b in zip("qkv", grad, grad_ref):
            err = (a.float() - b).abs().max().item()
            print(f"window={window} d{label} max_abs={err:.6f}")
            if not torch.isfinite(a).all() or err > 0.15:
                raise AssertionError(f"GB10 backward mismatch on d{label}")
    print("[OK] Attention forward and backward matched the dense reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
