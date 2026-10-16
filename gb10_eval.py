"""Memory-bounded full-vocabulary Record #89 validation for GB10.

The attention graph still processes the original 262144 tokens / microbatch;
ONLY the final vocabulary projection and CE are split into smaller rows.
This avoids massive [T, 50304] temporary logits without changing document
boundaries or the validation stream.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def chunked_eval_ce(
    hidden: Tensor, targets: Tensor, head: nn.Module, *,
    chunk_rows: int = 2048,
) -> Tensor:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    if flat_hidden.shape[0] != flat_targets.numel():
        raise ValueError("hidden and targets have inconsistent token dimensions")
    losses = []
    for start in range(0, flat_targets.numel(), chunk_rows):
        logits = head(flat_hidden[start: start + chunk_rows])
        # Preserve Record #89's eval logit softcap exactly.
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        losses.append(F.cross_entropy(
            logits.float(), flat_targets[start: start + chunk_rows],
            reduction="none",
        ))
    return torch.cat(losses, dim=0).view_as(targets)
