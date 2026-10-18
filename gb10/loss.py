"""Correctness-first PyTorch BF16 cross entropy for GB10 sm_121.

No H100 inline PTX e4m3 or NVRTC kernel is executed. Computes MTP +
prefix-token targets against either candidate or full vocabulary, using small
token slabs to bound LPDDR5X peak temporary memory. This is exact with respect
to the chosen candidate set, apart from BF16/FP32 rounding differences from
the original fused H100 CE.
"""
import os
import torch
import torch.nn.functional as F


def native_training_loss(x, targets, mtp_weights, prefix_targets, prefix_weight,
                         lm_head_weight, sampled_loss):
    """Return per-token losses, including shifted future targets and prefix CE."""
    T = x.shape[0]
    if sampled_loss is None:
        weight = lm_head_weight
        target_positions = targets
        prefix_positions = prefix_targets
    else:
        assert sampled_loss.candidate_ids is not None
        weight = lm_head_weight.index_select(1, sampled_loss.candidate_ids)
        target_positions = sampled_loss.target_pos[:T]
        prefix_positions = sampled_loss.prefix_pos[:T]
    if os.environ.get("GB10_FP8_LINEAR", "0") == "1":
        from gb10.fp8_linear import linear as gb10_linear
    else:
        # CPU/BF16 baseline should never import Triton/SM121 GPU kernels.
        gb10_linear = F.linear
    enable_fused = os.environ.get("GB10_FUSED_FP8_CE", "0") == "1"
    if enable_fused:
        from gb10.fused_fp8_ce import fused_loss
    slab = int(os.environ.get("GB10_CE_SLAB_ROWS", "128"))
    assert slab >= 1
    pieces = []
    for lo in range(0, T, slab):
        hi = min(T, lo + slab)
        logits = gb10_linear(x[lo:hi], weight.T)
        if enable_fused:
            # The Triton CE reads FP8 E4M3 logits and fuses softcap,
            # normalizer, MTP, prefix targets and BF16 logit gradients.
            pieces.append(fused_loss(logits.contiguous(), target_positions,
                                     prefix_positions, mtp_weights,
                                     prefix_weight, lo, T))
            continue
        logits = 23.0 * torch.sigmoid((logits + 5.0) / 7.5)
        logprob = F.log_softmax(logits.float(), dim=-1)
        length = hi - lo
        loss = torch.zeros(length, dtype=logprob.dtype, device=x.device)
        for k in range(mtp_weights.numel()):
            valid = max(0, min(length, T - lo - k))
            if valid:
                positions = target_positions[lo + k:lo + k + valid]
                values = -logprob[:valid].gather(1, positions[:, None]).squeeze(1)
                loss = loss + F.pad(values * mtp_weights[k], (0, length - valid))
        pp = prefix_positions[lo:hi]
        prefix_values = -logprob.gather(1, pp.clamp_min(0)[:, None]).squeeze(1)
        loss = loss + torch.where(pp >= 0, prefix_values * prefix_weight[0], 0.0)
        pieces.append(loss)
    return torch.cat(pieces)
