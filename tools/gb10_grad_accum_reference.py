"""Reference math for 8-way data-parallel gradient accumulation on one GB10.

NOT wired into the record #92 training engine yet. That engine's CUDA graph,
sparse n-gram, value embedding, and sampled-softmax logic currently assumes
exactly one forward/backward per optimizer step; changing only the schedule
would silently produce a non-equivalent training run.

The original kernel returns per-token losses and StepGraphs sums them. The
distributed optimizer ReduceOp.AVG computes sum(rank_loss_grads)/8. Therefore
16 GB10 microbatches at divisor=2 must sum gradients with factor 1/8 per
microbatch, *not* 1/16. With mean-reduced losses the factor would differ.
"""
from dataclasses import dataclass

ORIGINAL_RANKS = 8
ORIGINAL_OPTIMIZER_STEPS = 1122 + 52 + 20
ORIGINAL_BATCHES_PER_RANK = (16384, 32768, 49152, 40960, 16384)
STAGE_NAMES = ("stage0", "stage1", "stage2", "taper", "extension")


@dataclass(frozen=True)
class EffectiveBatch:
    stage: str
    microbatch_tokens: int
    microbatches_per_update: int
    global_tokens_per_update: int
    gradient_factor: float


def accumulation_plan(microbatch_divisor: int = 2) -> tuple[EffectiveBatch, ...]:
    if microbatch_divisor not in (1, 2, 4, 8):
        raise ValueError("microbatch_divisor must be 1, 2, 4 or 8")
    microbatches = ORIGINAL_RANKS * microbatch_divisor
    scale = 1.0 / ORIGINAL_RANKS  # original rank-mean gradient reduction
    stages = []
    for stage, local_batch in zip(STAGE_NAMES, ORIGINAL_BATCHES_PER_RANK):
        micro_tokens = local_batch // microbatch_divisor
        assert micro_tokens * microbatches == local_batch * ORIGINAL_RANKS
        stages.append(EffectiveBatch(stage, micro_tokens, microbatches,
                                     local_batch * ORIGINAL_RANKS, scale))
    return tuple(stages)


def reference_check() -> None:
    """A real autograd test of sum-of-loss-gradients /8 vs 16 sequential microbatches."""
    import torch
    torch.manual_seed(92)
    inputs = torch.randn(96, 6, dtype=torch.float64)
    target = torch.randn(96, 4, dtype=torch.float64)
    weight = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)

    def loss_sum(x, y):
        return (x @ weight.T - y).square().sum()

    expected, = torch.autograd.grad(loss_sum(inputs, target) / ORIGINAL_RANKS, weight)
    accumulated = torch.zeros_like(expected)
    for x, y in zip(inputs.split(6), target.split(6)):
        grad, = torch.autograd.grad(loss_sum(x, y), weight)
        accumulated.add_(grad, alpha=1.0 / ORIGINAL_RANKS)
    torch.testing.assert_close(accumulated, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    for plan in accumulation_plan():
        print(f"{plan.stage:10} {plan.microbatch_tokens:6} tokens x "
              f"{plan.microbatches_per_update:2} -> {plan.global_tokens_per_update:6} "
              f"tokens/update; grad scale={plan.gradient_factor:g}")
    print("Original optimizer updates:", ORIGINAL_OPTIMIZER_STEPS)
    reference_check()
    print("CPU autograd reference: PASS (does not validate GB10 training engine)")
