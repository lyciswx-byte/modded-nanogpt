# How to make one GB10's gradients match an eight-H100 data-parallel update

> **Status (2026-10-16): design and verified CPU reference math only.** The
> accumulation path is **not connected** to `train_gpt.py` and **no command-line
> switch currently enables it**. Running `bash run.sh` uses the prior
> smaller-batch / more-updates approximation. Do not claim the original global
> batch was reproduced until the engine changes below are implemented and tested.

## What "equivalent" means

The target is **the same effective training token count per optimizer update**
and the **original 1194 optimizer updates and optimizer/schedule events**. In
terms of gradient averaging, this is equivalent to *data parallelism* if the
same independent examples, per-rank losses, parameter state, and numerical
kernels are used. The port does **not** claim bitwise equivalence:

- Record #92's original hashed n-gram table has **84,602,880 × 768 BF16**
  entries (about 130 GB in decimal units). The GB10 port currently has only
  10,575,360 rows. Just the original weights exceed GB10's ~128 GB unified
  memory, before optimizer state, program memory or OS. Exact model parity
  requires external/offloaded storage or an accuracy-vs-memory trade-off.
- The original *sampled softmax candidate set is generated per rank*. A true
  eight-rank emulation needs **eight logical per-rank candidate sets**, with
  each candidate set built from the rank's **entire** original local batch and
  shared across its smaller microbatches. Using a fresh independent candidate
  set for each GB10 microbatch is *not* the same loss.
- Original H100 FP8 operations, CUDA graph execution order, and floating-point
  reduction order differ from Blackwell; gradient sums can be mathematically
  equivalent without bitwise equality.

## Exact effective-batch arithmetic

Eight H100s: each rank's sum-of-token-loss gradients are averaged across the
eight ranks by `dist.ReduceOp.AVG`.

GB10 default divisor 2: a microbatch has half the original per-H100 token
count; therefore **8 ranks × 2 microbatches/rank = 16 microbatches per update**.

| Stage | H100 per rank | Original global batch | GB10 microbatch | Microbatches/update |
|---|---:|---:|---:|---:|
| Stage 0 | 16,384 | 131,072 | 8,192 | 16 |
| Stage 1 | 32,768 | 262,144 | 16,384 | 16 |
| Stage 2 | 49,152 | 393,216 | 24,576 | 16 |
| Taper | 40,960 | 327,680 | 20,480 | 16 |
| Extension | 16,384 | 131,072 | 8,192 | 16 |

`GB10_MICROBATCH_DIVISOR=1` -> 8 microbatches per update;
`=2` -> 16; `=4` -> 32; `=8` -> 64.

```python
# Reference for SUM-reduced losses: StepGraphs._loss ends in .sum(), while
# original optimizer gathers use ReduceOp.AVG across eight physical ranks.
accumulated_gradient = sum(micro_loss_sum_grad for micro in micros) / 8
# Do not divide by 16 again; that halves the update's gradient.
```

The optimizer timeline must be **1122 scheduled + 52 growth + 20 extension
= 1194 updates**, not the current GB10 branch's **19,104 updates**. LR cooldown,
ANVIL rail engagement/warmup and n-gram period-4 switch all refer to this
**optimizer update index**, never the microbatch index.

## Why adding an ordinary .backward() loop is not enough

Record #92 does not use an ordinary `torch.optim.AdamW` training loop.
`track_1_short/perf/cuda_graphs/step_graphs.py` captures separate forward
and backward graphs; `backward()` currently *overwrites ANVIL .grad on
every replay*, while dense Adam gradients persist across two optimizer steps.
The table gradients are a different data structure.

Changes required before exposing a functioning accumulation mode:

1. **Logical data-parallel batch**: Have the loader issue eight virtual rank
   batches per logical optimizer step. Split each original per-rank batch into
   `GB10_MICROBATCH_DIVISOR` smaller microbatches while preserving BOS,
   target alignment, RoPE/attention segments, and n-gram hashes across split
   boundaries. The scheduler sees **one** optimizer step for the entire group.
2. **Dense graph gradients**: Change `StepGraphs.backward()` to accumulate
   ANVIL's (qk, vo, mlp) gradients into persistent buffers over the microbatch
   group. Do not update ANVIL velocities between micros. For Adam, preserve
   the existing two-*optimizer-step* gradient accumulation. Use FP32
   accumulation where possible and cast for the optimizer's existing BF16
   reduction. The original rank-averaging factor is `1/8` for each
   microbatch's unnormalized loss-sum gradient.
3. **Sampled softmax**: Reuse a distinct rank-local candidate set over the
   corresponding rank's microbatches. Rebuild sampled-target positions per
   microbatch but not negatives/candidate set. The existing
   `SampledSoftmax` assumes exactly one microbatch per optimizer step;
   its upload/WAR barrier and pinned buffer ownership must change.
4. **Value embeddings**: `ValueEmbedPull.grad_accum` is written directly by a
   custom backward into an FP16 sparse-row buffer. Scale each micro's
   contribution once by `1/8` **before** adding to the update-cycle buffer.
   Never multiply the persistent update-cycle gradient by `1/8` repeatedly.
5. **N-gram table**: `NgramTable.pending` holds one row-slot gradient per
   optimizer step, and its CUDA-graph sink buffers are keyed by microbatch
   size. Combine all microbatch slot gradients into a **single full logical
   step row-gradient contribution**, preserving row ordering and the two
   bigram/trigram halves. Only call `send_grads()` and row-sparse Adam on the
   original logical optimizer cadence (odd steps then period-4).
   Account for the original eight-rank `grad_mul` and `sq_mul` semantics.
6. **Optimizer step & weight refresh**: Invoke
   `training_manager.step_optimizers(logical_step)` **once** after all
   microbatch backpropagations. Update ANVIL's momentum rails, Adam's betas
   and weight decay, FP8 weight caches, tail averages and sparse pull cycles
   at logical-step boundaries, not at microbatch boundaries.
7. **CUDA Graph/prefetch sizing**: Captured step graphs use the *microbatch*
   shape. Row-prefetch wants and table caches must include the *whole
   logical update cycle*, potentially 16× more rows than the microbatch.
   Resize `ngram_grad_ring`, `NgramTable.cache` and pinned staging based
   on the logical maximum; inspect GB10 unified-memory residency and avoid
   growing every tensor to the global batch shape.
8. **Validation**: Compare small deterministic runs with the same inputs
   against an eight-rank reference. Check per-parameter gradient norms and
   sparse row updates *before* checking the final FineWeb loss.

### Pseudocode — NOT the current train_gpt.py implementation

```python
for update in range(1194):
    # All schedules and candidate count selection use "update".
    # The data group contains 8 virtual rank batches, each subdivided by 2.
    for virtual_rank in range(8):
        candidate_set = build_rank_candidate_set(virtual_rank, update)
        for micro in rank_microbatches(virtual_rank, update, divisor=2):
            per_token_loss_sum = model(micro, candidate_set).sum()
            # For the original sum-loss / AVG-across-ranks convention:
            accumulate_dense_sparse_grads(per_token_loss_sum, scale=1/8)

    # Exactly one update, not one update per microbatch!
    update_anvil_adam_sparse_tables(update)
    refresh_fp8_weights_and_step_graphs()
```

## What is available now

```bash
# Can run on a CPU-only development machine; this checks the accumulation
# arithmetic and an actual autograd gradient comparison, not the record trainer.
python tools/gb10_grad_accum_reference.py
python -m unittest tests.test_gb10_grad_accum_reference -v
```

See [GB10.md](GB10.md) for the current experimental single-GPU runner.
**No `GB10_GRAD_ACCUM_STEPS` / `GB10_EQUIVALENT` training switch has been
enabled: silently accepting the setting would pretend to reproduce the
eight-GPU batch without performing accumulation.**
