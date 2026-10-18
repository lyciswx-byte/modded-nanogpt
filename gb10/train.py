"""2-node GB10 trainer: correctness-first record-#92 algorithm port.

Launch through gb10/run_two_nodes.sh on each machine. GB10_PROFILE is set
before any stock imports, so the H100 patched FA3 .so is never loaded.
No H100 CUDA graphs/FP8/TMA kernels are used in the training forward/loss.
The n-gram, value-embedding and ANVIL Triton paths remain to be validated on
real sm_121 machines. This program intentionally does NOT claim record timing.
"""
import os
os.environ["GB10_PROFILE"] = "1"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
import time
import torch
import torch.distributed as dist

from gb10.eager import EagerStepRunner, NoFp8Refresh
from gb10.preflight import check_runtime, memory_plan
from track_1_short.canonical_mask import BackgroundCanonicalMask
from track_1_short.config import (
    Hyperparameters, TRAINING_STAGES, LR_COOLDOWN_FRAC,
    SPLIT_EMBED_STAGE, WS_POST_YARN_EXT,
)
from track_1_short.data import ScheduledBatches, cu_seqlens_rows, distributed_data_generator
from track_1_short.distributed import setup_distributed
from track_1_short.model.gpt import GPT
from track_1_short.model.prefix_prediction import build_prefix_table_bucket
from track_1_short.ngram_table import MAX_CYCLE_STEPS, NGRAM_DIM, NgramTable, is_update_step
from track_1_short.optim.anvil import anvil_bank_update
from track_1_short.perf.deferred_gathers import DEFERRED_LABELS, DeferredGathers
from track_1_short.perf.pinned_batches import PinnedBatchStaging
from track_1_short.perf.row_prefetch import PrefetchStaging, RowPrefetch
from track_1_short.perf.value_embed_pull import ValueEmbedPull
from track_1_short.sampled_softmax import SampledSoftmax
from track_1_short.schedule import TrainingSchedule
from track_1_short.tail_average import TailAverages
from track_1_short.training import TrainingManager


def gb10_step(manager, step_runner, row_prefetch, sampler, batches, gathers, step, model, has_next):
    batch = batches.peek(step)
    slots = row_prefetch.forward_slots(step, batch.ngram_ids)
    sampler.upload(step, batch.targets_cpu)
    gathers.flush()
    sample = sampler.gather(step, model.lm_head_f8_col)
    step_runner.forward(step, batch, slots, manager.get_forward_args(sample))
    row_prefetch.after_forward(step)
    grad = step_runner.backward()
    row_prefetch.ngram.accumulate_grad(
        slots, step_runner.hold_ngram_grad(grad, len(row_prefetch.ngram.pending), row_prefetch.is_update(step))
    )
    sampler.mark_readers_done()
    row_prefetch.post_prep(step)
    if has_next:
        sampler.prefetch(step + 1, batches.peek(step + 1).targets_cpu)
    out = manager.step_optimizers(step, row_prefetch, DEFERRED_LABELS)
    gathers.hand_over(out, refresh_lm=manager.is_adam_step(step))
    batches.take(step)


def main():
    check_runtime()
    args = Hyperparameters()
    env = setup_distributed()
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)
    # This is a conservative first pass, NOT a benchmark-matched 8xH100 run.
    max_step = max(s.batch_size for s in TRAINING_STAGES) // env.world_size
    val_step = args.val_batch_size // env.world_size
    memory_plan(env.device, max_step, val_step)
    schedule = TrainingSchedule(
        TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations,
        device=env.device, cooldown_frac=LR_COOLDOWN_FRAC,
        split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT,
    )
    stop = int(os.environ.get("GB10_SMOKE_STEPS", "0"))
    run_steps = schedule.total_steps if not stop else min(stop, schedule.total_steps)
    if run_steps <= 0:
        raise ValueError("GB10_SMOKE_STEPS must be positive")
    print0 = lambda *a, **kw: print(*a, flush=True, **kw) if env.master_process else None
    print0(f"[GB10] 2x sm_121, eager BF16/SDPA, train_steps={run_steps}/{schedule.total_steps}, "
           f"local_tokens={max_step}, local_val_tokens={val_step}")

    staging = PinnedBatchStaging(max(max_step, val_step),
                                max(cu_seqlens_rows(max_step), cu_seqlens_rows(val_step)),
                                env.device)
    train_loader = lambda: distributed_data_generator(
        args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, staging
    )
    val_loader = lambda: distributed_data_generator(args.val_files, args.val_batch_size, -1, staging,
                                                     align_to_bos=False)

    model = GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768,
                max_seq_len=max(val_step, MAX_CYCLE_STEPS * max_step),
                ngram_dim=NGRAM_DIM, world_size=env.world_size,
                device=env.device).cuda()
    model.cast_matrix_weights_bf16()
    for p in model.parameters():
        dist.broadcast(p.detach(), src=0)
    dist.broadcast(model.ngram_sign_pool, src=0)
    table = NgramTable(model.ngram_cache, max_step_tokens=max_step,
                       max_events=schedule.total_steps, rank=env.rank,
                       world_size=env.world_size, device=env.device)
    prefetch_staging = PrefetchStaging(
        max_ngram_rows=2 * MAX_CYCLE_STEPS * max_step,
        max_value_embed_rows=model.value_embeds.shape[0], device=env.device,
    )
    sampler = SampledSoftmax(schedule, model.vocab_size,
                             model_dim=model.lm_head.in_features,
                             max_rows=max_step, rank=env.rank, world_size=env.world_size,
                             device=env.device)
    manager = TrainingManager(model, schedule, bank_update=anvil_bank_update)
    value_pull = ValueEmbedPull(model.value_embeds, model.vocab_size, manager.optimizer,
                                rank=env.rank, world_size=env.world_size, device=env.device)
    averages = TailAverages(manager.optimizer, env.rank, value_embed_updates=is_update_step,
                            total_steps=schedule.total_steps)
    runner = EagerStepRunner(model, manager.optimizer, table, value_pull)
    gathers = DeferredGathers(value_pull, NoFp8Refresh())

    # Keep #92's prefix-token objective and same per-rank construction scheme.
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size,
                                                        bucket=env.rank, num_buckets=env.world_size))
    dist.all_reduce(model.prefix_table, op=dist.ReduceOp.MAX)
    sampler.set_prefix_table(model.prefix_table.cpu().numpy())
    model.train()
    batches = ScheduledBatches(train_loader(), schedule, steps=range(run_steps))
    prefetch = RowPrefetch(table, value_pull, batches, range(run_steps),
                           is_update_step, prefetch_staging)
    mask_builder = None
    if not stop:
        mask_builder = BackgroundCanonicalMask(model.vocab_size, owner=env.master_process,
                                               print0=print0)
        mask_builder.start()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(run_steps):
        manager.advance_schedule(step)
        gb10_step(manager, runner, prefetch, sampler, batches, gathers,
                  step, model, step + 1 < run_steps)
        averages.tick(step)
        if (step + 1) % int(os.environ.get("GB10_LOG_EVERY", "25")) == 0 or step + 1 == run_steps:
            torch.cuda.synchronize()
            print0(f"[GB10] step={step+1}/{run_steps}, elapsed={time.perf_counter()-t0:.2f}s, "
                   f"allocated={torch.cuda.memory_allocated()/2**30:.2f}GiB")
    gathers.flush_final()
    if stop:
        print0("SMOKE ONLY: no final tail-ship or FineWeb certification; do not compare with 3.28 record")
    else:
        manager.advance_schedule(schedule.total_steps)
        manager.apply_final_ws_ext()
        mask_builder.wait()
        mask_builder.collect(model.canon_mask)
        averages.ship()
        model.complete_yarn_tables()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = args.val_tokens // args.val_batch_size
        val_batches = [next(viter) for viter in [val_loader()] for _ in range(val_steps)]
        val_pulls = table.eval_pulls([b.ngram_ids for b in val_batches])
        for pull in val_pulls:
            table.land(pull)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t0
        model.eval()
        loss = torch.zeros((), device=env.device)
        with torch.no_grad():
            for batch, pull in zip(val_batches, val_pulls):
                table.land(pull)
                loss += model(batch.inputs, batch.targets, batch.cum_seqlens,
                              table.slots(pull, batch.ngram_ids),
                              manager.get_forward_args()).mean()
        loss /= val_steps
        dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
        print0(f"[GB10] val_loss={float(loss):.6f} train_seconds={train_seconds:.3f} "
               f"(NOT H100-comparable; different batch/table/backend)")
    gc.collect()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
