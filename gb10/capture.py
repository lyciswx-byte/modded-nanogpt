"""Two-rank GB10 compute-graph capture of forward, backward and ANVIL banks.

This is the H100 record's *correctness protocol* (not its SM90 kernels):
1) capture every distinct static shape after preparing native SM121 ops,
2) eager/replay self-check each forward and each optimizer bank,
3) reset model/ANVIL/n-gram to the identical initial state,
4) assert every captured pointer and optimizer-bank alias is unchanged,
5) seal captures: never recapture on the timed training loop.

Boundaries (CPU pinned transfers, n-gram/value row exchange, NCCL collectives,
candidate upload, tail shipping) remain eager/outside CUDA graph; these have
dynamic host data and cross-node synchronization and are not safely capturable
as one graph. All GPU differentiable FORWARD/BACKWARD including FP8 quantization
and fused CE, and the three ANVIL matrix-bank updates ARE captured.
"""
import copy
import os

import torch

from track_1_short.data import ScheduledBatches
from track_1_short.perf.cuda_graphs.baked_addresses import baked_addresses, assert_addresses_unchanged
from track_1_short.perf.cuda_graphs.capture_plan import plan_warmup, step_graph_key
from track_1_short.perf.cuda_graphs.optimizer_graphs import AnvilBankGraphs, assert_banks_alias_live_state
from track_1_short.perf.cuda_graphs.step_graphs import StepGraphs
from track_1_short.perf.row_prefetch import RowPrefetch
from gb10.eager import NoFp8Refresh


def enabled():
    return os.environ.get("GB10_CUDA_GRAPHS", "0") == "1"


def check_graph_configuration(run_steps):
    if not enabled():
        return
    if os.environ.get("GB10_ATTN_BACKEND", "sdpa") != "graph":
        raise RuntimeError("GB10_CUDA_GRAPHS=1 requires GB10_ATTN_BACKEND=graph: "
                           "CPU .tolist() varlen or FA4 host segment selection cannot replay safely")
    if run_steps < 4:
        raise ValueError("GB10_SMOKE_STEPS >=4 required to self-check optimizer graphs")
    if int(os.environ.get("GB10_GRAPH_QUERY_TILE", "128")) not in (64, 128, 256):
        raise ValueError("unsupported graph attention query tile")
    # Cannot silently enable old H100 fullgraph compile; first capture the
    # eager SM121 route, then test a separate compile-enabled option.
    if os.environ.get("GB10_GRAPH_COMPILE", "0") == "1":
        raise RuntimeError("GB10_GRAPH_COMPILE=1 not validated; no fullgraph compiler fallback")


def prewarm_tiles_if_requested(optimizer):
    if os.environ.get("GB10_GRAM_BACKEND", "aten") == "triton":
        from gb10.tile_tuner import prewarm
        prewarm(list(optimizer.banks.values()))


def capture_all(model, manager, table, value_pull, sampler, schedule,
                training_loader_factory, prefetch_staging, run_steps, device,
                gb10_step, batches_print, optimizer_graphs):
    """Pre-capture and reset all GPU training compute before timed run."""
    if not enabled():
        raise RuntimeError("call capture_all only when GB10_CUDA_GRAPHS=1")
    prewarm_tiles_if_requested(manager.optimizer)
    # The optimizer's bank_update callback refers to THIS owner, created
    # before TrainingManager. Never create a second owner in capture_all.
    assert manager.optimizer.bank_update.__self__ is optimizer_graphs
    # train.py supplies a manager whose bank_update points at graphs.update.
    step_keys = [step_graph_key(schedule, sampler.counts, step, 2)
                 for step in range(run_steps)]
    step_graphs = StepGraphs(model, manager.optimizer, step_keys, table,
                            value_pull,
                            max(s.batch_size for s in schedule.stages) // 2,
                            device)
    from track_1_short.perf.deferred_gathers import DeferredGathers
    # Native GB10 quantization and CE are inside the step forward capture;
    # there is no separate FP8 weight cache refresh to capture.
    deferred = DeferredGathers(value_pull, NoFp8Refresh())

    transition = set(manager.get_transition_steps()) | set(sampler.count_change_steps())
    transition = {s for s in transition if 0 < s < run_steps}
    split = schedule.split_step
    base = {s for s in (0, 1, 2, 3) if s < run_steps}
    base |= {s + offset for s in transition for offset in (-2, -1, 0, 1)
             if 0 <= s + offset < run_steps}
    base |= {split + delta for delta in (0, 1, 2, 3)
             if 0 <= split + delta < run_steps}
    # Reuse the record's shape-key planning. FP8 scales refresh *within* FWD,
    # so no separate FP8 refresh graph needs initial delayed-scale calls.
    plan = plan_warmup(sorted(base), step_keys, manager.is_adam_step,
                       fp8_exact_scale_calls=1)
    batches_print(f"[GB10 graph] {len(plan.steps)} warmup visits; "
                  f"{len(set(step_keys))} static step keys; 3 optimizer graphs; "
                  "FP8 quantization/CE inside forward graph")
    state = {"model": copy.deepcopy(model.state_dict()),
             "optimizer": manager.get_state()}
    warmup_batches = ScheduledBatches(training_loader_factory(), schedule,
                                     steps=plan.steps)
    warmup_prefetch = RowPrefetch(table, value_pull, warmup_batches,
                                 plan.steps, manager.is_adam_step, prefetch_staging)
    for pos, step in enumerate(plan.steps):
        manager.advance_schedule(step)
        deferred.flush()
        if pos == plan.optimizer_capture_at:
            # Assumes all banks and their scalar addresses are stable; stream
            # waits are outside graph. all-reduce is NOT captured.
            ordered = [manager.optimizer.banks[label] for label in manager.work_order
                       if label in manager.optimizer.banks]
            optimizer_graphs.capture(ordered)
        gb10_step(manager, step_graphs, warmup_prefetch, sampler, warmup_batches,
                  deferred, step, model, has_next=False)
    deferred.flush_final()
    warmup_prefetch.close()
    step_graphs.seal()
    optimizer_graphs.seal()
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved(device) / (2 ** 30)
    # A GB10's 128GB is CPU/GPU SHARED, not HBM independent of RAM.
    # Refuse to enter the timed run if capture pools already consumed too
    # much unified address space; do not silently rely on host paging.
    max_reserved = float(os.environ.get("GB10_GRAPH_MAX_RESERVED_GIB", "96"))
    if reserved > max_reserved:
        raise MemoryError(f"GB10 captured graphs reserve {reserved:.1f}GiB > "
                          f"GB10_GRAPH_MAX_RESERVED_GIB={max_reserved:.1f}. "
                          "Lower GB10_GRAPH_TILES_PER_GROUP, stage batches or ngram rows.")
    from gb10.preflight import _mem_available_bytes
    available = _mem_available_bytes() / (2 ** 30)
    min_available = float(os.environ.get("GB10_GRAPH_MIN_HOST_AVAILABLE_GIB", "8"))
    if available < min_available:
        raise MemoryError(f"GB10 only has {available:.1f}GiB shared RAM available "
                          f"after captures, need >= {min_available:.1f}GiB. "
                          "Reduce table size/graph shapes before training.")
    batches_print(f"[GB10 graph] reserved={reserved:.1f}GiB, "
                  f"host_available={available:.1f}GiB")

    before = baked_addresses(model, manager.optimizer, step_graphs)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(state["model"])
    manager.reset(state["optimizer"])
    table.reset()
    value_pull.reset()
    sampler.reset()
    del state, warmup_batches, warmup_prefetch
    after = baked_addresses(model, manager.optimizer, step_graphs)
    assert_addresses_unchanged(before, after)
    assert_banks_alias_live_state(manager.optimizer)
    # Verify no FP8/H100 cache update is required for native GB10.
    batches_print(f"[GB10 graph] SEALED: {len(step_graphs.captured)*2} fwd/bwd, "
                  f"{len(optimizer_graphs.graphs)} ANVIL graphs; "
                  "all captured tensor addresses unchanged after reset")
    return step_graphs, optimizer_graphs, deferred


def graph_owners(device):
    """Create this owner BEFORE TrainingManager so bank_update captures it."""
    return AnvilBankGraphs(device)
