# GB10 two-node performance — feedback and corrected accounting

Status: **PARTIAL hardware evidence, NOT a full training-speed certification**.

## What Codex actually measured on the two GB10s

From `codex/reexecute-ff98` (head `627209636e67`), using the NVIDIA
`modded-nanogpt:92-gb10` container and two machines:

| Result | Observation |
|---|---|
| Two-rank NCCL AllReduce | PASS, each rank observed `3.0` |
| NCCL network backend | `NET/Socket` on `enp1s0f1np1`, **not RDMA** |
| Two-node BF16 eager smoke | PASS, 8/8 steps on both ranks |
| Rank-0 elapsed time | **13.19 s**, includes this short smoke's timed execution |
| Rank-0 GPU allocated at step 8 | **19.39 GiB** |
| Advanced paths | FA4 blocked by CUTLASS API import; full graph / FP8 train convergence not measured |
| Validation CE | Not computed during `GB10_SMOKE_STEPS=8` |

The previous feedback stated **196,608 global training tokens** using
`8 × 12,288 local tokens × 2 ranks`. **That formula is incorrect for
the first eight steps.** `12,288` is the maximum per-rank size across
the later stage-3 configuration, printed as `local_tokens=max_step`;
it is NOT the number of tokens processed in stage 1.

The source's first-stage batch is `8*2048*8 = 131,072 GLOBAL tokens`,
then the default `GB10_BATCH_DIVISOR=16` produces **8,192 global /
4,096 local tokens per step**. Thus the **default-config eight-step
budget is 65,536 global tokens**, not 196,608. The previous run did not
persist the effective divisor in its logs, so 65,536 is a **source-derived
default-config calculation**, not a separately verified runtime counter.
The training entrypoint now logs the actual stage's planned global token
budget and accumulates it after each completed step.

Under default values, `65,536 / 13.19s ≈ 4,968 global tokens/s` for the
short smoke, but this is **not a benchmark or repeatable throughput
measurement**: launch/synchronization overhead is disproportionate and
may depend on network and container setup. Re-run with the new cumulative
token logging before reporting a definitive throughput.

## Current performance bottlenecks

1. **Network transport:** the initial two-host test used `NET/Socket`.
   A later Codex feedback run explicitly exposed `/dev/infiniband`,
   selected `NCCL_NET=IB` and the active `rocep1s0f1,roceP2p1s0f1`
   HCAs; **`NET/IB/0` and `NET/IB/1` were confirmed on both hosts**.
   Use optional `GB10_ROCE_PROFILE=1` on these measured machines to
   reproduce that mode, while keeping HCA names configurable. A successful
   8-step transport smoke does not measure RDMA bandwidth or scaling.
2. **FA4:** locally available SM120/121 bundle imports fail at
   `cutlass.cute.core.ThrMma` on the container's CUTLASS DSL.
   An opt-in compatibility shim now aliases moved **real** class objects
   for third-party annotation imports. No successful FA4 forward/backward
   run has been reported after this change.
3. **Full training:** 8 steps do not prove 1194-step convergence, n-gram
   table/cooldown behavior or 3.28 validation target. The default 2-GB10
   batch and n-gram table budgets intentionally differ from #92.

## Still required on both devices

```bash
python -m unittest gb10.test_accelerated.AcceleratedGB10.test_fa4_varlen_mixed_heads_forward_backward -v
python -m unittest gb10.test_accelerated.AcceleratedGB10.test_fp8_projection_and_grads -v
python -m gb10.bench_accelerators --kind all
GB10_GRAPH_SEGMENT_LOOKUP=separate python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
GB10_GRAPH_SEGMENT_LOOKUP=fused python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
```

Then record at least one measured 128-step/two-node baseline, NCCL
AllToAll and sparse row-prefetch timings, and eventually the full 1194-step
validation. Compare results only with the same token and table budgets.

## Later Codex RoCE evidence (branch `codex/reexecute-ff98`)

The feedback branch advanced after the first two-node smoke. When Docker
exposed `/dev/infiniband` and memlock was unlimited, NCCL used
`NET/IB/0` and `NET/IB/1` with HCA names
`rocep1s0f1,roceP2p1s0f1`. The two-host 8-step test completed again:

| Configuration | Rank-0 8-step elapsed | GPU allocated at step 8 |
|---|---:|---:|
| Earlier transport setup (Socket) | 13.19 s | 19.39 GiB |
| Explicit RoCE / IB setup | **12.69 s** | **19.39 GiB** |

This is a pair of **individual smoke observations**, not a controlled
repeated A/B benchmark. No standalone NCCL bandwidth comparison was
reported; do not infer a robust 3.8% speedup from these samples.

The later feedback repeated the previous **196,608 token estimate**,
but it still uses the wrong maximum-stage batch figure. Under the source
defaults the first eight steps have 65,536 global tokens; future runs
will log their actual effective divisor and cumulative tokens.

## Current 2026-10-19 probes

- Two-node NCCL `all_reduce`: **PASS**, result `3.0` on both ranks; no throughput number claimed.
- Eight-step training smoke: **BLOCKED** before step 1 because FineWeb binary shards were absent on both mounted checkouts.
- FA4 forward/backward: **BLOCKED** by Hugging Face HTTP 401; no latency or throughput number claimed.

## Data-mounted 8-step smoke (2026-10-19 UTC)

With FineWeb shards mounted read-only on both nodes, rank 0 completed 8/8 in
`12.88s`, `65,536` global tokens, and `19.39GiB` allocated. This is a measured
short smoke only; it is not comparable to the historical record and has no
validation CE.
