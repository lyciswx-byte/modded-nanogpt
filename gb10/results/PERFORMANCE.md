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

1. **Socket transport:** the successful two-host AllReduce does not prove
   the ConnectX-7 RDMA devices are exposed to Docker. New read-only
   `gb10.fabric` reports both sysfs and `/dev/infiniband` visibility;
   `GB10_REQUIRE_RDMA_DEVICE=1` fails fast if either side is missing.
   Even visible devices are NOT proof NCCL selected RDMA; inspect
   `NCCL_DEBUG=INFO` and `nccl-tests` transport/latency.
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
