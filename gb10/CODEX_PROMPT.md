# Codex execution prompt — 2× NVIDIA GB10 optimization and certification

> **Copy this entire Markdown file as the Codex task, or ask Codex to execute `gb10/CODEX_PROMPT.md` in this repository.**
>
> Repository: `https://github.com/lyciswx-byte/modded-nanogpt`
> Work only on branch: **`2xGB10`**.
> Source benchmark: modded-nanogpt world-record #92 / ANVIL2 (historical 8×H100).
> Target: **two separate GB10 (SM121) hosts, one GPU process each**, existing ARM64/CUDA-enabled NVIDIA Docker containers, ConnectX-7 network.
>
> **IMPORTANT:** This is a real implementation / audit / benchmarking task, not a speculative essay. Modify code, add tests, run every test supported by your environment, save reproducible results, and commit the work. NEVER claim hardware throughput, final FineWeb loss, or end-to-end correctness based on CPU CI alone. Do not run dangerous or irreversible host operations. Do not alter master or rewrite history.

## 0. Mandatory execution rules

1. Inspect `git status --short`, `git branch --show-current`, `git log -3 --oneline`; preserve unrelated local changes. Fetch and checkout `2xGB10` only if safe. Create a new work branch off it if Codex cannot directly update `2xGB10`; report that explicitly.
2. Read `gb10/README.md`, `gb10/AUDIT.md`, `gb10/train.py`, `gb10/capture.py`, `gb10/graph_attention.py`, `gb10/fa4.py`, `gb10/tile_tuner.py`, `gb10/triton_tiles.py`, `gb10/fp8_linear.py`, `gb10/fused_fp8_ce.py`, `gb10/loss.py`, `gb10/preflight.py`, plus the referenced `track_1_short/` modules.
3. Obtain the **actual** device capabilities, GPU memory/cgroup limits, CUDA driver and runtime, PyTorch, Triton, CuTe/FA4 and NCCL versions on each real GB10. Use the existing Docker images; **do not blindly replace** their CUDA/PyTorch/NCCL stack with H100 pins. If GB10 access is unavailable, continue with static/CPU work, clearly mark GPU tests BLOCKED, and do not make untested kernels the default.
4. All test outcomes must be recorded as `PASS / FAIL / BLOCKED` together with the exact command, commit SHA, software versions, node, and benchmark parameters. Include errors and tracebacks, not fabricated latency numbers.
5. Keep the H100/master path unchanged. Do not load `sm_90` patched FA3, compile `compute_capability="90"` NVRTC CE, or invoke H100-only WGMMA/TMA code in a GB10 process.
6. Prefer correctness-preserving low-memory optimizations, measured on **both** GB10s. Keep eager BF16 SDPA and the current graph-safe SDPA as safe fallbacks. New accelerated paths must be opt-in until two-node parity, reproducibility, loss and performance gates pass.
7. Never push credentials, host addresses, HF auth tokens, dataset content, binary checkpoints, profiler traces with secrets, local `.env` or driver artifacts. Do not invoke force-push.

## 1. Current known state (facts to verify before editing)

- Branch `2xGB10` contains a functional-intent two-rank trainer and SM121 opt-in FA4, Triton E4M3 projections, CE, ANVIL Gram Tile tuning, CUDA Graph compute capture, dynamic document masking, streaming validation and CPU/reference CI.
- Latest code audit fixed unconditional sm90 NVRTC import, unsafe rectangular symmetric Gram Tiles, unbounded validation batches, vocabulary-specific CE block calibration, and graph allocator defaults.
- Graph attention now has `GB10_GRAPH_SEGMENT_LOOKUP=separate|fused`, with CPU gradient tests for both modes and duplicate BOS offsets; **do not assume fused is faster until measured**.
- `GB10_CUDA_GRAPHS=1` requires `GB10_ATTN_BACKEND=graph`. CPU file loading, cross-host NCCL, sparse n-gram row exchange, some Adam paths and validation are outside captured graphs. This is **not** one monolithic end-to-end CUDA graph.
- Default `GB10_NGRAM_ROWS=21150720` and `GB10_BATCH_DIVISOR=16` change the total token budget and quality vs record #92. Do **not** assert historical 39.914s or CE 3.2773 equivalence.

## 2. Priority P0 — make the 2-node program provably runnable

Implement and demonstrate:
- Verify all imports and startup **inside both GB10 Docker containers**. Check for sm90/TMA import-time compilation throughout `track_1_short/perf/kernels` and third-party libraries, not just the two currently guarded files.
- Run a first two-rank NCCL sanity test over the real high-speed interface before allocating the 21M-row n-gram table. Confirm actual NCCL transport, timeout behavior, driver compatibility and duplicate tensor shapes on both ranks. If launch fails, produce root cause and patch.
- Inspect `gb10/train.py` against the `TrainingManager`, `ScheduledBatches`, `RowPrefetch`, `SampledSoftmax`, `ValueEmbedPull` and `TailAverages` APIs. Verify reset, final ship, validation score aggregation, cache ownership, data iterator lifetime, `GB10_SMOKE_STEPS` and loss scaling. Add minimal low-memory unit/integration tests with mocks or CPU where possible.
- Verify that `gb10/eager.py` handles ANVIL gradients, Adam accumulation and n-gram sink gradients identically to original `StepGraphs`/eager behavior across at least 8 consecutive steps, including sparse update cycles and candidate changes.
- Check `gb10/attention.py` reference QK norm/YaRN/paired head/key-offset semantics against `track_1_short/perf/kernels/qkv_rope.py`, including gradients and padded/duplicate BOS boundaries.
- Check FA4 `(Dqk,Dv)=(64,128),(64,64),(128,128)` with real forward **and backward**, masked varlen, sliding window and mixed-width paths; a kernel that silently produces the wrong mask must block the accelerated mode.
- Add cross-node fail-fast assertions and safe teardown so a rank-specific error does not silently leave the other rank hung.

**Gate:** a complete 2×GB10 BF16/eager 8-step smoke, with finite losses/gradients and a correct multi-rank barrier, before enabling all accelerators at once.

## 3. Priority P0/P1 — CUDA Graph replay validity and memory

1. Validate every distinct `StepGraphKey` and `plan_warmup` visit for full 1194-step schedule **plus reduced GB10 batch stages**. Test first-use vs second-use self-check and no missing graphs after sealing.
2. Verify captured forward/backward sees **new** `batch.inputs`, `targets`, `cum_seqlens`, `ngram_slots`, sampled candidate IDs and MTP/prefix weights on replay. Assert pointers remain stable after warmup reset, including all model/optimizer state and sampled candidate buffers.
3. Validate the paired-head path uses doubled sequence boundaries inside captured graph; dynamic segment masks cannot cross documents or virtual segments. Include test with multiple leading/interior duplicate offsets and very short documents.
4. Verify `torch.autograd.grad` and FP8 custom autograd are capture-compatible with the **installed** SM121 PyTorch/cuBLASLt; document any unsupported call and add a fast failure or corrected graph-safe implementation.
5. Verify ANVIL's `@torch.compile(fullgraph=True)` can safely trace the selected `gb10.tile_tuner` dispatch. Never execute an uncalibrated Tile search inside capture; exercise both `GB10_GRAM_BACKEND=aten` and `triton` while checking update parity.
6. Compare graph-safety and GPU memory for query Tiles `64,128,256`, grouped Tiles `1,2,4,8`, and `GB10_GRAPH_SEGMENT_LOOKUP=separate|fused`. Profile capture time, graph node count (if available), captured pool reserved bytes, post-reset host MemAvailable and per-step replay latency.
7. Try CUDA Graph compatibility with external FA4 **only if** a replay with live-changing document offsets has first passed numerical and gradient tests. Do not enable the CPU-based FA4/Hybrid dispatcher inside a graph.
8. Keep dynamic CPU data prep, n-gram and inter-node NCCL collectives OUTSIDE the captured graph unless a separate fully tested NCCL graph protocol proves capture/replay safe on both hosts.

**Gate:** one real GB10 successfully captures and replays all tested graph keys with changing live data and unchanged output/gradient semantics; repeat on second host; then two-rank smoke.

## 4. Priority P1 — Blackwell-specific Triton / FP8 tuning

- Audit `gb10/tile_tuner.py`: symmetrical XXT/XTX/poly mirrored-store safety, stride/layout variants, full-grid coverage, BF16 numerical error and auto-tune candidate rejection. The **square M/N restriction is mandatory**. Time on real SM121 with CUDA events, never use H100 Tile rankings as GB10 results.
- Audit `gb10/triton_tiles.py` separate CE widths **10240/14336/24576/50304**, FP8 quantizer scales, E4M3 clamping, correctness for NaN/inf/zero inputs, contiguous vs transposed operands and candidate replay. Optimize quant/CE Tile sizes independently by observed GB10 performance; the fastest isolated partial kernel may not minimize whole loss time.
- Check `gb10/fp8_linear.py` `torch._scaled_mm` supported signature, row/column scaling and output orientation on installed torch; confirm both forward and weight/input gradients against BF16, including `c_proj.T` bank storage layout. Test BF16 backward vs any proposed FP8 backward before using the latter.
- Audit `gb10/fused_fp8_ce.py`: global-row MTP targets across slab boundaries, repeated targets, absent prefix candidates, logits at extreme scales, precision in `logsumexp`, BF16 dLogits and upstream gradient scaling. Test scalar loss and **full dLoss/dLogits**, candidate-to-vocab positions, all widths and end-of-sequence boundaries.
- Optimize LM Head sampled-vocab gather and CE slab row count. Inspect whether per-slab `F.linear` causes excessive launch overhead/graph memory; compare existing fused FP8 CE and a native BF16 fallback across loss, gradient, latency and memory.
- Do not turn on an approximate FP8 backward or new Tile until parity is quantitatively established across different seeds and nontrivial actual training gradients.

**Gate:** real-SM121 forward/backward tests; benchmark tables with p50/p90 latency, peak reserved/shared memory and numerical error for each `(Dqk,Dv)`, sampled-vocab width, Tile shape, and `GB10_CE_SLAB_ROWS`. Report winners *only* from measured results.

## 5. Priority P1 — network, sparse table and unified-memory economics

- Probe high-bandwidth ConnectX-7 / NCCL with both containers. Measure AllGather, ReduceScatter, AllReduce, AllToAll, and overall communication overlap with optimizer work. Log whether RDMA/RoCE was actually selected; never assume it merely because an interface exists.
- Profile sparse n-gram Pull/Scatter and lazy row-Adam updates across `GB10_NGRAM_ROWS=21,150,720`, then (if memory passes) 42,301,440, finally original 84,602,880. Account for row states, GPU graph pools, data pinned pages and OS memory within **128GB shared physical memory per host**.
- Compare update cycle lengths (2 and 4 steps), sparse cache usage and network bytes, BUT retain the same effective optimizer step count or adjust the beta/WD history correctly when conducting an ablation.
- Refine CPU pinned ring depth, `GB10_VAL_PULL_WINDOW` (1..8), and prefetch slack only after logging CPU/RAM pressure, NCCL throughput and GPU stalls. Do not sacrifice validation exactly-once coverage.
- Keep 8×H100 vs 2×GB10 comparisons separated: use achieved validation CE **and** effective training token count, not just raw 1194 steps.

## 6. Required tests, artifacts and commands

Run the portable suite in *every* edit cycle:

```bash
python -m compileall -q gb10 track_1_short
python -m unittest discover -s gb10 -p 'test_*.py' -v
git diff --check
```

Run on each GB10 when hardware is available:

```bash
python -m unittest gb10.test_accelerated gb10.test_tiles gb10.test_graph_attention -v
python -m gb10.bench_accelerators --kind all
GB10_GRAPH_SEGMENT_LOOKUP=separate python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
GB10_GRAPH_SEGMENT_LOOKUP=fused python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
```

Two-host smoke (inside **both** existing GB10 Docker containers, at the same time):

```bash
export MASTER_ADDR=<rank0_fast_network_ip>   # set from actual environment
export NODE_RANK=<0_or_1>                    # distinct on the two machines
export GB10_SMOKE_STEPS=8
export GB10_PROFILE=1

# Start with no GB10 acceleration flags, then enable one change at a time:
bash gb10/run_two_nodes.sh
```

For graph + tuned Triton on two hosts **after** independent hardware parity tests:

```bash
export GB10_GRAM_BACKEND=triton
export GB10_TUNE_TRITON_TILES=1
export GB10_CUDA_GRAPHS=1
export GB10_ATTN_BACKEND=graph
export GB10_GRAPH_SEGMENT_LOOKUP=fused  # only after benchmarking vs separate
export GB10_FP8_LINEAR=1
export GB10_FUSED_FP8_CE=1
export GB10_SMOKE_STEPS=8
bash gb10/run_two_nodes.sh
```

Escalate to 128 steps, then a full run only after 8-step parity passes.
Never invent the interconnect IP or assume inaccessible nodes are available.

Write these files (with **measured or BLOCKED** fields):
- `gb10/results/ENVIRONMENT.md`: per-node GPU, driver, CUDA, torch, Triton, NCCL, FA4/CuTe versions and network interface/topology.
- `gb10/results/CORRECTNESS.md`: tests, exact commands, seed, tolerances and PASS/FAIL/BLOCKED, plus relevant issue/stack traces.
- `gb10/results/PERFORMANCE.md`: baseline vs tuned Tile/FP8/FA4/Graph settings; p50/p90, memory and end-to-end timings with explicit token budget.
- `gb10/results/DECISIONS.md`: code changes, selected defaults, rejected candidates and reasons.
- Unit and integration tests for every fixed bug; keep offline CPU/CI tests lightweight and deterministic.

## 7. Git / final response contract

1. Commit coherent changes with meaningful subjects, ideally one commit per independent fix/feature. Run tests after the final edit.
2. Verify clean or clearly account for working tree changes; `git diff --check`, `git log -5 --oneline`.
3. Push to `2xGB10` only if authorized and connected; otherwise present exact local commits and push instructions. Never force push. **Do not modify master.**
4. Final report in Chinese must include:
   - file-by-file code changes and severity-ranked root causes;
   - test summary separating CPU/CI from real GB10;
   - exact commit SHAs and the GitHub links to diff/PR;
   - measured speeds and validation losses **only where actual logs exist**;
   - known unresolved correctness risks and actionable next experiments.
5. A successful run is not merely a green CPU CI: actual two-node GB10 convergence and throughput require separate runtime evidence. If hardware is absent, conclude **`静态/CPU 已验证，GB10 实机验收待完成`**, not `双机移植已完成`.

**First action for Codex:** perform the P0 audit and reproduce the current CI tests before implementing any new SM121 kernel. Then identify the highest-impact provable defect, fix it, add regression coverage, benchmark where possible and continue by the priority gates above.
