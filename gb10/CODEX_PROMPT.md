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

## 1A. IMMEDIATE P0 TASK — investigate and fix FA4 backward diff=1544 (commit 4044024)

**This task takes precedence over the generic tuning sections below.** Start here; do not spend GPU time on Triton Tile speedups or a 1194-step training run while FA4 backward is incorrect.

### Confirm source and exact problem

- Review real commit **\`4044024820702bb61b4dc22c1b345f9ab377c41a\`** with \`git show 4044024 -- gb10/fa4.py gb10/results/CORRECTNESS.md\`. The commit is now available from GitHub. Compare it against the current \`2xGB10\` HEAD; **do not cherry-pick it again or overwrite newer fixes**.
- That commit changed the **adapter**, not the third-party CuTe backward implementation: \`window_size\` uses CuTe \`Int32\`, \`cute.make_fragment\` maps to \`make_rmem_tensor\`, and the \`nvvm.atomicrmw(res=...)\` wrapper rewrites to \`results=[...]\`. Native FA4 JIT reportedly compiled forward and backward, but \`test_fa4_varlen_mixed_heads_forward_backward\` failed with **maximum backward difference 1544.0**. That is a confirmed failure, **not** an acceptable FP8/BF16 tolerance issue.
- Read \`gb10/FA4_BACKWARD_DEBUG.md\`, \`gb10/diagnose_fa4_backward.py\`, \`gb10/fa4_masks.py\`, \`gb10/fa4.py\`, \`gb10/cute_compat.py\`, \`gb10/fa4_safebwd.py\`, \`gb10/test_accelerated.py\`, \`gb10/test_fa4_masks.py\`, \`gb10/test_fa4_diagnostics.py\`, and \`gb10/test_fa4_safebwd.py\` **before** editing.
- **Check out the real third-party bundle** at \`GB10_FA4_LOCAL_PATH\`; note that 4044024 does **not** contain the bundle's \`build/torch-cuda/interface.py\` or its native \`flash_bwd*\` kernel source. Do not pretend that the adapter alone fixes a JIT-compiled backward kernel.
- The current diagnostic and adapter share \`fa4_mask_kwargs(causal=..., window=...)\`, including typed \`Int32\`. Verify their **actual kernel invocation** and the effective mask after the bundle's own normalization. An upstream FlashAttention finite causal left window may canonicalize to a local \`(left,0)\` mask; do **not** assume \`causal=False\` is the culprit without comparing semantics.
- **Production default remains \`GB10_ATTN_BACKEND=sdpa\`** and FA4 remains opt-in/fail-closed. Never increase an \`atol/rtol\` threshold to hide an error of 1544, and never silently replace a failed native FA4 test with the \`fa4-safebwd\` diagnostic backend.

### P0-A. Capture reproducible inputs and environment on BOTH GB10 hosts

Perform these read-only checks in the **existing** CUDA/PyTorch ARM64 Docker container, with the locally patched FA4 bundle mounted. Substitute paths only after verifying their existence; do not guess a Git ref or mount location.

\`\`\`bash
git branch --show-current
git status --short
git rev-parse HEAD
git show --stat --oneline 4044024
python -m compileall -q gb10 track_1_short
python -m unittest gb10.test_fa4_masks gb10.test_fa4_diagnostics gb10.test_fa4_safebwd -v
python - <<'PY'
import os, platform, torch, triton
from gb10.cute_compat import cutlass_version
print("host=",platform.node(),"arch=",platform.machine())
print("GB10_PROFILE=",os.getenv("GB10_PROFILE"))
print("GB10_FA4_LOCAL_PATH=",os.getenv("GB10_FA4_LOCAL_PATH"))
print("torch=",torch.__version__,"torch_cuda=",torch.version.cuda)
print("triton=",triton.__version__,"cutlass_dsl=",cutlass_version())
print("cuda_available=",torch.cuda.is_available())
if torch.cuda.is_available():
    print("device=",torch.cuda.get_device_name(0))
    print("capability=",torch.cuda.get_device_capability(0))
PY
nvidia-smi
\`\`\`

Record both nodes' driver, container tag/digest (if visible), bundle git HEAD or exact file hashes, \`interface.py\` and \`flash_bwd*.py\` provenance, \`nvidia-cutlass-dsl\`, \`apache-tvm-ffi\`, \`kernels\`, CUDA, PyTorch, Triton, NCCL and SM121 capability. Redact host IPs, user paths, credentials and authentication tokens before committing logs. **Do not upgrade/downgrade CUDA or CUTLASS as a first response**.

Required environment result: \`gb10/results/FA4_ENVIRONMENT.md\`, with one row per host, source revision, tested flags and whether both hosts used the SAME bundle source.

### P0-B. Run the seven staged diagnostic cases and preserve failures

Run this **on each GB10 separately**, not through a two-rank training job. Use the exact locally patched bundle from the 4044024 JIT test:

\`\`\`bash
export GB10_PROFILE=1
export GB10_ATTN_BACKEND=sdpa
export GB10_CUDA_GRAPHS=0
export GB10_FA4_LOCAL_PATH=/opt/flash-attn-4-sm120  # first verify actual mount
mkdir -p gb10/results
set -o pipefail
python -m gb10.diagnose_fa4_backward --repeats 3 \
  --json gb10/results/fa4_backward_node1.json \
  2>&1 | tee gb10/results/fa4_backward_node1.log
printf 'FA4_DIAGNOSTIC_EXIT_CODE=%s\n' "$?"
\`\`\`

Use \`node2\` for the second host's filenames. Run \`gb10.test_accelerated.AcceleratedGB10.test_fa4_varlen_mixed_heads_forward_backward\` separately and record the complete traceback; it is **expected to fail until the native backward is fixed**, and an expected failure is not a PASS:

\`\`\`bash
python -m unittest \
  gb10.test_accelerated.AcceleratedGB10.test_fa4_varlen_mixed_heads_forward_backward \
  -v
\`\`\`

The isolator tests, **in this order**:

| Case ID | Sequence lengths | QK/V head | Causal | Window | Question |
|---|---|---|---|---|---|
| \`dense-noncausal-64\` | 192 | 64/64 | no | full | Is basic backward wrong? |
| \`dense-causal-64\` | 192 | 64/64 | yes | full | Does triangular masking break it? |
| \`varlen-causal-64\` | 96,160,64,192 | 64/64 | yes | full | Are packed offsets / scatter wrong? |
| \`single-window-64\` | 192 | 64/64 | yes | left 128 | Is local/causal window backward wrong? |
| \`varlen-window-64\` | 96,160,64,192 | 64/64 | yes | left 128 | Does varlen + window interaction fail? |
| \`varlen-window-mixed\` | 96,160,64,192 | 64/128 | yes | left 128 | Does zero-padding Q/K / V width break it? |
| \`varlen-window-128\` | 96,160,64,192 | 128/128 | yes | left 128 | Is 128-dim backward layout wrong? |

The diagnostic should attempt **all seven cases even if one fails** and must return a nonzero exit code if ANY case fails; the \`tee\` pipeline must preserve this exit code via \`set -o pipefail\`. In a failure log include any JIT traceback, not just the last line.

For **every case and every repeated run**, extract from JSON the following fields **separately** for \`out\`, \`dq\`, \`dk\`, \`dv\`:
\`finite\`, \`max_abs\`, \`p99_abs\`, \`rms_relative_to_ref\`, \`ref_rms\`, \`ref_max_abs\`, \`worst_flat_index\`, \`actual_at_worst\` and \`ref_at_worst\`. Also report the repeat-to-repeat gradient metrics from \`repeat_grads\`. Report per-case \`pass\` / \`exception\` and the **first failing case**. Do not collapse dQ/dK/dV to a single maximum of 1544.

Run a separate wide-QK scale test matching record #92: \`--case varlen-window-128 --scale 0.085 --repeats 3\`. The general diagnostic defaults to \`--scale 0.13\` for its first six cases; avoid claiming that its all-cases \`128/128\` configuration alone covers production 0.085.

At minimum, add a deterministic **second seed** for every failing case: if the script does not expose \`--seed\`, add that CLI option, unit-test it on CPU and print/store \`seed\` in each JSON case before claiming multi-seed coverage. Compare \`grad_output=ones\` against seeded random upstream gradients on the first failure to distinguish accumulation errors from mask errors.

### P0-C. Root-cause classification and exact source-level fix

Write one evidence-backed root-cause statement per failed case, naming the **actual third-party source file + function + line number**. Use these rules:

| Observed failure | Audit FIRST |
|---|---|
| \`out\` differs | Forward padding, window endpoints, causal/local normalization, head mapping and varlen offsets; **do not blame backward yet** |
| \`out\` matches, only \`dq\` fails | Softmax LSE, saved forward state, \`dP/dS\`, scale, dQ accumulate/layout and QK slicing |
| \`out\` matches, only \`dk\`/\`dv\` fails | \`nvvm.atomicrmw(results=...)\` argument/result ABI, FP32 accumulation, zero-init, global addresses, shared-memory layout and races |
| \`out\` matches, all gradients fail | Backward ABI, grad_output strides/layout, incorrect saved tensors or LSE scale |
| First failure at causal | Triangular mask / effective \`is_causal\` flag mismatch |
| First failure at window | \`is_local\`, inclusive left endpoint, right=0, negative-index clipping or typed \`Int32\` JIT specialization |
| First failure at varlen | \`cu_seqlens\`, segment-local offsets, head/batch flattened indexing or dK/dV scatter |
| First failure at mixed 64/128 | Padded Q/K zero tail, original scale 0.13, original-shape gradient slicing and per-head layout |
| Huge errors vary between identical reruns | Suspect uninitialized buffer, missing sync, data race, invalid atomic address or writes outside allocated regions |

Specific \`4044024\` review: the wrapper globally monkeypatches \`nvvm.atomicrmw\` on FA4 load and translates \`res\` to \`results=[...]\`. **Inspect the signature and the call sites of the installed CUTLASS/MLIR version and third-party bundle.** Show whether the caller expects the atomic's old value, FP32 accumulator, or an output container. Do not treat a keyword rename or successful JIT as proof of semantic equivalence. Prefer a narrowly scoped patch inside the vendored/external bundle over unbounded process-global monkeypatching if supported by the environment.

Compare the relevant \`flash_bwd*.py\` / \`interface.py\` lines with the **version-matched** Dao-AILab SM120 path; changing only the wrapper is insufficient if backward JIT logic changed. Validate each fix on one GB10 with the seven-case isolate **before** the other host. If a proposed fix regresses a passing case, reject it.

Acceptance thresholds: do **not** relax \`test_accelerated.py\` forward \`atol=0.085, rtol=0.06\` or native FA4 backward \`atol=0.12, rtol=0.12\`. Require finite output/gradients, all seven cases passing, repeatable results and no materially different errors between nodes. Record actual errors, not just PASS. More detailed RMS thresholds from the isolator are additional gates, not replacements for the existing unit test.

### P0-D. Prove a fix rather than just diagnosing

Once a candidate native FA4 backward patch passes the staged checks, run on **both** devices:

\`\`\`bash
python -m unittest gb10.test_fa4_masks gb10.test_fa4_diagnostics gb10.test_fa4_safebwd -v
python -m unittest \
  gb10.test_accelerated.AcceleratedGB10.test_fa4_varlen_mixed_heads_forward_backward \
  -v
python -m unittest gb10.test_accelerated -v
python -m gb10.bench_accelerators --kind attention --repeats 30
\`\`\`

Collect per-shape fwd+bwd CUDA-event p50/p90, BF16 reference error, per-rank allocated/reserved GiB, and FA4 versus native SDPA on the **same lengths/windows/heads**. If benchmark code lacks p90, extend it rather than invent the number. **Only after passing these gates** consider a two-node 8-step/128-step FA4 experiment; use the same seeds, token budget and n-gram capacity as the native SDPA baseline. Production remains \`sdpa\` until both-node success and acceptable CE/loss traces; never enable FA4 by changing the default just because JIT works.

\`GB10_ATTN_BACKEND=fa4-safebwd\` is a diagnostic-only bridge: FA4 forward parity-checked against SDPA, but backward recomputed by SDPA. It is **not** a native CuTe backward fix, is likely slower and is incompatible with CUDA Graph capture. If an FA4 fix is still blocked, leave native FA4 disabled, describe the exact failing kernel, and do not claim completion.

### P0-E. MANDATORY RESULT FILES AND FINAL ANSWER FORMAT

Codex **must create or update** the following, even when a failure blocks further work:

1. \`gb10/results/FA4_ENVIRONMENT.md\`: node1/node2 versions, image digest, actual installed CUTLASS & bundle revision/hash, SM121, mounting, source/commit IDs and exact masks/flags.
2. \`gb10/results/FA4_BACKWARD_DIAGNOSIS.md\`: **seven-row table** with \`out/dq/dk/dv\` max and normalized RMS, first failing case, expected vs actual at worst indices, repeat variance, two seeds, upstream grad type, classification and kernel file:line evidence. Include P0 severity.
3. \`gb10/results/fa4_backward_node1.json\` and \`fa4_backward_node2.json\` (plus \`.log\` only when suitably scrubbed): outputs of the actual diagnostics. If hardware is inaccessible, write \`BLOCKED\` in the Markdown files and **do not create fake JSON**.
4. \`gb10/results/FA4_PATCH_REVIEW.md\`: baseline \`4044024\` vs candidate patch, exact CuTe API/ABI changes, diff links, why the semantics are now correct, regressions, PASS/FAIL/BLOCKED and remaining risks.
5. Update \`gb10/results/CORRECTNESS.md\`, \`gb10/results/PERFORMANCE.md\`, and \`gb10/results/DECISIONS.md\` with a concise status and links; do not overwrite earlier historical evidence.

Required final Chinese report template:

| Field | Mandatory content |
|---|---|
| Revision tested | \`2xGB10\` Git SHA, external FA4 source SHA/hash, container digest |
| Hardware | node1/node2 and installed driver/CUDA/PyTorch/Triton/CUTLASS |
| First failing scenario | Case ID and \`out/dQ/dK/dV\` values |
| Reproducibility | seeds, repeats, \`grad_output\` type, stability |
| Root cause | **specific file:function:line**, evidence, not speculation |
| Changes | patches, unit tests and exact links |
| Tests | PASS/FAIL/BLOCKED separately for CPU, GB10 GPU, two-node training |
| Speed and quality | **measured** fwd+bwd latency, memory, CE only if available |
| Outstanding actions | exact blockers/commands, no vague promises |

**Done means:** real native SM121 FA4 forward AND backward numerically pass for the #92 mixed-QK/V varlen sliding workload on each GB10, both-node training passes, and performance is measured. A CI-only or diagnostic-bridge-only result is **NOT DONE**. If hardware/bundle sources are unavailable, complete source/CPU work, commit it, and explicitly report \`FA4 backward GPU fix BLOCKED\`.

---

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

**First action for Codex:** execute **Section 1A / 4044024 FA4 backward P0**: verify the actual commit and third-party bundle source, run baseline CPU tests, collect the seven real-SM121 per-tensor diagnostics on both hosts, identify the first failing case, and patch only the proven native backward defect. Only after its gates pass should you resume the generic P0/P1 performance work below. If GPU hardware is unavailable, produce honest BLOCKED artifacts and commit the static/CPU improvements.
