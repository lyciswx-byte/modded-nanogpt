# Codex Prompt — Record #92 on 1× NVIDIA GB10: Current Small-Batch Training Method

> Copy the **Task for Codex** below into Codex while the checkout is on branch `92-1xGB10`. This is an actionable engineering prompt for **the existing GB10 small-batch/more-updates method**, NOT an instruction to enable 8×H100-equivalent gradient accumulation.

## Task for Codex

You are a senior NVIDIA Blackwell / PyTorch / Triton performance engineer working in the GitHub repository:

- Repository: `https://github.com/lyciswx-byte/modded-nanogpt`
- Target branch: **`92-1xGB10`**; based on `92-1xH100`, itself a single-H100 port of speedrun record #92.
- Target hardware: **one NVIDIA GB10 / DGX Spark, SM121, ARM64 Grace, 128 GB shared CPU/GPU unified memory**.
- Goal: **make the *current* GB10 small-batch training method correct, buildable and robust on native GB10, then measure and improve throughput while preserving its present training semantics.** The target metric is FineWeb validation CE <= 3.28, but it is **NOT yet verified**, and the original 8×H100 39.9-second record is **not** a valid runtime expectation for one GB10.

### Non-negotiable scope: do NOT implement gradient accumulation

This prompt describes the **existing** algorithm, which processes **one microbatch and one optimizer update on each logical training step**. It compensates for a smaller global batch by making **more parameter updates**, not by accumulating 16 microbatches before one optimizer step.

**Do not** change to 16-microbatch gradient accumulation, virtual 8-rank reductions, original 1,194 optimizer updates, or an exact 8-GPU gradient-equivalence mode as part of this task. The separate `GB10_GRADIENT_ACCUMULATION.md` is **design-only**, and `tools/gb10_grad_accum_reference.py` (if present) is reference math, not the training entrypoint. Do not silently enable it. Propose such work separately if desired.

The present method intentionally differs from original 8×H100 training in global-batch noise, optimizer update cadence and smaller n-gram hash table. State this honestly.

### Current baseline: keep these behaviors unless a correctness bug requires explicit explanation

1. Single-GPU launch: `run.sh` delegates to `run_gb10.sh`; `torchrun --standalone --nproc_per_node=1 train_gpt.py`.
2. GB10 environment switches: `NANOGPT_TARGET=gb10`; `GB10_MICROBATCH_DIVISOR=2` (valid: 1/2/4/8); `NANOGPT_ATTENTION_BACKEND=flex`; `GB10_MLP_BACKEND=bf16`.
3. Stage batch tokens per **optimizer update**, defaults:
   - stage 0 = 8,192
   - stage 1 = 16,384
   - stage 2 = 24,576
   - taper = 20,480
   - extension = 8,192
4. `SINGLE_GPU_STEP_SCALE=8*GB10_MICROBATCH_DIVISOR=16` at world size 1. Default 17,952 scheduled + 832 growth + 320 extension = **19,104 optimizer steps**. ANVIL is updated each step; other Adam updates retain the repository's original alternating/cadence rules.
5. `VAL_BATCH_SIZE=131072` by default, `val_tokens=10485760`. Validation remains full-vocabulary FineWeb next-token CE.
6. Single-GPU default `NGRAM_VOCAB_SIZE=10575360` rows of width 768 BF16 (about 16.24 GB of weight storage alone), *not* original 84,602,880 rows. Preserve row-sparse Adam and local-only sparse-row communication bypass.
7. Keep the existing ANVIL optimizer, original FP8 QKV and LM-head paths, sampled softmax, n-gram lookup, value embeddings, model topology, schedule transitions, and CUDA-graph architecture unless correcting a concrete bug with an accompanying test.
8. No second GPU, no distributed data parallelism, no architecture spoofing or x86 CUDA emulation.

### Current operator / backend architecture

| Subsystem | Existing GB10 path | Required engineering attention |
| --- | --- | --- |
| Variable-length causal sliding-window attention | `track_1_short/perf/attention_backend.py` chooses PyTorch **FlexAttention** by default | Maintain packed-document boundaries, paired-head semantics, window mask, YaRN softmax scale, differentiability, tensor layouts; benchmark block-mask creation and compilation/capture. |
| Mixed QK/V head widths (64/128) | Zero-pad smaller width to shared width; slice output to original V width | Test forward **and** gradients against a small independent reference for (64,128), (128,64), (64,64), (128,128), and paired heads. |
| Alternative attention | `NANOGPT_ATTENTION_BACKEND=fa2` or `fa4` (opt-in) | Only use if the actual installed binary/backend supports SM121, unequal or padded head widths and varlen causal windows. Original checksum-pinned FA3 **targets H100/sm90a** and must stay blocked on GB10. |
| MLP | `GB10_MLP_BACKEND=bf16` calls standard `F.linear/ReLU/square/F.linear` with autograd | Preserve `relu(x @ W1.T)^2 @ W2` math and parameter gradients; inspect whether **unused FP8 MLP cache refresh / graph capture** still executes and can fail or waste time in BF16 mode. Do not remove FP8 lm_head refresh by accident. |
| Experimental FP8 MLP | Original fused Triton TensorDescriptor / TMA kernels | Opt-in only after SM121 forward/backward + numerical tests, resource/SMEM checks, and actual benchmark. Do not assume Hopper TMA configuration carries over. |
| QKV + RoPE | Original FP8 `torch._scaled_mm`, packed-QKV custom autograd, Triton rotary | Run complete custom-op forward/backward, not merely isolated FP8 GEMM. Validate BF16/FP8 scales and numerical stability. |
| Training CE | FP8 LM head, sampled softmax, custom CUDA/Triton loss | Validate actual compiled CE and sampled CE forward/backward, shapes and finite gradient; compiler API presence is **not** execution proof. |
| Sparse embeddings | Triton bigram/trigram + value embedding and row-wise sparse Adam | Verify repeated row IDs, gradients, on-device lookup bounds, update cadence and memory. |
| ANVIL / CUDA Graph | Captured fwd/bwd, ANVIL bank updates, FP8 refresh graphs | Verify stream dependencies, stable capture addresses, replay accuracy, event ordering and memory footprint on GB10. |

### Docker environment: prioritize native GB10 rather than blindly upgrading versions

Existing `Dockerfile.gb10` uses `nvcr.io/nvidia/pytorch:26.09-py3`, intended to provide PyTorch + CUDA 13.x + Triton together. The original H100 Dockerfile and `requirements.txt` are **not** the GB10 build.

- Inspect the **actual** image manifest for `linux/arm64` on the GB10 host, and check host driver compatibility. A registry tag, NGC release note or CPU CI pass alone does not prove the ARM64 manifest or SM121 execution works.
- Preserve NGC's CUDA-enabled arm64 PyTorch unless a verified change is required. Never replace it with the original H100 `torch==2.10/cu128` index wheel.
- Check `torch.version.cuda` reports CUDA 13.x, `torch.cuda.get_device_capability()` reports `(12, 1)`, `uname -m` is `aarch64`, and the NVIDIA container runtime sees one GB10.
- Check `TRITON_PTXAS_PATH`, compiler and GPU architecture settings. `TORCH_CUDA_ARCH_LIST="12.0+PTX"` is a portability choice, **not proof of SM121-specific optimal code**.
- Record pinned image tag plus pulled digest and versions of torch, CUDA, Triton, and NVIDIA driver from actual device logs.

### Execution plan: act in order and collect proof

**Phase A — inspect and reproduce**
1. Confirm `git branch --show-current` is `92-1xGB10`; inspect `GB10.md`, `Dockerfile.gb10`, `run_gb10.sh`, `train_gpt.py`, `track_1_short/config.py`, `track_1_short/model/{attention,gpt}.py`, `track_1_short/perf/{attention_backend,gb10_config}.py`, `track_1_short/perf/cuda_graphs/`, and sparse-row code. Avoid unrelated edits and avoid changing `master` or `92-1xH100`.
2. Review `git diff` and staged changes before editing; do not overwrite other contributors' work.
3. Run CPU syntax/tests first:
   ```bash
   python -m compileall -q train_gpt.py track_1_short tools tests
   python -m unittest discover -s tests -p 'test*.py' -v
   ```
4. On a **real GB10 host**:
   ```bash
   uname -m
   nvidia-smi
   docker buildx imagetools inspect nvcr.io/nvidia/pytorch:26.09-py3
   docker build --platform linux/arm64 -f Dockerfile.gb10 -t modded-nanogpt:92-gb10 .
   docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
     -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
     modded-nanogpt:92-gb10 python tools/gb10_kernel_probe.py --capture-attention
   ```
   The probe checks native attention forward/backward, compilation/capture, Triton launch, FP8 GEMM, BF16 MLP and n-gram gather; **do not mistake this for an end-to-end training test**.

**Phase B — fix highest-risk operator issues**
5. Resolve any real Docker build, arch, library ABI, unsupported SM121 kernel, Triton shared-memory, torch.compile or CUDA graph failures. Prefer a mathematically checked GB10-safe fallback first, then fused kernels after correctness.
6. Check default `GB10_MLP_BACKEND=bf16` is applied to both regular and parallel MLP paths. Audit `quantize_mlp_fp8`, `Fp8RefreshGraphs` and `lm_head` state; separate unused MLP cache refresh from **required** FP8 LM-head refresh if beneficial, preserving graph and optimizer invariants.
7. Explicitly probe the full QKV packed FP8 custom autograd and real sampled/full CE kernels, sparse-row update and ANVIL bank steps; expand `tools/gb10_kernel_probe.py` or add targeted tests where missing. Track tolerances vs a BF16/FP32 reference; do not silently skip failures.
8. Verify FlexAttention's live `cu_seqlens` mask construction under `torch.compile(fullgraph=True)` and CUDA graph replay with **different batches of identical static shapes**. Test no cross-document leakage and paired-head / window boundary behavior.

**Phase C — small-step training, full training, and performance**
9. If FineWeb shards are available, run a short **smoke test** with a documented bounded number of optimizer updates using existing `NUM_SCHEDULED_ITERATIONS` and `NUM_EXTENSION_ITERATIONS` overrides; account for any schedule assertions or CUDA-graph warmup requirements before choosing values. Keep one optimizer update **per** microbatch; do not add gradient accumulation.
10. Run the complete current 19,104-step configuration only on an equipped GB10. Report training loss, validation CE, elapsed time, tokens/s, memory peak, and any failure. If no GPU/data is accessible, stop at reproducible code/static tests and report the exact unverified stages.
11. Profile attention mask generation, QKV GEMM, MLP, loss, n-gram lookups, ANVIL, CUDA graph launch, H2D traffic and host CPU overhead. Establish a before/after timing under the **same** seed, token budget, data and model configuration. Prefer improvements proven to preserve the baseline's training semantics.
12. For memory pressure: test `GB10_MICROBATCH_DIVISOR=4` or `8`, smaller `VAL_BATCH_SIZE`, then (only if necessary) `NGRAM_VOCAB_SIZE`. Document that such changes alter step counts and/or embedding hash collision rate; do **not** call them equivalent reproductions.
13. Preserve `GB10.md` as the source of truth for actual launch flags and known limitations; update it with exact measurements, validated backends and reproduction instructions after any code changes.

### Acceptance criteria

- The default **small-batch/more-updates** method remains intact and the original GB10 stage sizes / 19,104 updates remain the default; no virtual rank or 16-way gradient accumulation code is enabled.
- Native arm64 Docker base builds **on an actual GB10**; pinned dependencies are internally consistent; kernel probe and a real model forward/backward/optimizer step pass.
- `torch.compile` + CUDA graph warmup/capture/replay and full training are verified **only if executed on a real GB10**. If not executed, state `NOT TESTED ON GB10` without inventing benchmark results.
- The numeric reference checks pass for causal/padded/unequal-width attention, BF16 MLP, FP8 paths used, sparse gradients and optimizer order. Record meaningful float tolerances.
- Validation computes the original FineWeb CE with 10,485,760 evaluation tokens; **CE <= 3.28 is a target, not a guaranteed or observed result** unless there is a real log.
- Keep modifications narrow; run unit/CI checks after edits; commit with a precise message on `92-1xGB10` only. Final report: commit SHA, changed files, commands actually executed, result of each check, before/after throughput if measured, unresolved blockers and recommended next steps.

### Existing files to consult

- [GB10.md](GB10.md): current GB10 assumptions, Docker and operator matrix.
- [Dockerfile.gb10](Dockerfile.gb10), [requirements.gb10.txt](requirements.gb10.txt), [run_gb10.sh](run_gb10.sh), [run.sh](run.sh).
- [track_1_short/perf/attention_backend.py](track_1_short/perf/attention_backend.py), [track_1_short/perf/gb10_config.py](track_1_short/perf/gb10_config.py).
- [track_1_short/model/attention.py](track_1_short/model/attention.py), [track_1_short/model/gpt.py](track_1_short/model/gpt.py).
- [track_1_short/perf/cuda_graphs/step_graphs.py](track_1_short/perf/cuda_graphs/step_graphs.py), [track_1_short/perf/cuda_graphs/fp8_refresh_graphs.py](track_1_short/perf/cuda_graphs/fp8_refresh_graphs.py).
- [track_1_short/perf/kernels/](track_1_short/perf/kernels/), [track_1_short/ngram_table.py](track_1_short/ngram_table.py), [track_1_short/sharded_rows.py](track_1_short/sharded_rows.py), [track_1_short/optim/anvil.py](track_1_short/optim/anvil.py).
- [tools/gb10_kernel_probe.py](tools/gb10_kernel_probe.py), [tests/test_gb10_adapters.py](tests/test_gb10_adapters.py), [.github/workflows/gb10-cpu-checks.yml](.github/workflows/gb10-cpu-checks.yml).

**Definition of done:** an honest, documented, reproducible and measured (if GB10 hardware is actually available) **current GB10 baseline**, keeping **one optimizer update per small batch**, with each changed operator checked for numerical correctness and platform compatibility. **Do not conflate this deliverable with eight-H100-equivalent gradient accumulation.**
