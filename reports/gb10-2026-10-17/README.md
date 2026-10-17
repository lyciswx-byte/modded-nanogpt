# Native GB10 validation — 2026-10-17

Branch: `92-1xGB10`. One microbatch and one logical optimizer update per step;
no gradient accumulation or virtual ranks enabled.

The default full run completed successfully on 2026-10-17. The native run proves
the default configuration completes on this GB10; its validation CE is above
the requested 3.28 threshold, so the quality target remains unmet.

## Changes

- Compile the actual CUDA CE kernels for the visible device (`121` on this host),
  rather than hardcoded Hopper `90`. The original body failed at module load
  with `no kernel image is available for execution on the device`.
- Correct truncated-MTP boundary gradients: missing future targets at the end of
  a microbatch no longer contribute to the softmax-normalizer gradient. This is
  an explicit correctness exception to preserving the previous numerical behavior;
  the forward objective, batch sizes, schedules and optimizer cadence are unchanged.
- In BF16 MLP mode, skip unused FP8 MLP cache quantization, empty MLP-only graphs
  and unused-cache snapshots. Retain the FP8 LM-head refresh, graph self-checks,
  bootstrap counter and stable-buffer/reset protocol.
- Separate forward-only CPU attention checks from real GPU gradient checks.
  Add native references for all head widths, paired heads, live document boundaries,
  packed FP8 QKV/RoPE, BF16 MLP, full/sampled CE, n-gram gradients, sparse row Adam
  and ANVIL's twin-rail cascade.
- Add optional bounded CPU/CUDA profiling and a final per-token training objective.
  The latter is distinct from validation CE.

## Environment and build

Actual host: `aarch64`, one GB10 `(12,1)`, NVIDIA driver `580.178.04`.
Actual container: torch `2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, Triton `3.8.0`.
The entrypoint selected CUDA forward compatibility with user-space driver
`615.71.09`. `TORCH_CUDA_ARCH_LIST=12.0+PTX` and
`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` were retained.

Base tag: `nvcr.io/nvidia/pytorch:26.09-py3`.
Pulled OCI index: `sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6`.
ARM64 manifest: `sha256:307c449df7dabed6b36e017a2b2f2e45bdec9dc203b81507111315f1e971de6e`.
Full-run image ID: `sha256:276a99265d501f4d9855307c9da5326d7d59ef63765e7b195b7e64027ed4b5c7`.

Executed successfully on the real device host:

```bash
uname -m
nvidia-smi
docker buildx imagetools inspect nvcr.io/nvidia/pytorch:26.09-py3
docker build --platform linux/arm64 -f Dockerfile.gb10 -t modded-nanogpt:92-gb10 .
```

Both the untouched base and project image fail `python -m pip check` with the
same four pre-existing metadata findings. See [GB10.md](../../GB10.md) for details.
No CUDA/PyTorch/CUTLASS replacements were made to silence optional-backend warnings.

## Static and operator checks

The host has no `python` alias and its `python3` lacks torch and NumPy. Host
`python3 -m compileall` passed, but the host unit suite reported two dependency
errors and two skips. The complete checks were rerun in the intended image:

```bash
docker run --rm --entrypoint bash \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 -lc \
  "python -m compileall -q train_gpt.py track_1_short tools tests && python -m unittest discover -s tests -p 'test*.py' -v"

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 python tools/gb10_kernel_probe.py \
  --capture-attention --benchmark-refresh
```

Results: 10 unit tests passed; all expanded native operator checks passed.
See [raw CPU output](cpu-tests.txt) and [raw operator/benchmark output](operators-and-refresh-benchmark.txt).
Packed QKV checks were also rerun with seeds `0`, `1`, `1234`: all 30 cases passed
([raw output](qkv-multiseed.txt)).

The first capture probe failed because it warmed autograd on the default stream;
its warmup now uses the capture stream. A later 6-head scalar-gradient comparison
exposed an incorrect idealized reference: the operator uses quantized intermediate
gradients and original BF16 weights for that scalar. The corrected reference
explicitly models those operations; it also checks input/weight gradients within
5% relative L2 against the quantized backward, retaining the independent idealized
18% bound. No training QKV math was changed to make the comparison pass.

A direct multi-target boundary check then reproduced another CE bug: at the last
row of a three-target microbatch, the non-target gradient was approximately 1.75x
its forward-objective derivative. The fused kernel summed all MTP weights even
when a future target did not exist. The normalizer gradient now sums only valid
future positions. All 12 combinations of widths 50304/10240/14336/24576 and MTP
counts 1/2/3 pass an independent autograd loss/elementwise-gradient reference,
including repeated targets and optional prefix targets. FP8 e5m2 per-element
bounds are rtol 0.16 and atol 2e-6 (1e-6 for non-target tail columns).
See [before-fix failure](ce-mtp-boundary-before.txt) and
[complete post-fix native probe](operators-after-mtp-fix.txt).
The original full-run attempt was intentionally stopped soon after entering its
timed loop to apply this correction; it is not a completed training result.

Other tolerances are documented in [GB10.md](../../GB10.md) and in the probe.
The kernel probe is not an end-to-end convergence result.

## Isolated refresh timing

Same seeded process, same default-sized MLP and LM-head weights, alternating
non-Adam/Adam refresh pairs; median of five groups of 100 pairs. CUDA event times
are divided by two to report per logical optimizer update.

| Refresh mode | Before (ms/update) | After (ms/update) |
|---|---:|---:|
| Eager | 1.546425 | 0.406287 |
| Captured two-update workload | 1.599426 | 0.403309 |

The captured refresh workload is about 74.8% cheaper. It excludes attention,
forward/backward, optimizer, data traffic and validation. This is **not** a measured
74.8% improvement in training throughput. The unmodified baseline cannot start
training on this GB10 because its CE kernel is compiled for sm90, so there is no
valid end-to-end before/after comparison for that source revision.

## Bounded smoke run

Executed:

```bash
docker run --rm --name nanogpt-gb10-smoke --gpus all --ipc=host --ulimit memlock=-1 \
  -e NUM_SCHEDULED_ITERATIONS=4800 -e NUM_EXTENSION_ITERATIONS=32 -e TRAIN_SEED=1234 \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 bash run.sh
```

- 4,800 scheduled + 832 growth + 32 extension = **5,664 updates**, 102,408,192 training tokens.
- All six training graph configurations, ANVIL graphs, reset/address checks,
  stage transitions, sampled/full CE, sparse updates, tail averaging and validation completed.
- Full 10,485,760-token validation CE: **3.6167** (target 3.28 not met).
- Charged training time: **1,789.051 s**, approximately **57,242 training tokens/s**.
- Process peak memory, including warmup/validation: **37,876 MiB allocated**, **48,694 MiB reserved**.
- Raw source/environment/run log: `logs/e462a49d-bcb8-418b-837f-2ee7fb1f0afd.txt`.

This smoke predates the final empty-graph removal, final-objective logging and
profiling hook and the MTP boundary-gradient fix; the final refresh graph and
corrected CE were tested independently, and the full run below uses the final implementation. Concurrent diagnostic checks/builds also ran
during the smoke, so its timing is a bring-up observation, not a controlled speed comparison.
The shorter schedule is not a default-run quality result.

An earlier 992-update attempt was rejected by the existing tail-average assertion:
the default divisor requires at least `336*16 + 250 = 5626` total steps.
No schedule/averaging rule was weakened to accommodate an artificially short run.

## Default full run

Launched at `2026-10-17T08:52:55Z` with the default 17,952 scheduled + 832 growth +
320 extension = **19,104 updates**, **328,626,176 training tokens**, default n-gram
vocabulary 10,575,360, default validation batch 131,072 and validation tokens 10,485,760.

```bash
/usr/bin/time -f 'elapsed_wall_seconds=%e\nexit_status=%x' -o /tmp/gb10-full-walltime.log \
  docker run --rm --name nanogpt-gb10-full --gpus all --ipc=host --ulimit memlock=-1 \
  -e TRAIN_SEED=1234 -e GB10_PROFILE_START=11000 -e GB10_PROFILE_STEPS=8 \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 bash run.sh
```

Raw source/environment/run log: `logs/dfea5293-a367-40d1-a568-603e76a79360.txt`;
raw process console and wall time: `/tmp/gb10-full-console.log` and
`/tmp/gb10-full-walltime.log` on the host. The process exited 0.

Results:

- 19,104 updates and 328,626,176 training tokens completed.
- Final training objective: **3.203940 per token**.
- Full 10,485,760-token validation CE: **3.5956**; the target of 3.28 was **not met**.
- Repository training clock: **5,494.381 s** (91m 34s), average **287.60 ms/update**,
  or about **59,811 training tokens/s**. This clock charges the eight profiled steps
  and excludes warmup and validation model forward.
- External wall time: **5,982.27 s** (99m 42s), including image/container startup,
  compilation, graph capture, training and validation.
- Peak CUDA memory: **37,889 MiB allocated**, **48,586 MiB reserved**.

The eight-step profile at updates 11,000–11,007 sampled stage 2. Its largest
individual device-time entries were BF16 CUTLASS GEMMs (9.90% and 8.07% each),
an SM121 FP8/TMA GEMM (7.23%), fused MLP activation/backward kernels (5.29% and
3.72%), and a FlexAttention backward kernel (2.74%). The table total is 2.853 s
of device activity across overlapping kernels in the eight-step window; these
percentages describe this profile sample only, not a full-run breakdown.
Trace: `logs/dfea5293-a367-40d1-a568-603e76a79360.rank0.profile.json`.

The earlier smoke run was a separate, shorter schedule and predates the final
boundary-gradient correction. The default result above uses the corrected kernel.
There is no valid end-to-end before/after speed comparison because the original
sm90 CE source cannot run on this GB10. Current run time is an absolute result,
not an official speedrun claim.

## Remaining limitations

- FineWeb validation CE 3.5956 exceeds the task target 3.28.
- `pip check` reports four dependency metadata issues also present in the untouched
  base image; details are in [GB10.md](../../GB10.md).
- Optional FA2/FA4 and FP8 MLP modes were not validated as training backends.
- This remains one microbatch and one optimizer update per step, with no gradient
  accumulation or eight-H100-equivalent optimizer trajectory.

Recommended next work is to investigate quality-affecting differences (especially
the smaller hashed n-gram table and the corrected boundary gradients) before any
claim of reproducing the 3.28 target, then run a clean timing pass without profiling
if a comparable throughput number is needed.
