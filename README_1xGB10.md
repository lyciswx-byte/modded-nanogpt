# Record #89 adaptation: 1 NVIDIA GB10 / DGX Spark (experimental)

> **Status: code and automated CPU checks only.** The **SM121 attention kernel,
> complete model warmup, FP8/Triton optimizer, 1285 training steps, FineWeb
> loss and elapsed time have NOT been run on a GB10 in this development session.
> This is an implementation candidate, NOT a certified Record #89 result.**

## 1. Baseline and intent

- Derived from `1xh100-dev`, which itself was based on Record #89's
  historical code at `75d6867a4adb4eaea071352444e8e1d37e308855`.
- **Do not merge the current ANVIL2 / Record #92 master trainer into this branch.**
- Keep the original **NorMuon + Adam**, 11-layer model, MUDD connections,
  Bigram embedding, FP8 MLP forward, data pipeline, training cross-entropy,
  and 1270 + 15 optimizer-step schedule.
- Global optimizer-step token batch remains 131072 / 262144 / 393216
  (stages 8 / 16 / 24 units), achieved with **8 microbatches per step**
  on a single GPU. Do not divide learning rates by 8.

## 2. GB10 is not an H100 with more memory

| | Single H100 | Single GB10 / DGX Spark |
|---|---|---|
| CUDA architecture | Hopper, SM90 | Blackwell, SM121 |
| CPU platform | Often x86_64 | ARM64 Grace |
| Memory | ~80GB dedicated HBM | 128GB CPU/GPU coherent LPDDR5x |
| Memory bandwidth | High-bandwidth HBM | 273 GB/s system memory |
| Attention backend | Hopper FlashAttention-3 | experimental FlexAttention block-sparse |
| Software | H100 CUDA wheel / FA3 | CUDA 13+ ARM64 NVIDIA NGC image |

GB10's larger shared memory does **not** imply H100 throughput. CPU
processes and GPU allocations use the same RAM, and memory bandwidth is a
key limitation. The 8xH100 benchmark result (~73.8 seconds) cannot be
extrapolated to a 1xGB10 wall time.

Sources:
- https://developer.nvidia.com/cuda/gpus
- https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html
- https://docs.nvidia.com/deeplearning/transformer-engine/examples/attention/attention.html

## 3. Implemented changes

1. `gb10_attention.py`: replaces the Hopper-only
   `kernels-community/flash-attn3` varlen self-attention call with
   PyTorch FlexAttention. Builds block candidate indices directly
   (O(T*window_blocks), not O(T^2)), then uses an exact mask based on
   padded `cu_seqlens`, causal order and left-window distance. Preserves
   the special paired-head path because the original call site still
   transforms Q/K/V and doubles `cu_seqlens` before calling the backend.
2. `gb10_eval.py`: full-vocabulary evaluation logits are computed in
   rows of 2048 (configurable with `RECORD89_GB10_EVAL_CHUNK`), to avoid
   a huge `[262144, 50304]` tensor. This **does not** shorten the
   validation stream, break attention sequences or sample softmax; it
   implements the original full-vocabulary CE in smaller pieces.
3. `train_gpt.py`: GB10 backend import replaces FA3; uses the eager
   model by default, with compiled FlexAttention inside the adapter.
   `RECORD89_GB10_COMPILE=1` is an optional, unverified optimization,
   with relaxed `fullgraph=False` to permit graph breaks.
4. `run.sh` + `scripts/preflight_gb10.py`: one-device torchrun,
   ARM64+SM121+CUDA13 check and historical FineWeb shard validation.
5. `Dockerfile.gb10` + `requirements_gb10.txt`: use the official NVIDIA
   NGC PyTorch 25.12 image and **keep its own** CUDA 13.1/PyTorch 2.10a
   build. Do NOT `pip install -r requirements.txt`: it pins torch in a
   manner unsuitable for this platform.
6. `scripts/verify_gb10_attention.py`: checks attention forward and
   Q/K/V backward against a dense FP32 reference on small packed,
   multi-document batches, including window boundaries.
7. `tests/test_gb10_port.py`: offline static guards and optional
   CPU checks for varlen masks and chunked validation.

## 4. Install on a GB10 ARM64 machine

Recommended: NVIDIA PyTorch NGC container `25.12-py3`
(Python 3.12, PyTorch 2.10a, CUDA 13.1).

```bash
git clone -b 1xgb10 https://github.com/lyciswx-byte/modded-nanogpt.git
cd modded-nanogpt

docker build --platform linux/arm64 -f Dockerfile.gb10 -t modded-nanogpt:gb10 .
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace/modded-nanogpt \
  -w /workspace/modded-nanogpt \
  modded-nanogpt:gb10 bash
```

Within the container:

```bash
python scripts/preflight_gb10.py --skip-data
# Must test forward AND backward before training:
python scripts/verify_gb10_attention.py

python data/cached_fineweb10B.py 9
python scripts/preflight_gb10.py
RECORD89_DEV_STEPS=2 bash run.sh
```

If the attention verifier passes and the 2-step smoke run reaches validation,
run the entire Record #89 schedule:

```bash
unset RECORD89_DEV_STEPS
bash run.sh
```

The first run compiles many Triton and FlexAttention kernels. Keep sufficient
disk capacity for the CUDA/Triton cache, data and logs. Expect performance to
differ sharply from 8xH100 due to FP8 kernels, memory bandwidth and
accumulated single-device compute.

## 5. Controls and invariants

| Environment variable | Default | Purpose |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 0 | Expose exactly one GB10 |
| `RECORD89_DEV_STEPS` | unset | Opt-in early-stop smoke; **invalid benchmark result** |
| `RECORD89_GB10_EVAL_CHUNK` | 2048 | Full softmax output rows per chunk; reduce to 1024 on OOM |
| `RECORD89_GB10_COMPILE` | 0 | Experimental full model torch.compile; turn on only after baseline |
| `DATA_PATH` | . | Root containing `data/fineweb10B` |

The validation stream and attention sequence boundaries stay unchanged
even when the logits are chunked. The expected target is a **mean**
FineWeb validation cross entropy of <=3.28 over multiple full runs,
not the loss of the development smoke test.

### Potential blockers (require real GB10 validation)

- FlexAttention custom `BlockMask` tracing, backward, numerical agreement
  and throughput on SM121. The GPU verifier must PASS; do not silently
  substitute `EFFICIENT_ATTENTION` without numeric validation.
- Record #89's hand-tuned H100 Triton matmul, FP8 quantization and
  NorMuon optimizer kernels may need Blackwell-specific tuning or
  source changes. None is proven to compile or run correctly on SM121.
- 128GB is **shared** system memory. Even with chunked CE, the training
  optimizer states and full evaluation token activations may OOM.
- H100 FA3 / precompiled Hopper artifacts do not run on GB10; the custom
  FlexAttention backend changes floating-point reduction order. Do
  not assert bitwise-equivalent gradients or published timing.
- PyTorch image compatibility is platform/build-dependent. If a different
  release is used, redo the attention and FP8 correctness checks.
- `nvidia-smi` may report GPU memory usage as unsupported on GB10.
  Collect `torch.cuda.max_memory_allocated()` and host RAM measurements.
- A single smoke pass is insufficient to validate <=3.28; collect
  loss curves, multiple final losses and timing on the same GB10 machine.

Do not advertise this as an official speedrun entry until the GPU
acceptance tests, full run and statistical certification actually pass.

## 6. Troubleshooting

- **Hopper FlashAttention errors**: ensure you checked out `1xgb10`,
  not `1xh100-dev`; the GB10 trainer should import `gb10_attention`.
- **GPU capability mismatch**: confirm `uname -m` -> `aarch64`, and
  `torch.cuda.get_device_capability(0)` -> `(12,1)`.
- **FlexAttention kernel/grad error**: run
  `python scripts/verify_gb10_attention.py` alone, and preserve its trace.
  No success is claimed until this test actually runs on GB10.
- **PyTorch dependency overwritten**: rebuild the NGC image and use
  `requirements_gb10.txt`, not the legacy `requirements.txt`.
- **OOM in validation**: first try
  `RECORD89_GB10_EVAL_CHUNK=1024 bash run.sh`, which preserves all tokens.
- **OOM during optimizer/warmup**: collect memory trace before making
  model-shape changes; reducing model/batch would no longer strictly
  reproduce Record #89.

For original single-H100 historical analysis see [README_1xH100.md](README_1xH100.md).
