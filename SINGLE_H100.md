# Record #92 on one H100 (experimental port)

This branch runs the **record #92 / PR #360 ANVIL2** model on **one NVIDIA H100 (80 GB recommended)**. It retains the FP8 forward/backward paths, sampled softmax, ANVIL, hash-based bigram/trigram embeddings, sparse row-wise Adam, validation logic, and training/optimizer CUDA graphs.

**This is not a reproduced 39.9-second record.** The official record was measured on **8 × H100** with **84,602,880 n-gram rows**. A complete BF16 table has approximately **65 billion elements / 130 GB**, so it cannot reside in one 80 GB H100. No single-GPU validation loss or timing has been measured for this branch. The record's **FineWeb validation loss <= 3.28 is a target, not a verified result**.

## What the single-GPU adaptation changes

| Parameter | Official record #92 (8 GPUs) | This branch (1 GPU) |
|---|---:|---:|
| torchrun workers | 8 | 1 |
| global training tokens per optimizer step, stage 0/1/2/taper/extension | 131072 / 262144 / 393216 / 327680 / 131072 | 16384 / 32768 / 49152 / 40960 / 16384 |
| scheduled + growth + extension steps | 1122 + 52 + 20 = 1194 | 8976 + 416 + 160 = 9552 |
| total training tokens | approximately the original budget | approximately the original budget (more, smaller updates) |
| n-gram table rows (bigram + trigram) | 84,602,880, sharded across 8 GPUs | 10,575,360, stored on 1 GPU |
| validation target | FineWeb CE <= 3.28 | same evaluation target, **unverified** |

The per-GPU **microbatch geometry** is kept similar to the original eight-GPU run. Multiplying the step count by eight keeps the *approximate number of training tokens*, **not** the original global batch size, optimization trajectory or model capacity. There is no hidden 8-way gradient accumulation. The smaller hashed embedding table increases collisions and could affect final validation loss. The original 8-GPU path remains available by launching with eight workers.

## Environment and run

Prerequisites: one H100 with enough free VRAM (80 GB recommended), recent NVIDIA driver (>= 580 according to upstream PR #360), CUDA 13 runtime for the custom FlashAttention-3 kernel, the pinned PyTorch/CUDA environment, and sufficient host storage for FineWeb data.

```bash
git clone https://github.com/lyciswx-byte/modded-nanogpt.git
cd modded-nanogpt
git checkout 92-1xH100
pip install -r requirements.txt
pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128

# Fetch first 900M train tokens plus validation shards, as in the upstream instructions.
python data/cached_fineweb10B.py 9

# Single-GPU launcher: torchrun --standalone --nproc_per_node=1 train_gpt.py
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

As in the upstream README, allow several minutes for the initial compilation and CUDA graph captures. See upstream instructions for Docker and authentication/offline caching of `devenpzak/flash-attn3-12864`. Verify `nvidia-smi` exposes only the intended GPU before running.

To run a quick CPU-only configuration/communication test (no H100 required):

```bash
python -m unittest discover -s tests -p 'test_single_h100.py' -v
```

The *only* benchmark that establishes success is a **complete** H100 run that reports final `val_loss <= 3.28`; note the measured training time separately and preserve the generated logs. Neither a successful import nor a CUDA warmup proves convergence.

## Configuration and memory

The launcher supplies `WORLD_SIZE=1` through torchrun. The table defaults to `NGRAM_VOCAB_SIZE=10575360`, requiring approximately 16.2 GB for its BF16 weights, plus optimizer state, model weights/graphs/activations and caches. The original full-size table **will not fit** on a single H100.

Available environment overrides:

- `NGRAM_VOCAB_SIZE`: number of n-gram rows (must be positive, even; default 10,575,360 for 1 GPU, 84,602,880 for 8 GPUs). Increasing it increases VRAM usage; a larger value is **not** automatically more accurate.
- `NUM_SCHEDULED_ITERATIONS`: base count of scheduled steps (single-GPU default 8976, eight-GPU default 1122).
- `NUM_EXTENSION_ITERATIONS`: extension steps (single-GPU default 160, eight-GPU default 20).
- `VAL_BATCH_SIZE`: global validation batch tokens (default 262144 on 1 GPU, 2097152 on 8 GPUs). Must divide 10485760 and be divisible by the world size. Reducing to 131072 or 65536 can reduce the eval batch footprint; the n-gram cache also accounts for the larger training update cycle.
- `LR_SCALE`: multiplies ANVIL and Adam base learning rates (default 1.0); can be tuned if single-GPU convergence differs.
- `TRAIN_SEED`: set an integer to make parameter initialization reproducible. More than one training run is needed to estimate reliability near the 3.28 threshold.

Example lower-memory evaluation (not validated for peak VRAM or convergence):

```bash
VAL_BATCH_SIZE=131072 TRAIN_SEED=42 CUDA_VISIBLE_DEVICES=0 bash run.sh
```

**Limitations:** Unlike the original multi-GPU record, this port cannot preserve the full 65B-parameter n-gram table in H100 VRAM. It changes global batch size and number of optimizer updates. Time-to-target, memory high-water mark, and the validation loss require H100 testing and tuning. This document makes no claim of an official speedrun record.
