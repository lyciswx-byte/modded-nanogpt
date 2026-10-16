# Record #89 on one NVIDIA H100 (development branch)

> **Status:** Single-H100 launcher, read-only preflight, and CPU-only unit tests
> are implemented. The complete GPU training run, final validation loss and
> wall-clock performance **have not yet been measured** on an H100 in this branch.
> Do not confuse this adaptation with a certified speedrun record.

## Why this branch starts from history

- Branch: `1xh100-dev`
- Baseline commit: `75d6867a4adb4eaea071352444e8e1d37e308855`
- Baseline: the original **Record #89** trainer, after its FP8 MLP
  down-projection merge and before the #90/#91/#92 replacements.
- Upstream 8xH100 Record #89 is approximately **1.23 min (~73.8 s)**
  to FineWeb validation cross-entropy <= 3.28. That time **is not a
  prediction for one GPU**.
- The modern `master` / `wxdev` branch contains **ANVIL2 (#92)** and is
  intentionally not used as the starting point here.

We keep the historical model, optimizer, kernels, training schedule and loss
math unchanged. The only addition to `train_gpt.py` is an **opt-in**
`RECORD89_DEV_STEPS` early-stop guard for smoke tests; with that variable unset,
the default 1285-step Record #89 training path is unchanged.

## What changes on one GPU?

The Record #89 trainer already contains these original lines:

```python
assert 8 % world_size == 0
grad_accum_steps = 8 // world_size
grad_scale = 1 / grad_accum_steps
```

When launched with one process, `grad_accum_steps=8`, compared with one
microbatch per step on eight processes. Each microbatch on the single H100
contains the same nominal number of tokens as each rank's microbatch on the
8xH100 run.

| Stage | Global tokens/optimizer step | GPUs | Accumulation | Tokens/microbatch |
|:--|--:|--:|--:|--:|
| 0 (batch 8) | 131,072 | 1 | 8 | 16,384 |
| 1 (batch 16) | 262,144 | 1 | 8 | 32,768 |
| 2/extension (batch 24) | 393,216 | 1 | 8 | 49,152 |

The optimizer has a built-in `world_size == 1` path: its communication
policies become `none`; the historical 8-rank-only sparse bigram
gradient exchange is disabled automatically. The bigram embedding gradient
is therefore **dense locally**, which may make the one-GPU step slower and
use more GPU memory. The actual optimizer is still **NorMuon + Adam**, NOT
ANVIL2. Global batch size, optimizer hyperparameters, train/eval token
streams, and default stages remain the historical #89 settings.

The numerical computation is not bit-identical to eight GPUs: gradient
reduction ordering, FP8 rounding, and optimizer shard/lane layout can
differ. Validate convergence empirically rather than assuming identical loss.

## Hardware and runtime prerequisites

- Exactly **one visible NVIDIA H100 80GB**; Hopper architecture, driver and
  CUDA-enabled PyTorch must be functional.
- Python compatible with **PyTorch 2.10.x** and the historical code.
- A working CUDA toolchain, Triton, `kernels`, and the third-party
  `kernels-community/flash-attn3` kernel used by this historical trainer.
- Sufficient free host RAM and disk space for the FineWeb download,
  just-in-time compilation, and run logs.
- This branch's inherited `Dockerfile` installs an **unpinned nightly**;
  it is retained for historical reference and is **not a validated or
  deterministic single-H100 environment**. For reproducibility use a
  separately verified CUDA/PyTorch stack, and check `torch.__version__`.

Suggested setup in a CUDA-capable environment with an installed NVIDIA driver:

```bash
git clone -b 1xh100-dev https://github.com/lyciswx-byte/modded-nanogpt.git
cd modded-nanogpt

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Check that the installed torch is 2.10.x with CUDA support.
# If a compatible CUDA wheel is not installed, choose the official
# 2.10.x CUDA wheel matching your driver/runtime.
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_1xh100.py --skip-data
```

If the FA3 kernel cannot be resolved, verify `kernels` / Hugging Face
access and the compatible CUDA wheel. Do not silently switch to #92's
patched FlashAttention-3 kernel; it targets a different attention geometry.

## Dataset and training

Download FineWeb10B GPT-2 token shards, using the same historical data
pipeline as Record #89:

```bash
python data/cached_fineweb10B.py 9
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_1xh100.py
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

`run.sh` runs the preflight, then launches:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
```

Using `python -m` ensures that the launcher and installed `torch` come
from the same virtual environment. If you want GPU 2 rather than GPU 0,
use `CUDA_VISIBLE_DEVICES=2 bash run.sh`. Never expose two or more GPUs
to this particular launcher.

For a **development-only 2-step GPU smoke run**, use:

```bash
RECORD89_DEV_STEPS=2 CUDA_VISIBLE_DEVICES=0 bash run.sh
```

This still performs the historical warmup/compilation and final validation;
it is useful for surfacing CUDA/optimizer failures, **not** for judging
Record #89 training quality. Omit `RECORD89_DEV_STEPS` for the full run.
The full default #89 schedule remains **1270 main steps + 15 extension
steps = 1285 optimizer steps**. `val_loss_every=250` in the original code,
and a full FineWeb validation also runs at the end. The target evaluation
uses 10,485,760 validation tokens and the **full vocabulary**.

To put downloaded data elsewhere, set `DATA_PATH=/path/to/repo-root`
where the FineWeb files are under
`$DATA_PATH/data/fineweb10B/fineweb_*.bin`.

The historical code defaults to `save_checkpoint=False`. It writes
source code and training progress to `logs/<uuid>.txt`; it does NOT
save a reusable model checkpoint unless you deliberately alter the
historical checkpoint setting (and supply sufficient storage).

## Lightweight checks before a costly H100 run

```bash
# CPU-only structural tests; no GPU or downloaded dataset required:
python -m unittest discover -s tests -p 'test_preflight_1xh100.py' -v

# Only GPU/PyTorch check (before dataset download):
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_1xh100.py --skip-data

# Full GPU + FineWeb header/size/count check:
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_1xh100.py
```

These tests verify setup and data format; they do NOT verify the complete
training graph, gradients, CUDA kernels, memory peak, final loss or speed.

## GPU acceptance protocol

1. Confirm one H100 visible, PyTorch 2.10.x and the historical FA3 load.
2. Run the historical kernel/model compilation and warmup, and verify no
   illegal memory accesses, OOM, NaN or distributed-process failures.
3. Verify `grad_accum_steps=8`, and that optimizer steps 0 and 1 finish.
4. Run the full 1285-step training schedule and collect each logged
   validation loss plus peak allocated/reserved GPU memory.
5. Repeat at least three times and compare **mean** final validation CE
   with the 3.28 threshold. The official speedrun standard additionally
   requires statistical significance, not just a single passing run.
6. Measure end-to-end time separately from the benchmark's timed training
   section; compile and warmup are excluded by the benchmark convention.

### What to do if OOM occurs

Do NOT immediately reduce the global batch size or remove model layers:
that would cease to be a faithful #89 comparison. First inspect whether
OOM occurs during warmup/validation, forward/backward, the optimizer's
full (rather than sharded) states, or another process's memory usage.
Log the failing stage, peak memory, and stack trace. The next adaptation
step should be a targeted memory optimization that preserves the same
effective batch and training loss, followed by a convergence test.

### Important limitations

- On one H100, there is no 8-way parallelism and no inter-GPU
  reduce-scatter speedup. It will NOT achieve the 8xH100's 73.8-second time.
- A single GPU's local optimizer state and dense bigram gradients differ
  operationally from the 8-rank sharded/sparse layout.
- The branch is a **working launch-path adaptation and untested GPU
  implementation**, not a claim of a successful or fast H100 reproduction.
- Any change to optimizer kernels or the training schedule should land in
  separate commits and be measured against this unmodified #89 baseline.

## Pointers

- [Historical baseline commit](https://github.com/lyciswx-byte/modded-nanogpt/commit/75d6867a4adb4eaea071352444e8e1d37e308855)
- [Record #89 run logs and statistics](https://github.com/lyciswx-byte/modded-nanogpt/tree/1xh100-dev/records/track_1_short/2026-07-17_FP8DownProjection/this_pr)
- [Development branch](https://github.com/lyciswx-byte/modded-nanogpt/tree/1xh100-dev)
