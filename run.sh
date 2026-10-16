#!/usr/bin/env bash
set -euo pipefail

# Preserve Record #89 training logic. Only change which GPU runs its eight
# gradient-accumulation microbatches. Use a single device ID, not '0,1,...'.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "[ERROR] Set CUDA_VISIBLE_DEVICES to exactly one H100 device ID." >&2
  exit 2
fi

# Verify the GPU and FineWeb shards BEFORE starting the expensive compile.
python scripts/preflight_1xh100.py

# python -m guarantees torchrun and torch are from the same Python environment.
exec python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py "$@"
