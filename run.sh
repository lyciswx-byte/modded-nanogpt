#!/usr/bin/env bash
set -euo pipefail

# GB10: one ARM64 Grace CPU + one Blackwell sm_121 GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "[ERROR] Expose exactly one GB10 GPU." >&2
  exit 2
fi
# Use CUDA Toolkit ptxas when supplied by NGC / DGX OS.
if [[ -z "${TRITON_PTXAS_PATH:-}" ]] && command -v ptxas >/dev/null 2>&1; then
  export TRITON_PTXAS_PATH="$(command -v ptxas)"
fi
python scripts/preflight_gb10.py
# Never launch legacy H100 FA3 on GB10; trainer imports GB10 FlexAttention.
exec python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py "$@"
