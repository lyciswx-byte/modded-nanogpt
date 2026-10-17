#!/usr/bin/env bash
# Local (GB10) launcher for the 1xh100-dev Record #89 branch.
#
# This is NOT the branch's H100 launch path. scripts/preflight_1xh100.py is a
# hard H100 gate and rejects this machine (NVIDIA GB10, Blackwell sm_121), and
# Record #89 pins a Hopper-only FA3 kernel. The two [GB10 port] edits in
# train_gpt.py / triton_kernels.py swap the attention backend and the CE kernel
# build target so the same training code runs here; the optimizer, schedule and
# loss math are unchanged.
#
# Usage:
#   bash run_gb10.sh                      # full Record #89 schedule (1285 steps)
#   RECORD89_DEV_STEPS=2 bash run_gb10.sh # dev smoke test only
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-modded-nanogpt-anvil2:gb10}"
HF_CACHE="${HF_CACHE:-/tmp/hf_cache_record89}"
LOG="${LOG:-$REPO_DIR/record89_gb10.log}"

mkdir -p "$HF_CACHE"

echo "[run_gb10] image=$IMAGE log=$LOG RECORD89_DEV_STEPS=${RECORD89_DEV_STEPS:-<unset, full schedule>}"

exec docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO_DIR":/workspace \
  -v "$HF_CACHE":/hf_cache \
  -w /workspace \
  -e HF_HOME=/hf_cache \
  -e DATA_PATH=/workspace \
  -e TIKTOKEN_CACHE_DIR=/hf_cache/tiktoken_cache \
  -e TORCHINDUCTOR_COMPILE_THREADS=4 \
  -e TORCHINDUCTOR_CACHE_DIR=/hf_cache/inductor \
  -e TRITON_CACHE_DIR=/hf_cache/triton \
  ${RECORD89_DEV_STEPS:+-e RECORD89_DEV_STEPS=$RECORD89_DEV_STEPS} \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
