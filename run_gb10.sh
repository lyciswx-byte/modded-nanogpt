#!/usr/bin/env bash
set -euo pipefail
export NANOGPT_TARGET=gb10
export NANOGPT_ATTENTION_BACKEND="${NANOGPT_ATTENTION_BACKEND:-fa2}"
export GB10_MLP_BACKEND="${GB10_MLP_BACKEND:-bf16}"
export GB10_MICROBATCH_DIVISOR="${GB10_MICROBATCH_DIVISOR:-2}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec torchrun --standalone --nproc_per_node=1 train_gpt.py
