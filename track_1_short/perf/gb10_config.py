"""Minimal environment switches shared by GB10 model and training code.

All switches are fixed at process start; changing a switch during compiled
training/CUDA graph replay is intentionally unsupported.
"""
import os


def gb10_target() -> bool:
    return os.environ.get("NANOGPT_TARGET", "").lower() == "gb10"


def gb10_mlp_backend() -> str:
    backend = os.environ.get("GB10_MLP_BACKEND", "bf16" if gb10_target() else "fp8").lower()
    if backend not in ("bf16", "fp8", "fp8_sm121"):
        raise ValueError("GB10_MLP_BACKEND must be bf16, fp8, or fp8_sm121")
    if backend == "fp8_sm121" and not gb10_target():
        raise ValueError("fp8_sm121 is supported only for NANOGPT_TARGET=gb10")
    return backend


def gb10_microbatch_divisor() -> int:
    default = "2" if gb10_target() else "1"
    divisor = int(os.environ.get("GB10_MICROBATCH_DIVISOR", default))
    if divisor not in (1, 2, 4, 8):
        raise ValueError("GB10_MICROBATCH_DIVISOR must be 1, 2, 4 or 8")
    return divisor
