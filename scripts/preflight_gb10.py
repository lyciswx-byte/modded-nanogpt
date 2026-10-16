#!/usr/bin/env python3
"""Read-only NVIDIA GB10 / DGX Spark preflight for historical Record #89."""
from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

from preflight_1xh100 import PreflightError, check_data_files


def is_gb10_device(name: str, capability: tuple[int, int]) -> bool:
    return "GB10" in name.upper() and capability == (12, 1)


def check_runtime() -> None:
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise PreflightError(
            f"GB10 Grace CPU requires ARM64/aarch64; got {platform.machine()}."
        )
    try:
        import torch
    except ImportError as exc:
        raise PreflightError(
            "PyTorch is missing. Use the NVIDIA NGC PyTorch ARM64 container "
            "(see README_1xGB10.md); do not install an x86-only wheel."
        ) from exc
    if not torch.cuda.is_available():
        raise PreflightError("CUDA unavailable: check the NVIDIA driver/container runtime.")
    if torch.cuda.device_count() != 1:
        raise PreflightError(
            "Expose exactly one GB10 GPU using CUDA_VISIBLE_DEVICES=0."
        )
    props = torch.cuda.get_device_properties(0)
    name = props.name
    cap = torch.cuda.get_device_capability(0)
    if not is_gb10_device(name, cap):
        raise PreflightError(f"Expected GB10 sm121; got {name!r}, capability={cap}.")
    version = torch.__version__
    if not version.startswith("2.10"):
        raise PreflightError(
            f"Use NGC 25.12's tested PyTorch 2.10.x stack before further ports; got {version}."
        )
    cuda = torch.version.cuda
    if cuda is None or int(cuda.split(".")[0]) < 13:
        raise PreflightError(f"GB10 path requires CUDA 13+, got {cuda!r}.")
    try:
        from torch.nn.attention.flex_attention import flex_attention, BlockMask
        del flex_attention, BlockMask
    except ImportError as exc:
        raise PreflightError("Installed PyTorch lacks FlexAttention.") from exc
    print(f"[OK] GB10 sm121 ARM64, torch={version}, CUDA={cuda}")
    print(f"[INFO] Unified system RAM: device-reported {props.total_memory / 2**30:.1f} GiB")
    print("[NOTE] GB10 uses CPU/GPU unified memory; nvidia-smi may omit GPU memory usage.")
    print("[NOTE] The GPU FlexAttention correctness test must pass before training.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-data", action="store_true",
                   help="Check device before downloading FineWeb data")
    p.add_argument("--skip-gpu", action="store_true",
                   help="Only check FineWeb files; useful without GB10 hardware")
    a = p.parse_args(argv)
    if not a.skip_gpu:
        check_runtime()
    if not a.skip_data:
        check_data_files(Path(os.environ.get("DATA_PATH", ".")).resolve())
    print("[OK] GB10 preliminary checks passed; not a training or numerical check.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PreflightError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)
