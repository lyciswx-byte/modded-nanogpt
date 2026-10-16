#!/usr/bin/env python3
"""Read-only preflight for the Record #89 single-H100 development branch.

This script intentionally does not import train_gpt.py: that file initializes CUDA,
loads FlashAttention and starts training at module import time.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import sys

MAGIC = 20240520
VERSION = 1
HEADER_BYTES = 256 * 4
RECORD89_VAL_TOKENS = 10_485_760
# The default 1270+15 step schedule processes roughly 339 million training
# tokens. Require a little margin for shard boundaries / BOS-aligned batches.
MIN_TRAIN_TOKENS = 350_000_000


class PreflightError(RuntimeError):
    """An actionable prerequisite check failed."""


def inspect_token_file(path: Path) -> int:
    """Check the GPT-2 FineWeb binary header and return the token count."""
    with path.open("rb") as handle:
        raw = handle.read(12)
    if len(raw) != 12:
        raise PreflightError(f"Invalid/short FineWeb shard header: {path}")
    magic, version, count = struct.unpack("<iii", raw)
    if (magic, version) != (MAGIC, VERSION) or count <= 0:
        raise PreflightError(
            f"Unexpected FineWeb header {magic=}, {version=}, {count=}: {path}"
        )
    expected = HEADER_BYTES + 2 * count
    actual = path.stat().st_size
    if actual != expected:
        raise PreflightError(
            f"Incorrect FineWeb shard length: {path} ({actual} bytes, expected {expected})"
        )
    return count


def check_data_files(
    root: Path,
    *,
    min_train_tokens: int = MIN_TRAIN_TOKENS,
    min_val_tokens: int = RECORD89_VAL_TOKENS,
) -> tuple[int, int, int]:
    data_dir = root / "data" / "fineweb10B"
    shards = sorted(data_dir.glob("fineweb_train_*.bin"))
    if len(shards) < 2:
        raise PreflightError(
            f"Expected at least two training shards in {data_dir}; "
            "run: python data/cached_fineweb10B.py 9"
        )
    for i, shard in enumerate(shards, start=1):
        expected = f"fineweb_train_{i:06d}.bin"
        if shard.name != expected:
            raise PreflightError(
                f"Training shards must be contiguous, starting at 000001: "
                f"expected {expected}, got {shard.name}"
            )
    val_path = data_dir / "fineweb_val_000000.bin"
    if not val_path.is_file():
        raise PreflightError(
            f"Validation shard missing: {val_path}; "
            "run: python data/cached_fineweb10B.py 9"
        )

    training_tokens = sum(inspect_token_file(shard) for shard in shards)
    validation_tokens = inspect_token_file(val_path)
    if training_tokens < min_train_tokens:
        raise PreflightError(
            f"Only {training_tokens:,} training tokens; need at least "
            f"{min_train_tokens:,}. Run: python data/cached_fineweb10B.py 9"
        )
    if validation_tokens < min_val_tokens:
        raise PreflightError(
            f"Only {validation_tokens:,} validation tokens; "
            f"need {min_val_tokens:,}."
        )
    print(
        f"[OK] FineWeb shards: {len(shards)} train / 1 validation; "
        f"tokens: {training_tokens:,} train / {validation_tokens:,} validation"
    )
    if len(shards) < 9:
        print("[NOTE] The historical Record #89 README recommends nine training shards.")
    return len(shards), training_tokens, validation_tokens


def check_gpu() -> None:
    try:
        import torch
    except ImportError as exc:
        raise PreflightError("PyTorch not installed. Run: pip install -r requirements.txt") from exc

    print(f"[INFO] PyTorch {torch.__version__}; wheel CUDA {torch.version.cuda}")
    if not torch.__version__.startswith("2.10"):
        raise PreflightError(
            "Record #89 branch expects torch 2.10.x; install a matching CUDA wheel. "
            "Do not rely on an unpinned nightly."
        )
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is not available; check driver and CUDA-enabled PyTorch.")
    count = torch.cuda.device_count()
    if count != 1:
        raise PreflightError(
            f"Expected exactly one visible GPU, found {count}. "
            "Set CUDA_VISIBLE_DEVICES to a single GPU index (e.g. 0)."
        )
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    memory_gib = props.total_memory / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)
    print(f"[INFO] GPU: {name}; {memory_gib:.1f} GiB; SM {capability[0]}.{capability[1]}")
    if "H100" not in name.upper():
        raise PreflightError("This development branch is validated only for a single H100.")
    if memory_gib < 70:
        raise PreflightError(
            f"This plan targets H100 80GB; detected only {memory_gib:.1f} GiB."
        )
    free_bytes, _ = torch.cuda.mem_get_info(0)
    print(f"[INFO] Free GPU memory: {free_bytes / (1024 ** 3):.1f} GiB")
    if free_bytes < 50 * 1024 ** 3:
        print("[WARN] Less than 50 GiB GPU memory available; OOM is possible.")
    print("[OK] One H100 visible. Record #89 will accumulate eight microbatches per step.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Check CUDA first, before downloading FineWeb shards",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Validate FineWeb files without importing torch (useful in CI)",
    )
    opts = parser.parse_args()
    if not opts.skip_gpu:
        check_gpu()
    if not opts.skip_data:
        check_data_files(Path(os.environ.get("DATA_PATH", ".")).resolve())
    print("[OK] Single-H100 preflight complete. This is not a training success check.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (PreflightError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(2)
