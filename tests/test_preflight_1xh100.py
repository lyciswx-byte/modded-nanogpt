"""CPU-only tests for the Record #89 single-H100 data preflight."""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from scripts.preflight_1xh100 import (
    HEADER_BYTES,
    MAGIC,
    VERSION,
    PreflightError,
    check_data_files,
    inspect_token_file,
)


def fake_shard(path: Path, count: int, *, magic: int = MAGIC, extra: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<iii", magic, VERSION, count)
    with path.open("wb") as f:
        f.write(header)
        f.write(bytes(HEADER_BYTES - len(header)))
        f.write(bytes(2 * count + extra))


class FineWebPreflightTests(unittest.TestCase):
    def test_reads_valid_shard(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.bin"
            fake_shard(path, 19)
            self.assertEqual(inspect_token_file(path), 19)

    def test_rejects_bad_magic(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.bin"
            fake_shard(path, 19, magic=0)
            with self.assertRaises(PreflightError):
                inspect_token_file(path)

    def test_rejects_wrong_length(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.bin"
            fake_shard(path, 19, extra=2)
            with self.assertRaises(PreflightError):
                inspect_token_file(path)

    def test_checks_contiguous_shards_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "data" / "fineweb10B"
            fake_shard(data / "fineweb_train_000001.bin", 11)
            fake_shard(data / "fineweb_train_000002.bin", 12)
            fake_shard(data / "fineweb_val_000000.bin", 18)
            stats = check_data_files(root, min_train_tokens=20, min_val_tokens=15)
            self.assertEqual(stats, (2, 23, 18))

    def test_rejects_shard_numbering_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "data" / "fineweb10B"
            fake_shard(data / "fineweb_train_000001.bin", 11)
            fake_shard(data / "fineweb_train_000003.bin", 12)
            fake_shard(data / "fineweb_val_000000.bin", 18)
            with self.assertRaises(PreflightError):
                check_data_files(root, min_train_tokens=20, min_val_tokens=15)

    def test_rejects_insufficient_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "data" / "fineweb10B"
            fake_shard(data / "fineweb_train_000001.bin", 11)
            fake_shard(data / "fineweb_train_000002.bin", 12)
            fake_shard(data / "fineweb_val_000000.bin", 18)
            with self.assertRaises(PreflightError):
                check_data_files(root, min_train_tokens=30, min_val_tokens=15)


if __name__ == "__main__":
    unittest.main()
