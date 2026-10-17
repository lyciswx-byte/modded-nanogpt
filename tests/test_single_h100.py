"""CPU-only checks for the single-H100 port (no CUDA or FineWeb required)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Execute in a fresh process so world_size-based config is recalculated on import.
CONFIG_DUMP = """
import json
from track_1_short.config import (Hyperparameters, TRAINING_STAGES,
    SCHEDULE_GROWTH_STEPS, SINGLE_GPU_STEP_SCALE, TRAIN_WORLD_SIZE)
h = Hyperparameters()
print(json.dumps(dict(world=TRAIN_WORLD_SIZE, scale=SINGLE_GPU_STEP_SCALE,
    batches=[s.batch_size for s in TRAINING_STAGES], val=h.val_batch_size,
    scheduled=h.num_scheduled_iterations, growth=SCHEDULE_GROWTH_STEPS,
    extension=h.num_extension_iterations)))
"""


def get_config(world_size: int):
    env = os.environ.copy()
    # These tests exercise the H100-port defaults even inside a GB10 Docker
    # container where NANOGPT_TARGET=gb10 is part of the image environment.
    env.pop("NANOGPT_TARGET", None)
    env.pop("GB10_MICROBATCH_DIVISOR", None)
    for name in ("NUM_SCHEDULED_ITERATIONS", "NUM_EXTENSION_ITERATIONS", "VAL_BATCH_SIZE"):
        env.pop(name, None)
    env["WORLD_SIZE"] = str(world_size)
    proc = subprocess.run([sys.executable, "-c", CONFIG_DUMP], cwd=ROOT,
                          env=env, text=True, capture_output=True, check=True)
    return json.loads(proc.stdout)


class TestSingleH100Config(unittest.TestCase):
    def test_single_gpu_shapes_and_total_steps(self):
        c = get_config(1)
        self.assertEqual(c["world"], 1)
        self.assertEqual(c["scale"], 8)
        self.assertEqual(c["batches"], [16384, 32768, 49152, 40960, 16384])
        self.assertEqual(c["val"], 262144)
        self.assertEqual(c["scheduled"] + c["growth"] + c["extension"], 9552)

    def test_eight_gpu_original_mode_and_token_budget(self):
        one, eight = get_config(1), get_config(8)
        self.assertEqual(eight["scale"], 1)
        self.assertEqual(eight["batches"], [131072, 262144, 393216, 327680, 131072])
        self.assertEqual(eight["scheduled"] + eight["growth"] + eight["extension"], 1194)
        self.assertEqual(one["batches"], [b // 8 for b in eight["batches"]])
        self.assertEqual(one["scheduled"], eight["scheduled"] * 8)
        self.assertEqual(one["growth"], eight["growth"] * 8)
        self.assertEqual(one["extension"], eight["extension"] * 8)
        self.assertEqual(one["val"], eight["val"] // 8)

    def test_single_rank_sparse_row_communication_works_without_process_group(self):
        import numpy as np
        import torch
        from track_1_short.sharded_rows import RowOwners

        owners = RowOwners(num_rows=16, rank=0, world_size=1)
        ids = np.array([1, 3, 7], dtype=np.int32)
        want = torch.from_numpy(ids.copy())
        pull = owners.request(want, ids)
        owners.exchange_ids(pull)
        self.assertEqual((pull.lo, pull.hi), (0, 3))
        self.assertEqual(pull.recv_counts, [0])
        shard = torch.arange(48, dtype=torch.float32).view(16, 3)
        received, work, served = owners.start_fetch(pull, shard)
        work.wait()
        self.assertEqual(tuple(received.shape), (0, 3))
        self.assertEqual(tuple(served.shape), (0, 3))
        rows, work = owners.send_grads(pull, torch.empty((0, 3)))
        work.wait()
        self.assertEqual(tuple(rows.shape), (0, 3))


if __name__ == "__main__":
    unittest.main()
