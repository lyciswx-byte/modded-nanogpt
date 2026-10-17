"""CPU-only correctness checks for portable GB10 operator math.

GPU/SM121 kernel support is NOT proved by these tests; run the real probe in Docker.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    import torch.nn.functional as F
    from track_1_short.perf.attention_backend import _flex_varlen_attention, BACKENDS
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch with FlexAttention is required")
class TestPortableFlexAttention(unittest.TestCase):
    def test_backend_registry(self):
        self.assertIn("flex", BACKENDS)
        self.assertIn("fa4", BACKENDS)
        self.assertIn("fa2", BACKENDS)

    def test_mixed_head_widths_and_document_windows(self):
        # Two packed documents; the causal mask must not cross their boundary.
        for qk_dim, v_dim in ((64, 128), (128, 64), (64, 64), (128, 128)):
            with self.subTest(qk=qk_dim, v=v_dim):
                torch.manual_seed(123)
                q = torch.randn(12, 2, qk_dim)
                k = torch.randn(12, 2, qk_dim)
                v = torch.randn(12, 2, v_dim)
                dim = max(qk_dim, v_dim)
                qp, kp = F.pad(q, (0, dim - qk_dim)), F.pad(k, (0, dim - qk_dim))
                vp = F.pad(v, (0, dim - v_dim))
                cu = torch.tensor([0, 5, 12], dtype=torch.int32)
                scale = qk_dim ** -0.5
                out = _flex_varlen_attention(qp, kp, vp, cu, scale, 3)[..., :v_dim]

                expected = []
                cursor = 0
                for length in (5, 7):
                    qs, ks, vs = [x[cursor:cursor + length].transpose(0, 1)
                                  for x in (q, k, v)]
                    scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
                    idx = torch.arange(length)
                    allow = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] <= 3)
                    expected.append((scores.masked_fill(~allow[None], -torch.inf)
                                     .softmax(-1) @ vs).transpose(0, 1))
                    cursor += length
                reference = torch.cat(expected)
                torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-4)
                # CPU FlexAttention is forward-only in NGC 26.09. The native
                # GPU probe compares actual forward AND gradients to this math.


class TestGB10Environment(unittest.TestCase):
    def test_sm121_backend_requires_gb10_target(self):
        from unittest.mock import patch
        from track_1_short.perf.gb10_config import gb10_mlp_backend
        with patch.dict(os.environ, {"NANOGPT_TARGET": "gb10",
                                     "GB10_MLP_BACKEND": "fp8_sm121"}):
            self.assertEqual(gb10_mlp_backend(), "fp8_sm121")
        with patch.dict(os.environ, {"NANOGPT_TARGET": "h100",
                                     "GB10_MLP_BACKEND": "fp8_sm121"}):
            with self.assertRaisesRegex(ValueError, "only for"):
                gb10_mlp_backend()
        with patch.dict(os.environ, {"NANOGPT_TARGET": "gb10",
                                     "GB10_MLP_BACKEND": "fake"}):
            with self.assertRaisesRegex(ValueError, "must be"):
                gb10_mlp_backend()


    def test_default_profile_does_not_modify_h100_profile(self):
        script = """
import json
from track_1_short.config import TRAINING_STAGES, SINGLE_GPU_STEP_SCALE, Hyperparameters
h=Hyperparameters()
print(json.dumps({'steps': h.num_scheduled_iterations + h.num_extension_iterations,
                  'growth': SINGLE_GPU_STEP_SCALE*52,
                  'batch': [s.batch_size for s in TRAINING_STAGES],
                  'val_batch': h.val_batch_size}))
"""
        import json
        def run(target):
            env = os.environ.copy()
            env["WORLD_SIZE"] = "1"
            if target is None:
                env.pop("NANOGPT_TARGET", None)
            else:
                env["NANOGPT_TARGET"] = target
            for k in ("GB10_MICROBATCH_DIVISOR", "NUM_SCHEDULED_ITERATIONS",
                      "NUM_EXTENSION_ITERATIONS", "VAL_BATCH_SIZE"):
                env.pop(k, None)
            p = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                               check=True, capture_output=True, text=True)
            return json.loads(p.stdout)
        h100, gb10 = run(None), run("gb10")
        self.assertEqual(h100["batch"], [16384, 32768, 49152, 40960, 16384])
        self.assertEqual(gb10["batch"], [8192, 16384, 24576, 20480, 8192])
        self.assertEqual(h100["steps"] + h100["growth"], 9552)
        self.assertEqual(gb10["steps"] + gb10["growth"], 19104)
        self.assertEqual(gb10["val_batch"], 131072)


if __name__ == "__main__":
    unittest.main()
