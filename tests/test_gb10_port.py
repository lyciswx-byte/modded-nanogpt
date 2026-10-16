"""GB10 port checks. Static tests run without CUDA/PyTorch; numerical tests are optional."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StaticGb10Tests(unittest.TestCase):
    def test_no_hopper_only_flash_attn_import(self):
        src = (ROOT / "train_gpt.py").read_text()
        self.assertIn("gb10_flash_attn_varlen_func", src)
        self.assertNotIn("from kernels import get_kernel", src)

    def test_default_single_gpu_launcher(self):
        script = (ROOT / "run.sh").read_text()
        self.assertIn("--nproc_per_node=1", script)
        self.assertIn("scripts/preflight_gb10.py", script)

    def test_training_schedule_preserved(self):
        src = (ROOT / "train_gpt.py").read_text()
        self.assertIn("num_scheduled_iterations: int = 1270", src)
        self.assertIn("num_extension_iterations: int = 15", src)
        self.assertIn("grad_accum_steps = 8 // world_size", src)

    def test_gb10_python_files_parse(self):
        for name in ("gb10_attention.py", "gb10_eval.py", "scripts/preflight_gb10.py",
                     "scripts/verify_gb10_attention.py"):
            ast.parse((ROOT / name).read_text(), filename=name)

    def test_requirements_do_not_replace_ngc_torch(self):
        src = (ROOT / "requirements_gb10.txt").read_text()
        pkgs = [s.strip().split("==")[0] for s in src.splitlines()
                if s.strip() and not s.lstrip().startswith("#")]
        self.assertNotIn("torch", pkgs)
        self.assertNotIn("flash-attn", pkgs)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Optional CPU math tests require torch")
class CpuGb10MathTests(unittest.TestCase):
    def test_window_block_plan(self):
        from gb10_attention import window_block_plan
        self.assertEqual(window_block_plan(192, 128), (2, 2))
        self.assertEqual(window_block_plan(16384, 384), (128, 4))
        with self.assertRaises(ValueError):
            window_block_plan(0, 128)

    def test_block_mask_document_boundaries(self):
        from gb10_attention import make_varlen_block_mask
        boundaries = torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32)
        mask = make_varlen_block_mask(boundaries, 8, 2).mask_mod
        def permit(q, k):
            return bool(mask(torch.tensor(0), torch.tensor(0),
                             torch.tensor(q), torch.tensor(k)))
        self.assertTrue(permit(3, 1))
        self.assertFalse(permit(3, 0))     # outside left window
        self.assertFalse(permit(4, 3))     # document boundary
        self.assertTrue(permit(4, 4))      # self
        self.assertFalse(permit(4, 5))     # noncausal

    def test_chunked_validation_matches_dense(self):
        import torch.nn as nn
        import torch.nn.functional as F
        from gb10_eval import chunked_eval_ce
        torch.manual_seed(7)
        head = nn.Linear(32, 257, bias=False)
        x = torch.randn(1, 29, 32)
        t = torch.randint(0, 257, (29,))
        dense = head(x)
        expected = F.cross_entropy((23 * torch.sigmoid((dense + 5) / 7.5)).float(),
                                   t, reduction="none")
        actual = chunked_eval_ce(x, t, head, chunk_rows=7)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6))


if __name__ == "__main__":
    unittest.main()
