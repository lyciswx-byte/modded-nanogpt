"""CPU-only FP8 numerical-budget regression.

Proves the old blanket elementwise tolerance fails even for a mathematically
correct E4M3 GEMM; verifies bounded approximation and catches gross errors.
"""
import unittest

import torch
import torch.nn.functional as F

from gb10.precision import assert_fp8_budget, fp8_error_metrics


class FP8PrecisionCPU(unittest.TestCase):
    def test_e4m3_approximation_budget(self):
        torch.manual_seed(103)
        x = torch.randn(65, 768, dtype=torch.bfloat16)
        w = (torch.randn(1280, 768) / 24).to(torch.bfloat16)
        sx = (x.float().abs().max() * 1.05 / 448).clamp_min(1e-8)
        sw = (w.float().abs().max() * 1.05 / 448).clamp_min(1e-8)
        qx = (x.float() / sx).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sx
        qw = (w.float() / sw).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sw
        approximate = (qx @ qw.T).to(torch.bfloat16)
        reference = F.linear(x, w)
        metrics = assert_fp8_budget(approximate, reference)
        self.assertGreater(metrics.normalized_rmse, 0)
        self.assertLess(metrics.normalized_rmse, .06)
        # Elementwise allclose is expected to fail for some outliers,
        # which is exactly why GPU tests compare the quantized GEMM too.
        self.assertFalse(torch.allclose(approximate, reference, atol=.1, rtol=.1))

    def test_reject_gross_precision_loss_or_nonfinite(self):
        base = torch.linspace(-1, 1, 100)
        with self.assertRaises(AssertionError):
            assert_fp8_budget(base + torch.ones_like(base) * .8, base)
        with self.assertRaises(ValueError):
            fp8_error_metrics(torch.tensor([float('nan')]), torch.ones(1))
        with self.assertRaises(ValueError):
            fp8_error_metrics(torch.ones(2), torch.ones(3))


if __name__ == "__main__":
    unittest.main()
