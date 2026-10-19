"""Hardware-independent checks for the reshaped bank MLP and two-rank staging.

Run with: python -m unittest gb10.test_model_math -v
"""
import unittest

import torch
import torch.nn.functional as F


class TestModelMath(unittest.TestCase):
    def test_mlp_bank_transposed_down_and_grad(self):
        torch.manual_seed(123)
        x = torch.randn(7, 16, requires_grad=True)
        w1 = torch.randn(32, 16, requires_grad=True)
        w2 = torch.randn(32, 16, requires_grad=True)  # bank stores W2 as [hidden,model]
        got = F.linear(F.relu(F.linear(x, w1)).square(), w2.T)
        ref = F.relu(x @ w1.T).square() @ w2
        self.assertTrue(torch.allclose(got, ref, rtol=1e-5, atol=1e-6))
        gg = torch.autograd.grad(got.sum(), (x, w1, w2), retain_graph=True)
        gr = torch.autograd.grad(ref.sum(), (x, w1, w2))
        for a, b in zip(gg, gr):
            self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
