"""CPU-level algorithmic tests for GB10's native CE path.

Run without GB10 hardware: python -m unittest gb10.test_loss -v
These check full-vocab, sampled targets and prefix/MTP indexing/gradient.
"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gb10.loss import native_training_loss


class NativeLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.t = 6
        self.dim = 8
        self.vocab = 16
        self.x = torch.randn(self.t, self.dim, dtype=torch.float32, requires_grad=True)
        self.w = torch.randn(self.dim, self.vocab, dtype=torch.float32, requires_grad=True)
        self.targets = torch.tensor([2, 1, 3, 1, 7, 4], dtype=torch.int64)
        self.prefix = torch.tensor([1, -1, 1, 2, 3, -1], dtype=torch.int64)
        self.mtp = torch.tensor([1.0, 0.5, 0.25])
        self.prefix_w = torch.tensor([0.2])

    def reference(self, logits, target_pos, prefix_pos):
        log_p = F.log_softmax(23 * torch.sigmoid((logits + 5) / 7.5), dim=-1)
        res = []
        for i in range(self.t):
            score = log_p.new_zeros(())
            for k, weight in enumerate(self.mtp):
                if i + k < self.t:
                    score = score - weight * log_p[i, target_pos[i + k]]
            if prefix_pos[i] >= 0:
                score = score - self.prefix_w[0] * log_p[i, prefix_pos[i]]
            res.append(score)
        return torch.stack(res)

    def test_full_vocab_and_grad(self):
        actual = native_training_loss(self.x, self.targets, self.mtp, self.prefix,
                                      self.prefix_w, self.w, sampled_loss=None)
        expect = self.reference(self.x @ self.w, self.targets, self.prefix)
        self.assertTrue(torch.allclose(actual, expect, atol=2e-5, rtol=2e-5))
        ga = torch.autograd.grad(actual.sum(), (self.x, self.w), retain_graph=True)
        ge = torch.autograd.grad(expect.sum(), (self.x, self.w))
        for a, b in zip(ga, ge):
            self.assertTrue(torch.allclose(a, b, atol=2e-5, rtol=2e-5))

    def test_sampled_shared_candidates(self):
        candidate_ids = torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64)
        idx = torch.full((self.vocab,), -1, dtype=torch.int64)
        idx[candidate_ids] = torch.arange(len(candidate_ids))
        sample = SimpleNamespace(
            candidate_ids=candidate_ids, target_pos=idx[self.targets],
            prefix_pos=idx[self.prefix.clamp_min(0)].where(self.prefix >= 0, -1),
        )
        actual = native_training_loss(self.x, self.targets, self.mtp, self.prefix,
                                      self.prefix_w, self.w, sample)
        expect = self.reference(self.x @ self.w[:, candidate_ids],
                                sample.target_pos, sample.prefix_pos)
        self.assertTrue(torch.allclose(actual, expect, atol=2e-5, rtol=2e-5))
        self.assertEqual(actual.shape, (self.t,))


if __name__ == "__main__":
    unittest.main()
