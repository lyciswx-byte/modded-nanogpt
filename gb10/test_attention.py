"""CPU correctness checks of the sm_121 fallback primitives (no GPU needed)."""
import unittest
import torch
import torch.nn.functional as F

from gb10.attention import segmented_sdpa, qk_norm_rope_reference


class TestGB10Attention(unittest.TestCase):
    def test_mixed_width_ragged_sliding_and_grad(self):
        torch.manual_seed(17)
        q = torch.randn(9, 3, 8, requires_grad=True)
        k = torch.randn(9, 3, 8, requires_grad=True)
        v = torch.randn(9, 3, 16, requires_grad=True)
        ends = torch.tensor([0, 4, 7, 9, 9, 9], dtype=torch.int32)
        y = segmented_sdpa(q, k, v, ends, left_window=2, scale=0.13)
        chunks = []
        for lo, hi in ((0, 4), (4, 7), (7, 9)):
            L = hi - lo
            p = torch.arange(L)
            m = (p[:, None] >= p[None, :]) & (p[:, None] - p[None, :] <= 2)
            ref = F.scaled_dot_product_attention(
                q[lo:hi].transpose(0, 1)[None],
                k[lo:hi].transpose(0, 1)[None],
                v[lo:hi].transpose(0, 1)[None], attn_mask=m, scale=0.13,
            )
            chunks.append(ref[0].transpose(0, 1))
        self.assertTrue(torch.allclose(y, torch.cat(chunks), atol=1e-5, rtol=1e-5))
        self.assertEqual(y.shape, (9, 3, 16))
        y.sum().backward()
        for t in (q, k, v):
            self.assertTrue(torch.isfinite(t.grad).all())

    def test_qk_norm_pair_and_mixed_dimensions(self):
        torch.manual_seed(0)
        t, heads, d = 7, 6, 64
        qk = torch.randn(t, 2 * heads, d, dtype=torch.float32, requires_grad=True)
        f1 = torch.ones(t, d * 2, dtype=torch.float32)
        f2 = torch.zeros_like(f1)
        q, k = qk_norm_rope_reference(qk, f1, f2, heads, 64, True, False)
        self.assertEqual(q.shape, (t * 2, heads // 2, d))
        self.assertEqual(k.shape, (t * 2, heads // 2, d))
        self.assertTrue(torch.allclose(q.square().mean(-1), torch.ones_like(q[..., 0]), atol=1e-5))
        (q.square().sum() + k.square().sum()).backward()
        self.assertTrue(torch.isfinite(qk.grad).all())

    def test_wide_key_offset(self):
        x = torch.randn(5, 12, 128)
        f1 = torch.ones(5, 128)
        f2 = torch.zeros(5, 128)
        q, k = qk_norm_rope_reference(x, f1, f2, 6, 64, False, True)
        self.assertEqual(q.shape, (5, 6, 128))
        self.assertTrue(torch.allclose(k[1, :, 64:], k[0, :, 64:], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
