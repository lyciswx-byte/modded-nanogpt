"""CPU regressions for b09d028's FP8 and FA4 hardware failure paths.

Do not require Triton or CUDA. These tests validate pad/layout semantics and
FA4 QK/V dimension compatibility; they do NOT certify SM121 CuTe compilation.
"""
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from gb10.gemm_layout import prepare_fp8_gemm_input, column_major_rhs
from gb10.fa4 import pad_mixed_qkv_for_fa4, load_fa4


class FP8LayoutCPU(unittest.TestCase):
    def test_ragged_rows_preserve_result_and_gradients(self):
        torch.manual_seed(12)
        for rows in (1, 15, 16, 65, 127, 128, 129):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 64, dtype=torch.float32, requires_grad=True)
                w = torch.randn(96, 64, dtype=torch.float32, requires_grad=True)
                padded, original = prepare_fp8_gemm_input(x)
                rhs = column_major_rhs(w)
                self.assertEqual(padded.shape, ((rows + 15) // 16 * 16, 64))
                self.assertEqual(original, rows)
                self.assertEqual(rhs.shape, (64, 96))
                self.assertEqual(rhs.stride(), (1, 64))
                self.assertFalse(rhs.is_contiguous())
                self.assertTrue(torch.equal(padded[rows:], torch.zeros_like(padded[rows:])))
                y = (padded @ rhs)[:original]
                reference = F.linear(x, w)
                torch.testing.assert_close(y, reference, rtol=1e-6, atol=1e-5)
                go = torch.randn_like(y)
                dy = torch.autograd.grad(y, (x, w), go, retain_graph=True)
                dr = torch.autograd.grad(reference, (x, w), go)
                for a, b in zip(dy, dr):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=3e-5)

    def test_reject_invalid_layout_input(self):
        for shape in ((2, 3, 4), ()):
            with self.assertRaises(ValueError):
                prepare_fp8_gemm_input(torch.empty(shape))
        with self.assertRaises(ValueError):
            column_major_rhs(torch.ones(2, 3, 4))
        for align in (0, 3, 7):
            with self.assertRaises(ValueError):
                prepare_fp8_gemm_input(torch.zeros(9, 16), align)


class FA4MixedCPU(unittest.TestCase):
    def test_padding_exact_attention_and_backward(self):
        torch.manual_seed(102)
        length = 9
        for d_qk, d_v in ((64, 64), (64, 128), (128, 128)):
            with self.subTest(d_qk=d_qk, d_v=d_v):
                q = torch.randn(length, 3, d_qk, requires_grad=True)
                k = torch.randn(length, 3, d_qk, requires_grad=True)
                v = torch.randn(length, 3, d_v, requires_grad=True)
                q_pad, k_pad, v_pad = pad_mixed_qkv_for_fa4(q, k, v)
                self.assertEqual(q_pad.shape[-1], d_v)
                self.assertEqual(k_pad.shape[-1], d_v)
                self.assertEqual(v_pad.shape[-1], d_v)
                if d_qk < d_v:
                    self.assertTrue(torch.equal(q_pad[..., d_qk:], torch.zeros_like(q_pad[..., d_qk:])))
                idx = torch.arange(length)
                delta = idx[:, None] - idx[None, :]
                mask = (delta >= 0) & (delta <= 3)
                scale = 0.13 if d_qk == 64 else 0.085
                def attention(qi, ki, vi):
                    return F.scaled_dot_product_attention(
                        qi.transpose(0, 1)[None],
                        ki.transpose(0, 1)[None],
                        vi.transpose(0, 1)[None],
                        attn_mask=mask, scale=scale
                    )
                y = attention(q_pad, k_pad, v_pad)
                ref = attention(q, k, v)
                torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
                g = torch.randn_like(y)
                grads = torch.autograd.grad(y, (q, k, v), g, retain_graph=True)
                ref_grads = torch.autograd.grad(ref, (q, k, v), g)
                for a, b in zip(grads, ref_grads):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_reject_unrecognized_dims(self):
        for dq, dv in ((32, 128), (128, 64), (256, 256)):
            with self.assertRaises(ValueError):
                pad_mixed_qkv_for_fa4(
                    torch.empty(2, 4, dq), torch.empty(2, 4, dq),
                    torch.empty(2, 4, dv)
                )

    def test_fa4_load_errors_keep_original_cause(self):
        def broken_kernel(*args, **kwargs):
            self.assertEqual(args, ("SecondNatureComputing/flash-attn-4-sm120",))
            self.assertEqual(kwargs["version"], 1)
            self.assertTrue(kwargs["trust_remote_code"])
            raise ValueError("CuTe package import failed")
        load_fa4.cache_clear()
        with patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)), \
             patch.dict(sys.modules, {"kernels": types.SimpleNamespace(get_kernel=broken_kernel)}):
            with self.assertRaisesRegex(RuntimeError, "FA4 bundle could not be loaded") as cm:
                load_fa4()
            self.assertIsInstance(cm.exception.__cause__, ValueError)
            self.assertIn("CuTe package import failed", str(cm.exception.__cause__))
        load_fa4.cache_clear()


if __name__ == "__main__":
    unittest.main()
