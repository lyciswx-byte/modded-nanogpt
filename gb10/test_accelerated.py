"""On-GB10 regression gates for SM121 kernels.

Run on EACH GB10 before enabling kernels for two-node training:
  python -m unittest gb10.test_accelerated -v

No silent backend fallback. Tests cover qk=64/v=128, 64/64, 128/128,
paired-head varlen, gradients, candidate/MTP/prefix overlapping targets.
"""
import os
import unittest

import torch
import torch.nn.functional as F

from gb10.attention import segmented_sdpa


def has_gb10():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)


@unittest.skipUnless(has_gb10(), "requires a real GB10 sm_121")
class AcceleratedGB10(unittest.TestCase):
    def test_fa4_varlen_mixed_heads_forward_backward(self):
        from gb10.fa4 import packed_fa4
        for qd, vd in ((64, 128), (64, 64), (128, 128)):
            torch.manual_seed(qd + vd)
            # Four segments: the shape often omitted in trivial FA4 tests.
            lens = (96, 160, 64, 192)
            ends = [0]
            for n in lens:
                ends.append(ends[-1] + n)
            cum = torch.tensor(ends + [ends[-1]] * 4, dtype=torch.int32)
            total = ends[-1]
            q = torch.randn(total, 6, qd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn(total, 6, vd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            # Pick a realistic long/short-window cap and scale.
            scale = 0.085 if qd == 128 else 0.13
            y = packed_fa4(q, k, v, cum, 128, scale)
            y_ref = segmented_sdpa(q, k, v, cum, 128, scale)
            self.assertEqual(y.shape, (total, 6, vd))
            # BF16 accumulations have different reduction order.
            self.assertTrue(torch.allclose(y.float(), y_ref.float(), atol=0.085, rtol=0.06),
                            f"FA4 != SDPA for head dims ({qd},{vd}); "
                            f"max abs={(y.float()-y_ref.float()).abs().max()}")
            probe = torch.randn_like(y)
            grads = torch.autograd.grad(y, (q,k,v), probe, retain_graph=True)
            ref_grads = torch.autograd.grad(y_ref, (q,k,v), probe)
            for a,b in zip(grads,ref_grads):
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(torch.allclose(a.float(),b.float(),atol=0.12,rtol=0.12),
                                f"FA4 backward diff {float((a-b).abs().max())}")
            torch.cuda.synchronize()

    def test_fp8_projection_and_grads(self):
        from gb10.fp8_linear import linear
        torch.manual_seed(103)
        x = torch.randn(65, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(1280, 768, device="cuda") / 24).to(torch.bfloat16).requires_grad_()
        y = linear(x,w)
        ref = F.linear(x,w)
        self.assertEqual(y.shape, ref.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.allclose(y.float(),ref.float(),atol=0.1,rtol=0.1))
        probe = torch.randn_like(y)
        gy = torch.autograd.grad(y,(x,w),probe,retain_graph=True)
        gr = torch.autograd.grad(ref,(x,w),probe)
        for a,b in zip(gy,gr):
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue(torch.allclose(a.float(),b.float(),atol=0.13,rtol=0.13))

    def test_fused_fp8_ce_value_gradient(self):
        from gb10.fused_fp8_ce import fused_loss
        torch.manual_seed(111)
        T, V = 7, 10240
        logits = (torch.randn(T,V,device="cuda",dtype=torch.bfloat16) * 0.8).requires_grad_()
        # Repeated MTP/prefix labels ensure multiple subtraction terms combine.
        target = torch.tensor([12,12,13,14,14,15,15],device="cuda",dtype=torch.int64)
        prefix = torch.tensor([12,-1,13,14,12,-1,15],device="cuda",dtype=torch.int64)
        w = torch.tensor([1.,0.5,0.25],device="cuda")
        pw = torch.tensor([0.2],device="cuda")
        actual = fused_loss(logits.contiguous(),target,prefix,w,pw,0,T)
        z = 23*torch.sigmoid((logits.float()+5)/7.5)
        logp = F.log_softmax(z,dim=-1)
        ref = []
        for i in range(T):
            item = torch.zeros((),device="cuda")
            for k in range(3):
                if i+k<T: item = item - w[k]*logp[i,target[i+k]]
            if prefix[i]>=0: item = item - pw[0]*logp[i,prefix[i]]
            ref.append(item)
        ref=torch.stack(ref)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.allclose(actual,ref,atol=0.4,rtol=0.05),
                        f"loss max diff: {float((actual-ref).abs().max())}")
        dg = torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0]
        dr = torch.autograd.grad(ref.sum(),logits)[0]
        self.assertTrue(torch.isfinite(dg).all())
        self.assertTrue(torch.allclose(dg.float(),dr.float(),atol=0.12,rtol=0.15),
                        f"grad max diff: {float((dg-dr).abs().max())}")


if __name__ == "__main__":
    unittest.main()
