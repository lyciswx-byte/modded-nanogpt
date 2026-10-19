"""CPU contract tests for per-stage FA4 numerical fault isolator."""
import unittest

import torch

from gb10.attention import segmented_sdpa
from gb10.diagnose_fa4_backward import _reference, _stats


class DiagnosticCPU(unittest.TestCase):
    def test_reference_matches_segmented_sdpa(self):
        torch.manual_seed(81)
        q=torch.randn(23,2,64,requires_grad=True)
        k=torch.randn(23,2,64,requires_grad=True)
        v=torch.randn(23,2,128,requires_grad=True)
        ends=[0,7,14,23]
        from_diag=_reference(q,k,v,ends,causal=True,window=5,scale=.13)
        from_native=segmented_sdpa(q,k,v,torch.tensor(ends),5,.13)
        torch.testing.assert_close(from_diag,from_native,rtol=1e-6,atol=1e-6)
        do=torch.randn_like(from_diag)
        a=torch.autograd.grad(from_diag,(q,k,v),do,retain_graph=True)
        b=torch.autograd.grad(from_native,(q,k,v),do)
        for i,j in zip(a,b):
            torch.testing.assert_close(i,j,rtol=1e-5,atol=1e-5)

    def test_full_causal_matches_reference_torch(self):
        torch.manual_seed(19)
        q=torch.randn(11,2,64)
        k=torch.randn(11,2,64)
        v=torch.randn(11,2,64)
        result=_reference(q,k,v,[0,11],causal=True,window=None,scale=.13)
        t=torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0,1)[None],
            k.transpose(0,1)[None],
            v.transpose(0,1)[None],
            is_causal=True,scale=.13,
        )[0].transpose(0,1)
        torch.testing.assert_close(result,t)

    def test_metric_identifies_large_gradient_corruption(self):
        x=torch.randn(21,2,64)
        zero=_stats(x,x)
        self.assertTrue(zero["finite"])
        self.assertEqual(zero["max_abs"],0)
        bad=x.clone()
        bad.flatten()[123]=1544
        stats=_stats(bad,x)
        self.assertGreater(stats["max_abs"],1000)
        self.assertGreater(stats["rms_relative_to_ref"],1)
        nonfinite=_stats(torch.full_like(x,float("nan")),x)
        self.assertFalse(nonfinite["finite"])


if __name__=="__main__":
    unittest.main()
