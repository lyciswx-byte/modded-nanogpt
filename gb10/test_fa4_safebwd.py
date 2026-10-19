"""CPU-only regression for FA4-forward/SDPA-backward diagnostic bridge.

The real CuTe bundle is NEVER loaded in this test. We mock its forward
with native SDPA to validate exact gradient plumbing and fail-closed checks.
"""
import os
import unittest
from unittest.mock import patch

import torch

from gb10.attention import segmented_sdpa
from gb10.fa4_safebwd import forward_fa4_backward_sdpa


class FA4SafeBackwardCPU(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(723)
        q=torch.randn(12,2,64,requires_grad=True)
        k=torch.randn(12,2,64,requires_grad=True)
        v=torch.randn(12,2,128,requires_grad=True)
        offsets=torch.tensor([0,4,12,12],dtype=torch.int32)
        return q,k,v,offsets

    def test_matches_native_forward_and_all_three_gradients(self):
        q,k,v,offsets=self._inputs()
        def fake(q,k,v,seqlens,window,scale):
            return segmented_sdpa(q,k,v,seqlens,window,scale)
        with patch("gb10.fa4.packed_fa4",side_effect=fake):
            out=forward_fa4_backward_sdpa(q,k,v,offsets,3,.13)
        expected=segmented_sdpa(q,k,v,offsets,3,.13)
        torch.testing.assert_close(out,expected,rtol=1e-6,atol=1e-6)
        probe=torch.randn_like(out)
        grads=torch.autograd.grad(out,(q,k,v),probe,retain_graph=True)
        reference=torch.autograd.grad(expected,(q,k,v),probe)
        for actual,ref in zip(grads,reference):
            torch.testing.assert_close(actual,ref,rtol=1e-5,atol=1e-5)

    def test_wrong_fa4_forward_fails_closed(self):
        q,k,v,offsets=self._inputs()
        def broken(q,k,v,seqlens,window,scale):
            return torch.zeros_like(v)
        with patch("gb10.fa4.packed_fa4",side_effect=broken):
            with self.assertRaises(AssertionError):
                forward_fa4_backward_sdpa(q,k,v,offsets,3,.13)

    def test_cannot_be_selected_inside_graph_mode(self):
        q,k,v,offsets=self._inputs()
        with patch.dict(os.environ,{"GB10_CUDA_GRAPHS":"1"}):
            with self.assertRaisesRegex(RuntimeError,"cannot enter CUDA graphs"):
                forward_fa4_backward_sdpa(q,k,v,offsets,3,.13)


if __name__=="__main__":
    unittest.main()
