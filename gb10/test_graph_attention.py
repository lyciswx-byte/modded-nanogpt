"""Reference tests for the graph-safe SM121 tiled ragged attention.

CPU covers masks/gradients with changing live segment boundaries. CUDA test
captures one small forward/backward and replays after changing the boundary
tensor WITHOUT recapturing: this detects the most dangerous stale-BOS bug.
"""
import unittest
import torch

from gb10.graph_attention import graph_sdpa
from gb10.attention import segmented_sdpa


class GraphAttentionReference(unittest.TestCase):
    def _inputs(self, n=33, h=2, qdim=64, vdim=128, device="cpu"):
        torch.manual_seed(132)
        q = torch.randn(n, h, qdim, device=device, requires_grad=True)
        k = torch.randn(n, h, qdim, device=device, requires_grad=True)
        v = torch.randn(n, h, vdim, device=device, requires_grad=True)
        return q, k, v

    def test_cpu_varlen_mask_and_grad(self):
        for qd, vd in ((64, 128), (64, 64), (128, 128)):
            q, k, v = self._inputs(n=33, qdim=qd, vdim=vd)
            for ends in ((0, 4, 13, 33, 33), (0, 8, 17, 33, 33)):
                off = torch.tensor(ends, dtype=torch.int32)
                got = graph_sdpa(q, k, v, off, 8, .13)
                ref = segmented_sdpa(q, k, v, off, 8, .13)
                self.assertTrue(torch.allclose(got, ref, atol=2e-5, rtol=2e-5),
                                f"({qd},{vd}) {ends}: {(got-ref).abs().max()}")
                dg = torch.autograd.grad(got.sum(), (q, k, v), retain_graph=True)
                dr = torch.autograd.grad(ref.sum(), (q, k, v), retain_graph=True)
                for a, b in zip(dg, dr):
                    self.assertTrue(torch.allclose(a, b, atol=2e-5, rtol=2e-5),
                                    f"({qd},{vd}) {ends} backward {float((a-b).abs().max())}")

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
    def test_cuda_graph_replay_reads_live_cu_seqlens(self):
        q, k, v = self._inputs(n=64, device="cuda")
        ends = torch.tensor([0, 16, 32, 64, 64], dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                y = graph_sdpa(q, k, v, ends, 16, .13)
                torch.autograd.grad(y.square().mean(), (q, k, v))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = graph_sdpa(q, k, v, ends, 16, .13)
            grads = torch.autograd.grad(output.square().mean(), (q, k, v))
        graph.replay()
        baseline = output.detach().clone()
        new_ends = torch.tensor([0, 8, 28, 64, 64], device="cuda", dtype=torch.int32)
        ends.copy_(new_ends)
        graph.replay()
        changed = output.detach().clone()
        eager = graph_sdpa(q, k, v, ends, 16, .13)
        self.assertFalse(torch.allclose(changed, baseline, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(changed, eager, atol=2e-3, rtol=2e-3))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
