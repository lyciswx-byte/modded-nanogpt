"""GB10-only tile correctness, replay and numerical acceptance test."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available() and
                     torch.cuda.get_device_capability() == (12, 1),
                     "requires sm_121 GB10")
class TritonTileGB10(unittest.TestCase):
    def test_tiled_gram_xxt_xtx_and_poly(self):
        from gb10 import tile_tuner
        torch.manual_seed(9)
        for m, n in ((128, 64), (64, 128), (256, 128)):
            a = (torch.randn(2, m, n, device="cuda") * .025).bfloat16()
            kind = "xtx" if m > n else "xxt"
            ref = a.mT @ a if kind == "xtx" else a @ a.mT
            target = torch.empty_like(ref)
            tile_tuner.tune(kind, a)
            if kind == "xtx":
                tile_tuner.XTX(a, target)
            else:
                tile_tuner.XXT(a, target)
            self.assertTrue(torch.isfinite(target).all())
            self.assertLess(float((target - ref).abs().max()), max(.02, .025 * float(ref.abs().max())))
            symmetric = (torch.randn_like(ref) * .1).bfloat16()
            symmetric = (.5 * (symmetric + symmetric.mT)).contiguous()
            ref_poly = 1.3 * (symmetric @ symmetric.mT) - .25 * symmetric
            out = torch.empty_like(symmetric)
            tile_tuner.tune("poly", symmetric)
            tile_tuner.ba_plus_cAA(symmetric, 1.3, -.25, out)
            self.assertLess(float((out - ref_poly).abs().max()),
                            max(.02, .025 * float(ref_poly.abs().max())))

    def test_tile_graph_replay_uses_live_input(self):
        from gb10 import tile_tuner
        inp = (torch.randn(1, 128, 64, device="cuda") * .1).bfloat16()
        tile_tuner.tune("xtx", inp)
        result = torch.empty((1, 64, 64), device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                tile_tuner.XTX(inp, result)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            tile_tuner.XTX(inp, result)
        graph.replay()
        a = result.clone()
        inp.add_(.1)
        graph.replay()
        torch.cuda.synchronize()
        expected = inp.mT @ inp
        self.assertFalse(torch.allclose(a, result))
        self.assertTrue(torch.allclose(result, expected, atol=.02, rtol=.025))


if __name__ == "__main__":
    unittest.main()
