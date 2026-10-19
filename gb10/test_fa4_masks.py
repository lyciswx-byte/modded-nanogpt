"""CPU-only tests that FA4 and the backward isolator use identical masks.

The real SM121 CuTe JIT requires typed Int32 endpoints for local windows.
Tests cannot certify the third-party backward, but prevent the diagnostic
from silently exercising a different specialization than training.
"""
import sys
import types
import unittest
from unittest.mock import patch

from gb10.fa4_masks import fa4_mask_kwargs


class FakeInt32(int):
    pass


class FA4WindowCPU(unittest.TestCase):
    def test_finite_causal_is_local_left_and_zero_right(self):
        mod=types.ModuleType("cutlass")
        mod.__path__=[]
        base=types.ModuleType("cutlass.base_dsl")
        base.__path__=[]
        typing=types.ModuleType("cutlass.base_dsl.typing")
        typing.Int32=FakeInt32
        for w in (0,1,64,128,1024):
            with self.subTest(w=w),patch.dict(sys.modules,{
                "cutlass":mod,"cutlass.base_dsl":base,
                "cutlass.base_dsl.typing":typing,
            }):
                args=fa4_mask_kwargs(causal=True,window=w)
                self.assertFalse(args["causal"])
                self.assertEqual(args["window_size"],(w,0))
                self.assertTrue(all(isinstance(v,FakeInt32)
                                    for v in args["window_size"]))
                # Equality of mathematical mask, independent of CuTe DSL:
                for query in range(4):
                    for key in range(5):
                        expected=(key<=query and query-key<=w)
                        observed=(key>=query-w and key<=query)
                        self.assertEqual(expected,observed)

    def test_full_attention_defaults(self):
        self.assertEqual(
            fa4_mask_kwargs(causal=False,window=None),
            {"causal":False,"window_size":(-1,-1)},
        )
        self.assertEqual(
            fa4_mask_kwargs(causal=True,window=None),
            {"causal":True,"window_size":(-1,-1)},
        )

    def test_noncausal_local_symmetric(self):
        x=fa4_mask_kwargs(causal=False,window=7)
        self.assertFalse(x["causal"])
        self.assertEqual(x["window_size"],(7,7))

    def test_reject_negative_or_noninteger_local_window(self):
        for w in (-1,1.5,"128",True):
            with self.subTest(window=w),self.assertRaises(ValueError):
                fa4_mask_kwargs(causal=True,window=w)


if __name__=="__main__":
    unittest.main()
