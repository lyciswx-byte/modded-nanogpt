"""CPU-only regression checks for tile races and bounded two-rank validation."""
import unittest

from gb10.policy import require_symmetric_tile, validation_windows


class GB10PolicyTests(unittest.TestCase):
    def test_symmetric_mirror_tile_accepts_only_safe_shapes(self):
        for edge, k in ((32,32),(64,32),(64,64),(128,64)):
            require_symmetric_tile(edge,edge,k)
        for m,n,k in ((32,64,32),(64,128,32),(128,64,32),
                      (64,64,0),(48,48,32),(64,64,48)):
            with self.subTest(m=m,n=n,k=k):
                with self.assertRaises(ValueError):
                    require_symmetric_tile(m,n,k)

    def test_streaming_validation_windows_exactly_once(self):
        for count in (0,1,2,3,7,8,9,37,160,161):
            for width in (1,2,3,4,8):
                with self.subTest(count=count,width=width):
                    groups = list(validation_windows(count,width))
                    got = [i for group in groups for i in group]
                    self.assertEqual(got,list(range(count)))
                    self.assertTrue(all(1 <= len(g) <= width for g in groups))
                    self.assertEqual(groups,list(validation_windows(count,width)))

    def test_validation_rejects_unsafe_input(self):
        for count, window in ((-1,2),(0,0),(5,9),(10,-1)):
            with self.assertRaises(ValueError):
                list(validation_windows(count,window))


if __name__ == "__main__":
    unittest.main()
