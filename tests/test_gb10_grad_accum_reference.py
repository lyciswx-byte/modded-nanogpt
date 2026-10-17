"""Reference checks only: these do not activate accumulation in train_gpt.py."""
import unittest

from tools.gb10_grad_accum_reference import (
    ORIGINAL_OPTIMIZER_STEPS, ORIGINAL_RANKS,
    accumulation_plan, reference_check,
)


class TestGB10EightRankAccumulationReference(unittest.TestCase):
    def test_every_stage_matches_original_global_batch(self):
        expected = (131072, 262144, 393216, 327680, 131072)
        for divisor in (1, 2, 4, 8):
            with self.subTest(divisor=divisor):
                plans = accumulation_plan(divisor)
                self.assertEqual(tuple(p.global_tokens_per_update for p in plans), expected)
                self.assertEqual(tuple(p.microbatches_per_update for p in plans),
                                 (ORIGINAL_RANKS * divisor,) * len(expected))
                for plan in plans:
                    self.assertEqual(plan.microbatch_tokens * plan.microbatches_per_update,
                                     plan.global_tokens_per_update)
                    self.assertEqual(plan.gradient_factor, 1 / ORIGINAL_RANKS)

    def test_optimizer_update_count_is_original(self):
        self.assertEqual(ORIGINAL_OPTIMIZER_STEPS, 1194)

    def test_sum_reduction_gradient_scale(self):
        reference_check()

    def test_invalid_microbatch_divisor(self):
        for value in (0, 3, 16):
            with self.subTest(value=value), self.assertRaises(ValueError):
                accumulation_plan(value)


if __name__ == "__main__":
    unittest.main()
