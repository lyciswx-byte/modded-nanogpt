"""CPU-only tests: global token count is not local max × ranks × steps."""
import unittest
from types import SimpleNamespace

from gb10.token_metrics import global_token_budget, local_tokens_for_step


class FakeSchedule:
    def lookup(self, step):
        if step < 3:
            return SimpleNamespace(batch_size=8192), step
        if step < 5:
            return SimpleNamespace(batch_size=16384), step
        return SimpleNamespace(batch_size=24576), step


class GB10TokenMetricsTests(unittest.TestCase):
    def test_eight_steps_with_stage_change(self):
        schedule = FakeSchedule()
        self.assertEqual(global_token_budget(schedule, 8),
                         3 * 8192 + 2 * 16384 + 3 * 24576)
        self.assertEqual(local_tokens_for_step(schedule, 0, 2), 4096)
        self.assertEqual(local_tokens_for_step(schedule, 5, 2), 12288)

    def test_actual_initial_eight_steps_under_default_profile(self):
        class DefaultFirstStage:
            def lookup(self, step):
                # Original #92 stage 0 is 8*2048*8 global tokens,
                # GB10_BATCH_DIVISOR defaults to 16, WORLD_SIZE=2.
                return SimpleNamespace(batch_size=8 * 2048 * 8 // 16), step
        schedule = DefaultFirstStage()
        self.assertEqual(global_token_budget(schedule, 8), 65536)
        self.assertEqual(local_tokens_for_step(schedule, 0, 2), 4096)
        self.assertNotEqual(global_token_budget(schedule, 8), 196608)

    def test_invalid_steps_and_worker_count(self):
        for steps in (-1, 1.5, "4"):
            with self.assertRaises(ValueError):
                global_token_budget(FakeSchedule(), steps)
        with self.assertRaises(ValueError):
            local_tokens_for_step(FakeSchedule(), 0, 0)
        with self.assertRaises(ValueError):
            local_tokens_for_step(FakeSchedule(), -1, 2)

    def test_empty_run(self):
        self.assertEqual(global_token_budget(FakeSchedule(), 0), 0)


if __name__ == "__main__":
    unittest.main()
