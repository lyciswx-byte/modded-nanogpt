"""CPU-only regression checks preventing eager Hopper kernel compilation on GB10."""
import ast
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class ImportIsolationTests(unittest.TestCase):
    def test_sm90_nvrtc_compiler_is_conditional_on_h100_profile(self):
        source = (REPO / "track_1_short/perf/kernels/cross_entropy.py").read_text()
        tree = ast.parse(source)
        kernel_init = []
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "ce_fwd_bwd_kernel" for t in targets):
                    kernel_init.append(node.value)
        self.assertEqual(len(kernel_init), 1)
        expr = kernel_init[0]
        self.assertIsInstance(expr, ast.IfExp, "H100 NVRTC must NOT compile unconditionally")
        test = ast.unparse(expr.test)
        self.assertIn("GB10_PROFILE", test)
        self.assertIsInstance(expr.body, ast.Constant)
        self.assertIsNone(expr.body.value)
        self.assertIn("compile_ce_kernel", ast.unparse(expr.orelse))

    def test_h100_fa3_loader_is_conditional(self):
        source = (REPO / "track_1_short/model/attention.py").read_text()
        tree = ast.parse(source)
        assignments = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "flash_attn_interface" for t in node.targets
            ):
                assignments.append(node.value)
        self.assertEqual(len(assignments), 1)
        self.assertIsInstance(assignments[0], ast.IfExp)
        self.assertIn("GB10_NATIVE", ast.unparse(assignments[0].test))
        self.assertIsNone(assignments[0].body.value)
        self.assertIn("load_flash_attn3", ast.unparse(assignments[0].orelse))


if __name__ == "__main__":
    unittest.main()
