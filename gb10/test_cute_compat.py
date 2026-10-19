"""CPU-only tests for the CuTe DSL namespace migration in GB10 FA4.

The installed CUTLASS >=4.6 exports ThrMma at cutlass.cute, while older
third-party FA4 code eagerly evaluates cute.core.ThrMma annotations.
These tests model only API aliasing, not actual GPU compilation.
"""
import os
import sys
import tempfile
import pathlib
import types
import unittest
from unittest.mock import patch

import torch

from gb10.cute_compat import install_legacy_cute_annotations


def fake_cutlass(*, with_top_level=True, with_core=False):
    cutlass = types.ModuleType("cutlass")
    cutlass.__path__ = []
    cute = types.ModuleType("cutlass.cute")
    cute.__path__ = []
    core = types.ModuleType("cutlass.cute.core")
    classes = {
        "ThrMma": type("ThrMma", (), {}),
        "TiledMma": type("TiledMma", (), {}),
        "ThrCopy": type("ThrCopy", (), {}),
        "TiledCopy": type("TiledCopy", (), {}),
    }
    if with_top_level:
        for key, cls in classes.items():
            setattr(cute, key, cls)
    if with_core:
        setattr(core, "ThrMma", classes["ThrMma"])
    cutlass.cute = cute
    cute.core = core
    return {
        "cutlass": cutlass,
        "cutlass.cute": cute,
        "cutlass.cute.core": core,
    }, classes


class CutlassCompatibilityTests(unittest.TestCase):
    def test_legacy_annotations_alias_real_newer_types(self):
        modules, classes = fake_cutlass()
        with patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {"GB10_FA4_CUTE_COMPAT": "1"}):
            changed = install_legacy_cute_annotations()
            self.assertEqual(set(changed), set(classes))
            for key, cls in classes.items():
                self.assertIs(getattr(modules["cutlass.cute.core"], key), cls)
            self.assertEqual(install_legacy_cute_annotations(), ())

    def test_disabled_mode_does_not_mutate_namespace(self):
        modules, _ = fake_cutlass()
        with patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {"GB10_FA4_CUTE_COMPAT": "0"}):
            self.assertEqual(install_legacy_cute_annotations(), ())
            self.assertFalse(hasattr(modules["cutlass.cute.core"], "ThrMma"))

    def test_preexisting_legacy_symbol_is_preserved(self):
        modules, classes = fake_cutlass(with_core=True)
        previous = modules["cutlass.cute.core"].ThrMma
        with patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {"GB10_FA4_CUTE_COMPAT": "1"}):
            changed = install_legacy_cute_annotations()
            self.assertNotIn("ThrMma", changed)
            self.assertIs(modules["cutlass.cute.core"].ThrMma, previous)

    def test_missing_real_class_fails_instead_of_stubbing(self):
        modules, _ = fake_cutlass(with_top_level=False)
        with patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {"GB10_FA4_CUTE_COMPAT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "cannot be mapped"):
                install_legacy_cute_annotations()

    def test_actual_local_bundle_loader_imports_legacy_annotation(self):
        """Integration-like test with a tiny fake FA4 package, no GPU needed."""
        from gb10.fa4 import load_fa4
        modules, classes = fake_cutlass()
        with tempfile.TemporaryDirectory() as temp:
            pkg = pathlib.Path(temp) / "build" / "torch-cuda"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text(
                "from cutlass.cute.core import ThrMma\n"
                "def flash_attn_varlen_func(*args, **kwargs):\n"
                "    return ThrMma\n",
                encoding="utf-8",
            )
            load_fa4.cache_clear()
            with patch.dict(sys.modules, modules), \
                 patch.dict(os.environ, {
                     "GB10_FA4_LOCAL_PATH": temp,
                     "GB10_FA4_CUTE_COMPAT": "1",
                 }), \
                 patch.object(torch.cuda, "get_device_capability", return_value=(12, 1)):
                fn = load_fa4()
                self.assertIs(fn(), classes["ThrMma"])
            load_fa4.cache_clear()
            sys.modules.pop("flash_attn_4_sm120", None)


if __name__ == "__main__":
    unittest.main()
