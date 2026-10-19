"""Pure CPU checks for GB10 socket vs RDMA container visibility."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gb10.fabric import probe_fabric, report_fabric


class FabricTests(unittest.TestCase):
    def test_socket_only_without_rdma_devices(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "sys"; s.mkdir()
            dev = Path(d) / "dev"; dev.mkdir()
            p = probe_fabric(s, dev, "enp1s0f1np1")
            self.assertFalse(p.container_has_rdma_devices)
            self.assertEqual(p.network_interface, "enp1s0f1np1")

    def test_exposed_sysfs_and_device_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "sys"; s.mkdir()
            (s / "mlx5_0").mkdir()
            dev = Path(d) / "dev"; dev.mkdir()
            (dev / "uverbs0").touch()
            (dev / "rdma_cm").touch()
            p = probe_fabric(s, dev)
            self.assertTrue(p.container_has_rdma_devices)
            self.assertEqual(p.rdma_sysfs_devices, ("mlx5_0",))
            self.assertIn("uverbs0", p.ib_device_nodes)

    def test_enforced_rdma_preflight_is_fail_closed(self):
        with patch("gb10.fabric.probe_fabric",
                   return_value=probe_fabric("/nonexistent/ib", "/nonexistent/dev")), \
             patch.dict("os.environ", {"GB10_REQUIRE_RDMA_DEVICE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "cannot see both"):
                report_fabric()


if __name__ == "__main__":
    unittest.main()
