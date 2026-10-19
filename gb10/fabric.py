"""Read-only network-fabric checks for two GB10 Docker containers.

These checks distinguish 'NCCL can communicate via IP Socket' from 'NCCL
actually selected a verified RDMA/RoCE transport'. They cannot prove the
NCCL backend used: only NCCL_DEBUG=INFO runtime logs and nccl-tests can.
"""
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class FabricProbe:
    rdma_sysfs_devices: tuple[str, ...]
    ib_device_nodes: tuple[str, ...]
    network_interface: str | None

    @property
    def container_has_rdma_devices(self):
        return bool(self.rdma_sysfs_devices and self.ib_device_nodes)


def probe_fabric(sys_class="/sys/class/infiniband", dev_path="/dev/infiniband",
                 interface=None):
    sys_path, device_dir = Path(sys_class), Path(dev_path)
    drivers = tuple(sorted(item.name for item in sys_path.iterdir())) if sys_path.is_dir() else ()
    nodes = tuple(sorted(item.name for item in device_dir.iterdir())) if device_dir.is_dir() else ()
    return FabricProbe(drivers, nodes, interface)


def report_fabric():
    fabric = probe_fabric(interface=os.environ.get("NCCL_SOCKET_IFNAME"))
    print(f"[GB10 fabric] socket_if={fabric.network_interface or 'auto'} "
          f"sysfs_rdma={list(fabric.rdma_sysfs_devices)} "
          f"dev_infiniband={list(fabric.ib_device_nodes)}", flush=True)
    if not fabric.container_has_rdma_devices:
        print("[GB10 fabric] No exposed RDMA device pair in container. "
              "NCCL may use NET/Socket even over the fast ConnectX interface; "
              "check Docker device passthrough, host RDMA driver and NCCL logs.",
              flush=True)
    if os.environ.get("GB10_REQUIRE_RDMA_DEVICE", "0") == "1" and not fabric.container_has_rdma_devices:
        raise RuntimeError(
            "GB10_REQUIRE_RDMA_DEVICE=1 but this container cannot see both "
            "/sys/class/infiniband and /dev/infiniband. Cannot assert RoCE "
            "availability. This check does NOT prove NCCL selected RDMA."
        )
    return fabric
