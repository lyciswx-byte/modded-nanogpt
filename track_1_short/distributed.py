"""Process-group setup for record #92 on one or eight H100 GPUs."""
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

SUPPORTED_WORLD_SIZES = (1, 8)


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    device: torch.device

    @property
    def master_process(self) -> bool:
        # rank 0 does logging, checkpointing etc.
        return self.rank == 0


def setup_distributed() -> DistEnv:
    """Initialize NCCL from torchrun's RANK / WORLD_SIZE / LOCAL_RANK."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Single-rank row exchanges are short-circuited by sharded_rows.py. The eight-rank
    # path remains available to compare with the original record.
    assert world_size in SUPPORTED_WORLD_SIZES, f"expected 1 or 8 GPUs, got world_size={world_size}"
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    dist.barrier()
    return DistEnv(rank=rank, world_size=world_size, device=device)
