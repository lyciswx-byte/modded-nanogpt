"""Process-group setup for 8xH100 and 2xGB10 multi-node training."""
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

WORLD_SIZE = 2 if os.environ.get("GB10_PROFILE") == "1" else 8


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
    # The sampled softmax and the sparse row exchanges assume one microbatch per step on 8 ranks.
    assert world_size == WORLD_SIZE, f"track 1 runs on exactly {WORLD_SIZE} GPUs, got world_size={world_size}"
    assert torch.cuda.is_available()
    if os.environ.get("GB10_PROFILE") == "1":
        assert int(os.environ.get("LOCAL_WORLD_SIZE", "1")) == 1, "One GPU process per GB10 node"
        cap = torch.cuda.get_device_capability(int(os.environ["LOCAL_RANK"]))
        assert cap == (12, 1), f"Expected GB10 sm_121, detected sm_{cap[0]}{cap[1]}"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    dist.barrier()
    return DistEnv(rank=rank, world_size=world_size, device=device)
