"""Preflight for two GB10s with coherent CPU/GPU memory and sm_121."""
import os
import platform
import socket
import warnings


def check_runtime():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GB10 profile requires CUDA in the NVIDIA container")
    cap = torch.cuda.get_device_capability(0)
    if cap != (12, 1):
        raise RuntimeError(f"Expected NVIDIA GB10 compute capability sm_121, found {cap}")
    if platform.machine() not in ("aarch64", "arm64"):
        warnings.warn(f"GB10 normally uses ARM64, host reports {platform.machine()}")
    world = int(os.environ.get("WORLD_SIZE", "0"))
    local = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if (world, local) != (2, 1):
        raise RuntimeError(f"Expected torchrun --nnodes=2 --nproc-per-node=1, got WORLD_SIZE={world}, LOCAL_WORLD_SIZE={local}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Installed PyTorch build cannot use BF16 on GB10")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("Installed PyTorch has no NCCL support")
    print(f"[GB10 preflight] host={socket.gethostname()}, torch={torch.__version__}, "
          f"torch_cuda={torch.version.cuda}, CC={cap}, "
          f"NCCL_IF={os.environ.get('NCCL_SOCKET_IFNAME', '(automatic)')}", flush=True)


def _mem_available_bytes():
    # /proc/meminfo reflects unified physical memory more usefully than only
    # torch.cuda.mem_get_info for the GB10 CPU/GPU coherent allocator.
    system_available = None
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                system_available = int(line.split()[1]) * 1024
                break
    if system_available is None:
        raise RuntimeError("No MemAvailable value in /proc/meminfo")
    # Docker's memory.max can be lower than system RAM; use the stricter
    # limit without mistaking the 128GB shared CPU/GPU pool for discrete VRAM.
    limit_path = "/sys/fs/cgroup/memory.max"
    used_path = "/sys/fs/cgroup/memory.current"
    try:
        with open(limit_path, encoding="ascii") as f:
            raw = f.read().strip()
        if raw != "max":
            with open(used_path, encoding="ascii") as f:
                used = int(f.read().strip())
            system_available = min(system_available, max(0, int(raw) - used))
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return system_available


def memory_plan(device, max_step_tokens: int, val_step_tokens: int):
    import torch
    from track_1_short.ngram_table import NGRAM_VOCAB_SIZE, NGRAM_DIM
    world = 2
    rows_per_rank = NGRAM_VOCAB_SIZE // world
    # persistent shard, FP32 per-row second moment, int32 lazy replay time,
    # plus the fixed row cache. Other optimizer, model, temporary, PyTorch
    # compilation and OS buffers are covered by configurable conservative reserve.
    shard = rows_per_rank * NGRAM_DIM * 2
    row_states = rows_per_rank * 8
    cache = 2 * 4 * max_step_tokens * NGRAM_DIM * 2
    val_masks = 50304 * (50304 // 8)
    persistent = shard + row_states + cache + val_masks
    reserve_gib = float(os.environ.get("GB10_SYSTEM_RESERVE_GIB", "24"))
    available = _mem_available_bytes()
    required = persistent + int(reserve_gib * (2 ** 30))
    print(f"[GB10 memory] rows/rank={rows_per_rank:,} ngram_bf16={shard/2**30:.2f}GiB "
          f"row_states={row_states/2**30:.2f}GiB cache={cache/2**30:.2f}GiB "
          f"system_available={available/2**30:.2f}GiB "
          f"reserve={reserve_gib:.1f}GiB "
          f"val_tokens/rank={val_step_tokens}", flush=True)
    if available < required:
        raise MemoryError(f"GB10 unified-memory preflight failed: need at least "
                          f"{required/2**30:.1f}GiB available, have {available/2**30:.1f}GiB. "
                          "Reduce GB10_NGRAM_ROWS, close other processes or adjust reserve "
                          "ONLY after observing real peak usage.")
    return {"ngram_bytes": shard, "row_state_bytes": row_states, "available_bytes": available}
