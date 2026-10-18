# 2×GB10 / sm_121 port of record #92 (ANVIL2) — experimental

This branch is a **two-node bring-up implementation, not a certified port or a speed record**.
Its goal is a correct gradient path, measured memory use and communication on TWO GB10
(Grace Blackwell / CUDA compute capability **12.1**) machines, *before* any Kernel-specific tuning.
It is based on the readable refactored track-1 code, which differs in some details from the
historical PR #360. Original 8×H100 remains available in the master branch.

## Why NOT just run the H100 code

| Resource | 8×H100 original | 2×GB10 target |
|---|---|---|
| CUDA architecture | Hopper sm_90 | Blackwell **sm_121** |
| GPU memory | separate 80GB HBM per GPU | **shared** CPU/GPU LPDDR5X, typically 128GB per system |
| Bandwidth | multi-TB/s HBM | about **273GB/s per GB10** |
| Device topology | 8 GPU, high-bandwidth intranode | 1 GPU per host, two hosts via ConnectX-7 200Gbps-capable fabric |
| Attention | patched, sha-verified H100 FA3 | **native per-segment SDPA**, no sm_90 binary |
| MLP / CE | H100-tuned Triton/TMA and e4m3 NVRTC | **native BF16 PyTorch**, token-slab CE |
| ANVIL Gram | H100-autotuned symmetric Triton | **ATen GEMM** until sm_121 tuning |
| CUDA Graph | full capture | intentionally **off**, eager gradient runner until memory/pointer checks pass |
| Hashed n-gram | 84.6M rows split eight ways | quarter-size by default; full original size opt-in only |

The GB10 SDPA dispatches through the installed PyTorch/cuDNN; this is not yet a benchmark
claim that SDPA beats all available sm_121 FlashAttention/FlexAttention implementations.
It avoids the common fatal error of loading the pinned sm_90 FA3 library on sm_121.
The reference implementation iterates document boundaries on **CPU pinned** metadata,
NOT by copying the packed `cu_seqlens` tensor from GPU seven times per step.

The native loss preserves the MTP, prefix CE, sampled candidate mapping and logit softcap;
it does **not** promise bitwise agreement with the original FP8 CE. Most row-sparse
and ANVIL paths are still Triton and need testing against the GB10 CUDA toolchain.

## System assumptions and the Docker image you already have

Do **not** run the repo root Dockerfile or `pip install -r requirements.txt`: these
pin a H100-oriented cu128 PyTorch image/stack. Reuse your **ARM64**, CUDA-13-based,
GPU-enabled NVIDIA PyTorch container that has PyTorch, NCCL and Triton supporting sm_121.

Run inside EACH Docker container at the same absolute checkout path:

```bash
git fetch origin
git checkout 2xGB10
python -m pip install -r gb10/requirements-gb10.txt
python -m unittest gb10.test_attention -v

nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())'
ip -br addr
ibv_devinfo     # if RoCE/RDMA devices are exposed
```

Suggested Docker flags (adjust image/path and device mappings to your actual system):

```bash
docker run --rm -it --gpus all --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$PWD":/workspace/modded-nanogpt \
  -v /your/FineWeb/data:/workspace/modded-nanogpt/data:ro \
  -w /workspace/modded-nanogpt YOUR_EXISTING_GB10_PYTORCH_IMAGE bash
```

For RDMA, expose the host InfiniBand/RoCE devices via the **appropriate**
`--device`/mount settings if your Docker runtime has them. Do not blindly
hardcode a `mlx5_0` HCA or `eth0` interface; interfaces differ with each GB10 OEM,
firmware and network configuration.

### Two-host rendezvous

The ConnectX-7 port/cable and network must already be configured on both GB10s.
Use real IP addresses from `ip -br addr`, preferably on the high-bandwidth fabric.
Set these **inside both containers**, with `MASTER_ADDR` reachable on that fabric:

GB10 #0:

```bash
export MASTER_ADDR="<RANK0_CONNECTX_IP>"
export MASTER_PORT=29500
export NODE_RANK=0
export NCCL_SOCKET_IFNAME="<ACTUAL_NETWORK_INTERFACE>"
export GB10_PROFILE=1
export GB10_SMOKE_STEPS=8
bash gb10/run_two_nodes.sh
```

GB10 #1 (same repo and data available):

```bash
export MASTER_ADDR="<RANK0_CONNECTX_IP>"
export MASTER_PORT=29500
export NODE_RANK=1
export NCCL_SOCKET_IFNAME="<ACTUAL_NETWORK_INTERFACE>"
export GB10_PROFILE=1
export GB10_SMOKE_STEPS=8
bash gb10/run_two_nodes.sh
```

The two commands run **concurrently**. This uses `torchrun --nnodes=2
--nproc-per-node=1`, without the inappropriate `--standalone` setting.

Start by testing `nccl-tests` (all_reduce_perf, all_gather_perf, alltoall)
between the two machines, preferably with `NCCL_DEBUG=INFO` to confirm the
transport. The script intentionally does not auto-select or fake RDMA routing.

### Controlled runs

```bash
# First: memory-guarded eight-step launch on both hosts, no validation/ship.
GB10_SMOKE_STEPS=8 bash gb10/run_two_nodes.sh

# After two-host tests, training through all five stages:
unset GB10_SMOKE_STEPS
bash gb10/run_two_nodes.sh
```

Configuration knobs (set **identically on both nodes**):

| Environment variable | Default | Meaning |
|---|---|---|
| GB10_PROFILE | 1 | Explicit GB10 path, required by launcher |
| GB10_BATCH_DIVISOR | 16 | Divide #92's global token batches; reduce LPDDR traffic |
| GB10_NGRAM_ROWS | 21150720 | 1/4 of #92 hash table; 15.1GiB BF16 per node |
| GB10_VAL_BATCH_TOKENS | 65536 | Global validation tokens per batch, 32768 per GB10 |
| GB10_VAL_TOKENS | 10485760 | Full original validation token count |
| GB10_CE_SLAB_ROWS | 128 | CE GEMM token block size (memory/throughput knob) |
| GB10_SYSTEM_RESERVE_GIB | 24 | Available RAM to reserve for OS/compiler/other buffers |
| GB10_SMOKE_STEPS | unset | Stop early and SKIP score certification |
| GB10_LOG_EVERY | 25 | Log progress every N iterations |
| NUM_SCHEDULED_ITERATIONS | 1122 | Main training steps before +52 growth/+20 extension |
| TRAIN_SEED | unset | Optional reproducibility |

Full n-gram table (advanced, AFTER measuring memory):

```bash
export GB10_NGRAM_ROWS=84602880
```

That table alone is approximately **60.5 GiB of BF16 weights per GB10**,
plus row state, staging, mask, gradients, optimizer states, RAM reserved for the OS,
and compile/CUDA allocation overhead. The 128GB is **unified memory**, not a
separate 128GB of GPU RAM plus 128GB system RAM. Two machines do NOT make one
shared 256GB GPU address space. Never override the preflight budget without a
real peak-memory measurement.

## Algorithmic fidelity vs record #92

**Preserved in the portable path:** irregular 11-slot GPT topology, seven
attention layers, mixed (QK,V) widths, paired heads, value embeddings, n-gram
bigram+trigram features, MUDD, ReLU², QK norm, YaRN, MTP, prefix CE, sampled
class schedule, five-phase token/sequence/window schedule, ANVIL twin rails,
per-row sparse Adam, cadence switching, and tail shipping in a full run.

**Intentionally changed:** number of workers (8 -> 2), global batch size
(default /16), n-gram table size (default /4), FP8 -> BF16 to establish parity,
CUDA graph -> eager execution, H100 FA3 -> segmented native SDPA, H100
symmetric kernels -> ATen GEMM. This alters **total training tokens and may
alter validation loss**. It does not reproduce 39.9 seconds or 3.2773 CE
automatically. In particular, the quarter-size table failed to reliably hit
3.28 in the original ablation and should not be used for a certification claim.

The full five-stage schedule has 1194 steps when NUM_SCHEDULED_ITERATIONS=1122.
With GB10_BATCH_DIVISOR=16, its token count is about 20.54M, **not** 328.66M.
Blindly multiplying step count by 16 is NOT an exact cure: optimizer rail
engagement, embedding cadence switches, tail-window length and LR cooldown
must then be retuned. Keep the smaller defaults until step-time, loss and
network time are measured.

## Performance optimization experiments for sm_121 — still to do on hardware

1. Record actual versions, GPU utilization, peak resident memory, NVLink absence,
   NIC speeds, NCCL transport and `GB10_PROFILE` settings from **both** containers.
2. Verify parity: `python -m unittest gb10.test_attention`, then two-rank
   `GB10_SMOKE_STEPS=8`, then deterministic short-seed loss comparison.
3. Microbenchmark SDPA mixed widths on sm_121 (see `gb10/bench_attention.py`)
   and compare against FlexAttention / cuDNN SDPA with paired-head sequences
   and true ragged window masks; group/bucket documents if CPU dispatch dominates.
4. Retune ANVIL Gram GEMMs for sm_121: benchmark ATen vs **new Blackwell**
   symmetric kernels, not the H100 128×128×64 tile configs.
5. Benchmark BF16 CE slab sizes and only then implement a Blackwell-specific
   FP8 candidate GEMM + fused CE. The Hopper inline PTX / descriptor pipeline
   is not the default and has not been proven safe on sm_121.
6. Profile all_gather/reduce_scatter/all_to_all over 200Gbps fabric versus
   gradient computation; try wider pull cycles, compression and overlap
   only with end-to-end loss-equivalence checks.
7. Measure scaling the table 21M -> 42M -> 84.6M while watching unified-memory
   pressure and paging. Choose a table size from quality-per-second, not H100 defaults.
8. Once stable, add **GB10-specific** CUDA graph capture after testing address
   stability and PyTorch's ability to capture native attention/communication.
9. Rerun full validation seeds with restored 328M token budget and the
   original #92 table before reporting any comparative quality result.

### Known unresolved risks

- The underlying n-gram/value embedding and communication Triton kernels are
  compiled at runtime; we have not tested their codegen on an actual sm_121.
- CPU-looped segmented SDPA is a correctness baseline, not a fast final kernel.
- Most of the H100 optimized CE/FP8 paths are intentionally disabled. FP8
  reintroduction requires a benchmarked **GB10-native** implementation.
- A full run on GB10 has NOT been executed, and multi-node RDMA is NOT
  automatically configured. The Docker runtime must provide NCCL and devices.
- The refactored master track contains additional verification behavior
  (canonical token masking); this branch is not bytewise identical to the
  historical PR #360.

References: NVIDIA DGX Spark hardware/user guide and NVIDIA's "NCCL for
Multiple Sparks" playbook. Review NIC topology and container NCCL transport
on each physical setup rather than assuming the full 200Gbps link is in use.
