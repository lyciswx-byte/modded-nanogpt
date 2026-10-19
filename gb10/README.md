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
The reference implementation iterates document boundaries on **CPU-resident** metadata,
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

## SM121 accelerated kernels (experimental, implemented on this branch)

The new kernels are **not** copies of the Hopper implementation:
- `gb10/fa4.py`: FA4/CuTe SM120 consumer-Blackwell bundle adapter, variable-length
  and mixed `(Dqk, Dv) = (64,128), (64,64), (128,128)`. This is a
  hardware-specialized **integration of an externally maintained FA4 kernel**,
  not a claim that we authored the third-party FA4 CUDA/CuTe implementation.
  It requires the `SecondNatureComputing/flash-attn-4-sm120` version-1 bundle.
  `hybrid` dispatch selects FA4 at/above `GB10_FA4_MIN_SEQLEN` and native
  per-segment SDPA for shorter batches. The FA4 path is never silently invoked
  if dependencies are missing. See
  https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120
  (BSD-3-Clause).
- `gb10/fp8_linear.py`: new Triton 1024-element per-program E4M3 quantizer,
  dynamic per-tensor amax scales with saturation, and SM121-architecture
  dispatched `torch._scaled_mm` GEMM for QKV, attention-O, both MLP paths
  and LM projection. Unlike H100's `tcgen05` or `WGMMA`, this requires
  a runtime supporting warp-level FP8 MMA on **SM121**. For the initial
  accelerated route, backward GEMMs intentionally retain BF16 precision;
  benchmark before attempting FP8 backward. The weight transpose follows
  the bank's actual layout, not a standard `nn.Linear` down weight.
- `gb10/fused_fp8_ce.py`: new tiled Triton fused softcap / log-sum-exp /
  multi-token loss / prefix loss / gradient for **FP8 E4M3 logits**.
  First stage sums over 1024-class tiles, second generates row loss and
  normalization, backward generates BF16 logit gradients directly without
  a probability tensor. Target duplication and MTP boundary masks are handled.
  Quantization and the straight-through logit gradient introduce numerical
  error relative to the stock FP8-H100 kernel; the code requires on-device
  parity testing before use.

**How to enable** — first install the optional accelerated dependencies inside
your existing GB10 PyTorch Docker container without allowing pip to replace
the vendor-tested torch, triton or CUDA stack:

```bash
python -m pip install --no-deps -r gb10/requirements-accelerated.txt
python -m unittest gb10.test_attention gb10.test_loss gb10.test_model_math -v
python -m unittest gb10.test_accelerated -v
python -m gb10.bench_accelerators --kind all
```

One GB10 host at a time may run these benchmarks. The two-host trainer still
requires both containers to participate in the same process group.

After GPU tests pass on BOTH hosts, set identical flags on both hosts:

```bash
export GB10_ATTN_BACKEND=hybrid      # sdpa | fa4 | hybrid
export GB10_FA4_MIN_SEQLEN=384       # threshold is a profiling starting point
export GB10_FP8_LINEAR=1             # quantize/GEMM with sm121-capable PyTorch
export GB10_FUSED_FP8_CE=1           # sampled and full CE over E4M3 logits
export GB10_SMOKE_STEPS=8
bash gb10/run_two_nodes.sh
```

The trainer executes fail-closed accelerated-kernel parity tests **on each
rank before allocating the 21M-row hash table**. It does not catch and
silently mask compilation or numerical errors; if a kernel is unsupported,
the run should stop. Selectively disable switches to bisect an incompatibility.
For a BF16 baseline, omit the four accelerator exports.

**Caveats:** SM121 / consumer Blackwell cannot use SM100's `tcgen05`/TMEM
path or H100's WGMMA binaries. The external SM120 FA4 has documented
shape and correctness coverage, but untested packed/varlen memory layouts
may still fail on some driver/FA4 releases. Its short-sequence launch
overhead can exceed the gain; benchmark both forward and backward. E4M3
quantization may raise the final FineWeb loss. The accelerated route has
not been executed on your two GB10 machines yet, and neither its optimal
dispatch threshold nor an end-to-end speedup has been established.

**Acceptance criteria for turning on by default:** exact batch-boundary
matching; per-head shape and forward/backward parity tests on sm_121;
NaN-free 8/128/full-step smokes on both nodes; sampled/full CE target
mapping parity; measured per-stage latency, peak unified memory,
NCCL throughput; and final validation vs the BF16 baseline across
multiple seeds. Until then, BF16 SDPA is the default.

## Two-node CUDA graph capture and SM121 Triton Tile calibration (October 2026)

Additional source:
- `gb10/tile_tuner.py`: ANVIL's symmetric `XXT`, `XTX`, `b*A+c*A@A`
  kernels now have an **on-device** Tile search over 32/64/128 M/N,
  K=32/64, 4/8 warps, 2/3 pipeline stages. It compiles and verifies
  output against ATen before selecting the quickest valid GB10 configuration.
  A configuration is cached per kernel, matrix shape, stride and dtype,
  and cannot retune in a CUDA graph.
- `gb10/triton_tiles.py`: independently measures 256/512/1024/2048
  element Triton Tile sizes for FP8 E4M3 quantization and CE softcap
  reduction; numeric rejection precedes latency choice. Frozen for a
  worker BEFORE capturing the first step.
- `gb10/graph_attention.py`: a **graph-safe** variable-length attention
  implementation. In contrast to FA4/CPU-segmented SDPA it does not
  read document boundaries on the host. GPU searchsorted finds the
  document ID of each query/key during every replay, so different
  batches with the same shapes can change BOS positions safely. It
  only gathers a query tile's causal key window (not a giant T×T mask).
  This is a native SDPA implementation, not an SM100/SM90 FlashAttention
  binary nor a claim to be faster than the external SM121 FA4.
- `gb10/capture.py`: uses the proven shape-key protocol from
  `track_1_short/perf/cuda_graphs/`: captures and self-checks every
  unique **forward and backward** graph, plus the **three ANVIL
  matrix-bank optimizer** graphs. The FP8 input/weight quantization and
  fused FP8 CE forward/backward are inside the step graphs, so **no
  separate H100-style FP8 refresh graph is needed**. After a warmup
  the model, optimizer and sparse table are reset; graph pointers and
  optimizer-bank aliases are verified; a missing step graph then
  becomes an error, never a timed-region fallback.
- `gb10/test_graph_attention.py` and `gb10/test_tiles.py`: CPU
  numerical parity with eager segment-local attention, and real-SM121
  CUDA graph replay with changing document boundaries/Gram inputs.

**What is NOT captured:** the CPU data loader, candidate construction
and uploads, the CPU shard metadata, dynamic n-gram/Value Embedding
all-to-all / row-prefetch, 2-node NCCL ReduceScatter/AllGather/AllReduce,
sharded/replicated Adam updates, tail-shipping and validation. These
have dynamic host/collective lifetimes; capturing the entire distributed
training step including NCCL requires a separate, topology-specific
correctness and graph registration study. The new feature therefore
means *all differentiable training compute + ANVIL banks captured*, not
a single graph containing every distributed operation.

### Run the GB10 Tile tuner and compute graphs

Use the same flags on both nodes. Inside each existing NVIDIA GB10 Docker
(with sm121-compatible PyTorch, NCCL, Triton and the FineWeb data mounted):

```bash
git fetch origin && git checkout 2xGB10
python -m unittest gb10.test_graph_attention -v
python -m unittest gb10.test_tiles -v

export GB10_GRAM_BACKEND=triton
export GB10_TUNE_TRITON_TILES=1
export GB10_CUDA_GRAPHS=1
export GB10_ATTN_BACKEND=graph

# Optional FP8 projection and fused CE (each separately parity-tested):
export GB10_FP8_LINEAR=1
export GB10_FUSED_FP8_CE=1

# Keep the first run short; graph capture still warms all required keys.
export GB10_SMOKE_STEPS=8
export MASTER_ADDR="<RANK0_IP_ON_FAST_FABRIC>"
export NODE_RANK=0 # 1 on second GB10
bash gb10/run_two_nodes.sh
```

No FA4 host-varlen/hybrid option may be combined with `GB10_CUDA_GRAPHS=1`.
That path computes CPU sequence offsets at dispatch time and freezing it
in a graph would silently attend across the wrong documents.
Choose `GB10_ATTN_BACKEND=fa4` or `hybrid` with graphs OFF, and
`GB10_ATTN_BACKEND=graph` with graphs ON until a graph-safe FA4
adapter has passed live-offset replay tests.

Tile calibration, per-configuration graph capture, self-check, and
post-warmup reset are outside training wall time. Real end-to-end cold
startup **does** include their costs, and no speedup is claimed here.
The graph capture can increase the unified-memory pool substantially:
`GB10_GRAPH_MAX_RESERVED_GIB=96` and
`GB10_GRAPH_MIN_HOST_AVAILABLE_GIB=8` are fail-closed guards.
`GB10_GRAPH_QUERY_TILE=128` (64,128,256 supported) and
`GB10_GRAPH_TILES_PER_GROUP=4` (1..16) control graph attention.
These are initial safety defaults, not measured optimal GB10 values.
The hardware test suite must run on BOTH nodes prior to claiming correctness.

### Compare graph replay against eager on the actual GB10

```bash
GB10_GRAPH_QUERY_TILE=64 python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
GB10_GRAPH_QUERY_TILE=128 python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
GB10_GRAPH_QUERY_TILE=256 python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
python -m gb10.bench_graph --tokens 2048 --qk 128 --v 128 --window 896
```

The measured output includes both **forward and backward**, median eager
and graph replay time, graph-pool reserved GiB and gradient parity after
changing the live document offsets. A small isolated attention benchmark
is not proof of an end-to-end two-machine speedup. Record settings and
driver/PyTorch versions with each result.

## Post-implementation audit fixes (2026-10-18)

See [AUDIT.md](AUDIT.md) for severity, implementation details and acceptance.
Major changes: import-time **Hopper SM90 NVRTC CE** was suppressed for
`GB10_PROFILE=1`; symmetric Gram autotuning now excludes mirrored rectangular
M/N Tiles; FineWeb validation streams `GB10_VAL_PULL_WINDOW=2` batches rather
than retaining ~160 GPU batches; FP8 CE calibration selects a separate
`GB10_CE_TUNE_ROWS=64` block policy at 10,240 / 14,336 / 24,576 / 50,304
vocabulary widths; and graph mode defaults to PyTorch's native allocator
unless the container already provides `PYTORCH_ALLOC_CONF`.

These are **code-level corrections**. Only CI CPU/reference checks have been
run; an actual 2×GB10 end-to-end smoke, numerical FP8 graph backward and
full FineWeb validation remain hardware acceptance blockers. Do not report
this branch as measured-faster or record-equivalent until those tests pass.

## Latest follow-on work and Codex handoff

The repository now contains a task-ready [Codex prompt](CODEX_PROMPT.md)
covering correctness gates, real two-node SM121 profiling, Triton Tile/FA4/FP8
experiments, GPU Graph replay and reproducible benchmark reporting.

`gb10/graph_attention.py` also supports
`GB10_GRAPH_SEGMENT_LOOKUP=separate|fused`; `fused` issues one
`searchsorted` over concatenated Q/K positions instead of two.
Both modes are compared on CPU for forward/backward parity over all
record #92 QK/V width combinations and duplicate BOS boundaries. The default
remains `separate` until on-device timing shows a real gain; use
`gb10/bench_graph.py` to compare them on each GB10.

For code-level regression and hardware acceptance requirements, consult
[AUDIT.md](AUDIT.md). Static/CPU CI cannot certify two-node training.

## Codex two-node feedback and Oct-18 follow-up (branch `codex/reexecute-ff98`)

The remote feedback branch was fast-forward-integrated into `2xGB10`.
The initial real two-rank BF16 eager 8-step smoke PASSED (rank0 13.19s,
19.39GiB allocated); NCCL used `NET/Socket`, not verified RoCE/RDMA.
The reported 196,608 tokens were **overcounted** using the maximum
stage-3 local batch. With default `GB10_BATCH_DIVISOR=16`, first 8
steps have 65,536 GLOBAL tokens. New trainer logs exact actual batch
token totals after every configured reporting interval.

FA4's local SM120 bundle failed at
`cutlass.cute.core.ThrMma`, which CUTLASS DSL moved to `cutlass.cute`.
The explicit FA4 loader now applies `gb10.cute_compat` before import:
it restores only the **real moved type objects** (ThrMma, TiledMma,
ThrCopy, TiledCopy), not an emulation. This is a source API annotation
compatibility layer, **not** a claim that FA4 JIT or backward passed.
To disable it, set `GB10_FA4_CUTE_COMPAT=0`.

On each GB10 with the supplied local FA4 bundle mounted:
```bash
export GB10_FA4_LOCAL_PATH=/opt/flash-attn-4-sm120
export GB10_FA4_CUTE_COMPAT=1
python -m unittest gb10.test_cute_compat -v
python -m unittest gb10.test_accelerated.AcceleratedGB10.test_fa4_varlen_mixed_heads_forward_backward -v
```
Capture the **entire** chained traceback if that still fails. Avoid
blindly downgrading the container's tested CUDA/PyTorch/CUTLASS stack.

For RDMA visibility, run `python -c 'from gb10.fabric import
report_fabric; report_fabric()'` on each container. Optional
`GB10_REQUIRE_RDMA_DEVICE=1` aborts if sysfs and device nodes are not
visible; it **does not certify** that NCCL selected RDMA. For actual
RoCE investigate host driver, Docker device passthrough and NCCL
backend logs with the two-node NCCL tests. Socket can be correct but
may limit performance.

See [CORRECTNESS](results/CORRECTNESS.md) and
[PERFORMANCE](results/PERFORMANCE.md) for honest PASS/BLOCKED entries.

### Verified optional two-host RoCE transport profile

A subsequent Codex feedback run exposed `/dev/infiniband` to the two
GB10 containers, selected `NCCL_NET=IB` and the *actually observed*
`rocep1s0f1,roceP2p1s0f1` HCAs. Both ranks confirmed `NET/IB` and
completed the same eight-step smoke. On these two specific devices,
set **on each node**:

```bash
export GB10_ROCE_PROFILE=1
export NCCL_DEBUG=INFO
# optional topology overrides if device names differ:
# export NCCL_IB_HCA="<your-active-HCA-ports>"
# export NCCL_SOCKET_IFNAME="<your-fast-IP-interface>"
bash gb10/run_two_nodes.sh
```

This flag enforces RDMA device visibility, selects the measured HCA pair
unless overridden and rejects an inactive link if the `rdma` utility
is installed. The Docker runtime still needs host networking and
`/dev/infiniband` passthrough, plus adequate `memlock`.
**Do not enable GB10_ROCE_PROFILE indiscriminately:** interface names
are host-specific; absent hardware should fail instead of silently
pretending RDMA is active.

The two single-run rank0 timings of 13.19s and 12.69s are not an
end-to-end speedup measurement; see [PERFORMANCE](results/PERFORMANCE.md).
