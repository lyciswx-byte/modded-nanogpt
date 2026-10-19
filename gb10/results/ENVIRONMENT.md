# GB10 environment audit

## Current shell re-execution (2026-10-19 UTC)

Checkout: `2xGB10`, commit `b09d028e6d3031465be7814d789e78a48e885d19`.
The host reports `aarch64`, NVIDIA GB10, driver `580.178.04`, and CUDA
`13.0` from `nvidia-smi`. `ip -br addr` shows the up ConnectX interfaces
`enp1s0f1np1` (`192.168.100.10/24`) and `enP2p1s0f1np1`
(`192.168.101.10/24`). PyTorch, Triton, NCCL, and CuTe versions are
**BLOCKED** in this shell because `python3` has no `torch`; the existing CUDA
container must be used for those probes. A second-node probe was not run.

Status: **PARTIAL** on 2026-10-18 (America/Goose_Bay).

The repository checkout is commit `ff98b809e3d8114f690292363d89659fa20a0620`
on branch `codex/reexecute-ff98` for the initial probe. Both GB10 hosts
were later tested using `modded-nanogpt:92-gb10` (NVIDIA Release 26.09).

| Field | Node 0 | Node 1 |
|---|---|---|
| GPU / capability | NVIDIA GB10 / SM121 | NVIDIA GB10 / SM121 |
| Driver / CUDA runtime | 580.178.04 / CUDA 13.4 | 580.178.04 / CUDA 13.4 |
| PyTorch / Triton / NCCL | 2.14.0a0+b2c75dd / 3.8.0 / 2.31.2 | 2.14.0a0+b2c75dd / 3.8.0 / 2.31.2 |
| FA4 / CuTe | Local bundle import failed (`ThrMma` moved); compatibility shim implemented, hardware re-test pending | hardware re-test pending |
| ConnectX-7 interface / transport | `enp1s0f1np1`; HCA `rocep1s0f1,roceP2p1s0f1`; initial Socket, later verified `NET/IB` | `enp1s0f1np1`; same HCA pair; initial Socket, later verified `NET/IB` |

`nvidia-smi`, the CUDA/PyTorch capability probe, and the two-node launch passed
in the container. The initial two-node probe used Socket; the later RoCE-enabled run
confirmed NCCL `NET/IB` on both hosts.

The follow-up code checks whether the new CUTLASS DSL exports the moved
annotation classes at `cutlass.cute`, and makes those **same class objects**
available to the older local bundle's `cutlass.cute.core` imports.
The current branch has no post-fix FA4 GPU result.

`NCCL_SOCKET_IFNAME=enp1s0f1np1` selects the fast IP interface, but
the logged `NET/Socket` is **not** an RDMA result. Query
`gb10.fabric.report_fabric()`, host `ibv_devinfo` and NCCL
`NCCL_DEBUG=INFO` from both containers before changing communication flags.

A later Codex rerun on both hosts provided real RoCE evidence:
`NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`,
exposed `/dev/infiniband` and unlimited `memlock`, with NCCL logs
showing `NET/IB/0` and `NET/IB/1`. The earlier `NET/Socket` is
historical, not the latest reported transport. Other GB10 installations
may use different interface/HCA names.
