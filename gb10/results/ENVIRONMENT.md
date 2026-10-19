# GB10 environment audit

Status: **PARTIAL** on 2026-10-18 (America/Goose_Bay).

The repository checkout is commit `ff98b809e3d8114f690292363d89659fa20a0620`
on branch `codex/reexecute-ff98`. One local GB10 is available
through `modded-nanogpt:92-gb10` (NVIDIA Release 26.09).

| Field | Node 0 | Node 1 |
|---|---|---|
| GPU / capability | NVIDIA GB10 / SM121 | NVIDIA GB10 / SM121 |
| Driver / CUDA runtime | 580.178.04 / CUDA 13.4 | 580.178.04 / CUDA 13.4 |
| PyTorch / Triton / NCCL | 2.14.0a0+b2c75dd / 3.8.0 / 2.31.2 | 2.14.0a0+b2c75dd / 3.8.0 / 2.31.2 |
| FA4 / CuTe | Local bundle import failed (`ThrMma` moved); compatibility shim implemented, hardware re-test pending | hardware re-test pending |
| ConnectX-7 interface / transport | `enp1s0f1np1`; HCA `rocep1s0f1,roceP2p1s0f1`; initial Socket, later verified `NET/IB` | `enp1s0f1np1`; same HCA pair; initial Socket, later verified `NET/IB` |

`nvidia-smi`, the CUDA/PyTorch capability probe, and the two-node launch passed
in the container. NCCL used Socket rather than RDMA on the selected interface.

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
