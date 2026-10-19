# GB10 environment audit

Status: **PARTIAL** on 2026-10-18 (America/Goose_Bay).

The repository checkout is commit `ff98b809e3d8114f690292363d89659fa20a0620`
on branch `codex/reexecute-ff98`. One local GB10 is available
through `modded-nanogpt:92-gb10` (NVIDIA Release 26.09).

| Field | Node 0 | Node 1 |
|---|---|---|
| GPU / capability | NVIDIA GB10 / SM121 | BLOCKED |
| Driver / CUDA runtime | 580.178.04 / CUDA 13.4 | BLOCKED |
| PyTorch / Triton / NCCL | 2.14.0a0+b2c75dd / 3.8.0 / 2.31.2 | BLOCKED |
| FA4 / CuTe | Local bundle found; import BLOCKED (`ThrMma` missing) | BLOCKED |
| ConnectX-7 interface / transport | BLOCKED (not a two-node test) | BLOCKED |

`nvidia-smi` and the CUDA/PyTorch capability probe passed in the container.
`ip -br addr`, `ibv_devinfo`, and the two-node launch remain unrun.
