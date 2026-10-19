#!/usr/bin/env bash
# Run INSIDE each GB10's existing NVIDIA Docker container (one GPU per node).
set -euo pipefail
: "${MASTER_ADDR:?set the IP of the rank-0 GB10 (reachable over ConnectX-7)}"
: "${NODE_RANK:?set NODE_RANK=0 on the first GB10, =1 on the second}"
: "${MASTER_PORT:=29500}"
if [[ "$NODE_RANK" != 0 && "$NODE_RANK" != 1 ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
fi
export GB10_PROFILE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# The two measured GB10 hosts support RoCE over active ConnectX-7 HCAs.
# Opt in explicitly: hardware names are site-specific and must NOT become
# defaults on all sm_121 Docker installations.
if [[ "${GB10_ROCE_PROFILE:-0}" == "1" ]]; then
  export NCCL_NET="${NCCL_NET:-IB}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
  export GB10_REQUIRE_RDMA_DEVICE=1
  if command -v rdma >/dev/null 2>&1; then
    if ! rdma link 2>/dev/null | grep -q 'state ACTIVE'; then
      echo "[GB10] RoCE requested, but no ACTIVE RDMA link; refusing Socket fallback" >&2
      exit 3
    fi
  fi
  echo "[GB10] explicit RoCE profile: net=${NCCL_NET} HCA=${NCCL_IB_HCA}" >&2
fi
# PyTorch creates a small CPU/Gloo process group alongside NCCL. On hosts
# whose hostname resolves to 127.0.1.1, leaving this unset makes startup fail
# before NCCL even reaches the training loop. Reuse the explicitly selected
# fabric interface unless the caller supplied a separate Gloo interface.
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
fi
# GB10 sm_121 needs an appropriate CUDA-13 ptxas for runtime-generated Triton.
if [[ -x /usr/local/cuda/bin/ptxas && -z "${TRITON_PTXAS_PATH:-}" ]]; then
  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
fi
# Do not invent an RDMA port name. Set NCCL_SOCKET_IFNAME / NCCL_IB_HCA
# after checking 'ip -br addr', 'ibv_devinfo', and 'nccl-tests' on both hosts.
echo "[GB10] node=${NODE_RANK} rendezvous=${MASTER_ADDR}:${MASTER_PORT} " \
     "socket_if=${NCCL_SOCKET_IFNAME:-auto} net=${NCCL_NET:-auto} " \
     "hca=${NCCL_IB_HCA:-auto} table=${GB10_NGRAM_ROWS:-21150720}"
exec torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m gb10.train
