# GB10 performance audit

Status: **BLOCKED**. No GB10 device or CUDA container is available, so there
are no measured latency, memory, throughput, communication, or validation-loss
values. In particular, this run makes no claim about historical 39.914 seconds
or 3.2773 CE.

The two-node NCCL sanity completed successfully with AllReduce value `3.0` on
both ranks. With `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`, and the
container exposing `/dev/infiniband` plus unlimited memlock, NCCL selected
`NET/IB/0` and `NET/IB/1` across the two active RoCE ports. This is transport
validation only; no bandwidth benchmark was run. With the local FineWeb data mounted read-only, the
8-step smoke completed on both nodes:

- global token budget: 196,608 tokens (8 × 12,288 local tokens × 2 ranks)
- rank-0 elapsed time: 13.19 s
- rank-0 allocated memory at step 8: 19.39 GiB
- settings: eager SDPA, ATen Gram, no CUDA Graph/FP8/FA4 acceleration; NCCL IB

This is a smoke result, not a full-run throughput or validation-loss claim.

The required benchmark commands remain pending on each node:

```text
python -m gb10.bench_accelerators --kind all
GB10_GRAPH_SEGMENT_LOOKUP=separate python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
GB10_GRAPH_SEGMENT_LOOKUP=fused python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
```
