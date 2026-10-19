# GB10 performance audit

Status: **BLOCKED**. No GB10 device or CUDA container is available, so there
are no measured latency, memory, throughput, communication, or validation-loss
values. In particular, this run makes no claim about historical 39.914 seconds
or 3.2773 CE.

The two-node NCCL sanity completed successfully with AllReduce value `3.0` on
both ranks. NCCL selected `NET/Socket`; IB/SPCX reported no device, so no RDMA
bandwidth claim is made. The 8-step training smoke reached preflight and
memory planning on both nodes but was blocked before step 1 by missing FineWeb
training files, so there are no throughput or loss measurements.

The required benchmark commands remain pending on each node:

```text
python -m gb10.bench_accelerators --kind all
GB10_GRAPH_SEGMENT_LOOKUP=separate python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
GB10_GRAPH_SEGMENT_LOOKUP=fused python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
```
