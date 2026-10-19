# GB10 performance audit

Status: **BLOCKED**. No GB10 device or CUDA container is available, so there
are no measured latency, memory, throughput, communication, or validation-loss
values. In particular, this run makes no claim about historical 39.914 seconds
or 3.2773 CE.

The required benchmark commands remain pending on each node:

```text
python -m gb10.bench_accelerators --kind all
GB10_GRAPH_SEGMENT_LOOKUP=separate python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
GB10_GRAPH_SEGMENT_LOOKUP=fused python -m gb10.bench_graph --tokens 1024 --qk 64 --v 128
```
