# GB10 FP8 MLP validation: 128 KiB shared-memory failure

Follow-up: the explicitly selected `GB10_FP8_MLP_IMPL=hopper` now has
[native SM121 TMA tiles](tma-mlp-sm121.md) and can use the fused kernel in
validation. The default portable implementations retain the fallback below.

## Reproduced failure

Native NVIDIA GB10 `(12,1)`, PyTorch `2.14.0a0+b2c75dd062.nv26.09`,
CUDA `13.4`, Triton `3.8.0`, existing `modded-nanogpt:92-gb10` image.

With `NANOGPT_TARGET=gb10` and `GB10_MLP_BACKEND=fp8`, compiling the real
`GPT._mlp(None, x, w1, w2, None)` dispatch under `torch.no_grad()` and
`torch.compile(dynamic=False, fullgraph=True)` failed for BF16 inputs
`x=[128,768]`, `w1=w2=[2816,768]`:

```text
linear_relu_square_kernel
torch._inductor.exc.InductorError: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376.
```

Inductor also tried disabling pipelining; the resulting kernel still exceeded
the limit. The legacy `fp8` backend dispatched validation to the Hopper TMA
BF16 kernel, although SM121 priming only covered the FP8 training variants.

## Fix and regression checks

All GB10 backends now dispatch MLP calls without FP8 arguments to standard
BF16 linear → ReLU² → linear operations. The FP8 training branches and the
H100 dispatch are retained. No attention tile settings were changed.

`tools/gb10_kernel_probe.py --eval-mlp-only` exercises the real dispatch with
768→2816→768 projections, 128/129/8192 rows, and an independent FP32 reference.
Native GB10 results after the fix:

| Backend | All three row counts compile and run | Relative L2 error |
| --- | --- | --- |
| `bf16` | PASS | < 0.03 |
| `fp8` | PASS | < 0.03 |
| `fp8_sm121` | PASS | < 0.03 |

The full default operator probe with `--capture-attention` also passed,
including attention graph replay, QKV/RoPE gradients, full/sampled CE,
sparse updates, ANVIL, and BF16-mode FP8 refresh graph checks. CPU tests:
11 passed. Syntax compilation and `git diff --check`: passed.

The complete 19,104-step training and final validation were not rerun for
this fix; no throughput or convergence improvement is claimed. Reproduction
commands are in [GB10.md](../../GB10.md).
