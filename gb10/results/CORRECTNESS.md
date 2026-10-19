# GB10 correctness audit

## Portable checks

| Result | Exact command | Observation |
|---|---|---|
| PASS | `python3 -m compileall -q gb10 track_1_short` | completed successfully |
| PASS | `python3 -m unittest discover -s gb10 -p 'test_*.py' -v` (policy/import subset) | 5 CPU tests passed |
| BLOCKED | same unittest command (torch modules) | `ModuleNotFoundError: No module named 'torch'` in accelerated, attention, graph attention, loss, model math and tiles modules |
| PASS | `git diff --check` | no whitespace errors |

## Hardware checks

On commit `ff98b80`, the container suite passed the CPU/reference, attention,
graph, loss, model, tile, layout and precision tests before the FA4 test.

Using `modded-nanogpt:92-gb10`, SM121 attention, graph attention, native loss,
model math, tile kernels and fused FP8 CE passed. The exact command was:
`docker run --rm --gpus all --ipc host --network host -v "$PWD":/workspace/modded-nanogpt -w /workspace/modded-nanogpt modded-nanogpt:92-gb10 python -m unittest gb10.test_import_guards gb10.test_policy gb10.test_attention gb10.test_graph_attention gb10.test_loss gb10.test_model_math gb10.test_tiles -v`.

The initial accelerated run exposed and fixed Triton FP8 masked-load typing and
the new `kernels` trust flag. Fused FP8 CE then passed its forward/gradient
test. FA4 remains **BLOCKED/FAIL**: the loader reaches the third-party bundle
but the bundle itself was not successfully executed in this container. The
current exact traceback is HTTP 401 from Hugging Face while resolving
`SecondNatureComputing/flash-attn-4-sm120` without a container credential.
FP8 projection passes the revised parity gate in this commit and remains
opt-in pending two-node convergence evidence.
Two-node NCCL smoke and full training remain BLOCKED.

## Follow-up review of `b09d028` (post-commit patches)

Historical failures at `b09d028e6d3031465be7814d789e78a48e885d19`:
- **FP8 Linear parity FAIL**: the former `_scaled_mm` right operand was
  `wq.T.contiguous()`, converting a column-major GEMM operand back into
  row-major, and odd token-row counts (e.g. 65) were passed directly to
  an alignment-sensitive FP8 GEMM. Fixed via CPU-tested column-major
  `wq.T`, pad-to-16-rows, and trim-back to original shape.
- The old `torch.allclose(atol=.1, rtol=.1)` comparison against an
  unquantized BF16 GEMM incorrectly rejects some *mathematically valid*
  E4M3 outputs: CPU reproducer with the same 65×768×1280 seed has
  nonzero outliers and ~3.8% normalized RMS approximation error. It has
  been replaced with **two independent gates**: strict comparison against
  an explicitly quantized GEMM, and bounded normalized RMS / p99.9 /
  max error compared against BF16. BF16 backward gradients remain checked
  independently. This is a numerical test correction, not a declaration
  that FP8 preserves original training convergence.
- **FA4 FAIL/BLOCKED**: an error inside the third-party SM120/CuTe bundle
  was previously observed but the original exception traceback was not
  captured in the log. The #92 mixed (QK,V)=(64,128) path is now adapted
  by zero-padding Q/K to 128 while **preserving the original attention
  softmax scale** and gradient. Bundle-loader/execution failures retain
  their Python exception chains. Compatibility on the target image is
  still **BLOCKED pending a new real GB10 run and the complete traceback**.

New offline CPU tests:
`python -m unittest gb10.test_accel_layout gb10.test_precision -v`

Required on the actual GB10 image (run each separately, capture full stderr):
```bash
python -m unittest gb10.test_accelerated.AcceleratedGB10.test_fp8_projection_and_grads -v
python -m unittest gb10.test_accelerated.AcceleratedGB10.test_fa4_varlen_mixed_heads_forward_backward -v
```

**Do not mark either historical GPU FAIL as PASS until the two commands above
actually pass inside the original GB10 Docker container.** The GitHub
Actions CPU suite and source checks cannot certify third-party CuTe execution.

Retest with the supplied local bundle used:
`GB10_FA4_LOCAL_PATH=/opt/flash-attn-4-sm120` and a Docker bind mount from
`/media/lycis/data/flash-attn-4-sm120`. The bundle was found, but import failed
with `AttributeError: cutlass.cute.core has no attribute ThrMma`; FA4 remains
BLOCKED by a CuTe/CUTLASS dependency mismatch in the container.

Same-host two-rank NCCL probe was also attempted with
`NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=enp1s0f1np1 torchrun --standalone --nproc_per_node=2`.
It **FAIL**ed with `ncclInvalidUsage: Multiple Ranks are using the same
GPU/Partition`; this is an invalid same-GPU simulation, not a two-node result.
Real two-node training remains BLOCKED pending a second GB10 host.

With the second host available, the real two-node launch reached preflight on
both ranks and initialized NCCL successfully after setting
`GLOO_SOCKET_IFNAME=enp1s0f1np1`. The 8-step smoke then stopped on both ranks
at data loading with `FileNotFoundError: ./data/fineweb10B/fineweb_train_*.bin`.
No training step, loss, or gradient was produced.
