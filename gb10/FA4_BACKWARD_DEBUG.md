# FA4 SM121 backward error (reported diff=1544) — diagnosis and safe isolation

## Source/commit status

The user's local Codex change `4044024` has been reported as:
- `window_size` -> CuTe `Int32`
- `cute.make_fragment` -> `cute.make_rmem_tensor`
- `nvvm.atomicrmw(res=...)` -> `results=...`

It now reportedly JIT-compiles both directions but fails the old
`test_fa4_varlen_mixed_heads_forward_backward` at **max gradient diff
1544**. **This short commit is not currently reachable from the connected
GitHub repository**. The current branch therefore does not contain, audit,
or overwrite its source changes. Publish a branch/ref containing 4044024
before modifying the corresponding external CuTe kernels.

Compilation is not gradient correctness. Do not increase tolerances, turn
on FA4 in training, or silently fall back to an incompatible kernel.

## A. Isolate the first failing dimension / feature on the actual GB10

Inside the existing CUDA 13.4 / PyTorch SM121 Docker container, with the
locally patched bundle that includes 4044024 mounted:

```bash
export GB10_PROFILE=1
export GB10_FA4_LOCAL_PATH=/opt/flash-attn-4-sm120

# JIT and read-only diagnostics (no training, no optimizer):
python -m gb10.diagnose_fa4_backward --repeats 3 \
  --json gb10/results/fa4_backward_diagnostics.json
```

Run cases **in this order**:
1. `dense-noncausal-64`: one document, Dqk=Dv=64, no mask.
2. `dense-causal-64`: same, causal but no sliding window.
3. `varlen-causal-64`: four ragged documents, no local window.
4. `single-window-64`: causal window with one document.
5. `varlen-window-64`: causal sliding + ragged metadata.
6. `varlen-window-mixed`: QK=64, V=128; zero-padded Q/K.
7. `varlen-window-128`: full 128/128.

The JSON report gives per-test, per-run and per-tensor `out,dq,dk,dv`:
maximum absolute error, p99, normalized RMS, reference RMS/max,
worst-element flattened index and repeated-run stability.
The program exits **nonzero** if an experiment fails.
`GB10_ATTN_BACKEND=sdpa` stays the production default.

Interpretation:
- **Only dK / dV fail**: focus on the SM120 backward's gradient atomics,
  accumulation types (FP32 vs BF16), shared-memory layout, output
  initialization, `nvvm.atomicrmw(results=...)` operand contract and
  write scope. Verify repeatability: changing huge outliers across runs
  suggests a race/uninitialized scratch.
- **Only dQ fails**: focus on backward softmax scale, `dP`, `dS`,
  QK zero pad slicing, row-wise reduction, LSE and mask positions.
- **All gradients fail but output passes**: backward ABI, wrong saved
  intermediates/layout, accumulation or upstream gradient stride.
- **Failure first at causal or local window**: forward/backward mask
  disagreement, row/col offset or CuTe dynamic `Int32` specialization.
  FlashAttention's local `window_size_left` counts offsets
  **inclusively**, so compare the allowed key range to SDPA exactly.
- **Failure only at varlen**: offsets, max-seqlen, per-segment local
  indexes, padded/duplicate BOS endpoints or dQ/dK scatter indexing.
- **Failure only at QK=64, V=128**: padded Q/K dimension, shape
  assumptions, scale must stay 0.13 rather than becoming 1/sqrt(128).
- **Forward itself wrong**: do not blame CuTe backward yet. Fix the
  forward masks/position mapping before investigating gradients.

Additional experiments: inspect original CUTLASS/CuTe DSL version and
bundle revision; repeat with one/four heads; single segment 128/256;
window `None`, 64, 128, 256; upstream `grad_output` of ones
and random numbers. Do not recompile the entire train job to iterate.

## B. Temporary diagnostic bridge, NOT a production FA4 fix

`GB10_ATTN_BACKEND=fa4-safebwd` enables:
- real third-party FA4 **forward**;
- unconditional per-call comparison to the trusted SDPA forward;
- a recomputed SDPA **backward**, bypassing the broken CuTe backward;
- a real-GPU preflight test of the same path before the large sparse
  n-gram table is allocated.

```bash
export GB10_ATTN_BACKEND=fa4-safebwd
unset GB10_CUDA_GRAPHS
python -m unittest \
  gb10.test_accelerated.AcceleratedGB10.test_fa4_forward_sdpa_backward_debug_mode -v
```

This incurs substantial duplicate computation, can slow training
dramatically, and the gradient is for SDPA's mathematically equivalent
attention operation, **not the derivative of the actual rounded FA4
forward**. It is useful only for isolating issues and small smoke tests.
No change is made to production FA4 or native SDPA. CUDA Graph capture
with this debug bridge is prohibited.

## C. Required upstream/backend fix before native FA4 backward acceptance

1. Obtain/publish the complete 4044024 diff **and full JSON diagnostic
   output**. The isolated failure determines which exact `flash_bwd*`
   kernel needs a patch.
2. Validate the bundle's specific CUTLASS DSL version against the
   matching Dao-AILab SM120 implementation. The bundle's public model
   card mainly certifies dense/varlen causal + noncausal, **not** the
   combined #92 causal sliding + mixed QK/V backward.
3. If dK/dV fail with 1544 outliers: verify accumulator stores and the
   changed `nvvm.atomicrmw` function ABI. The keyword rename is a
   syntactic compatibility fix, not guaranteed semantic equivalence.
   Check the actual CUDA IR operand order, result shape, memory order
   and types. Compare with correctly version-matched upstream SM120
   backward code and test zero-init and repeated execution.
4. After the native kernel change, run all seven isolation cases
   with two seeds and 2+ repetitions, then
   `python -m unittest gb10.test_accelerated -v`.
5. Only enable native `GB10_ATTN_BACKEND=fa4` if **all**
   outputs and three gradients match, and after real two-node 8/128-step
   gradient/CE convergence checks. Benchmark fwd+bwd against native SDPA,
   including launch overhead and unified-memory pool; if slower or
   numerically wrong, use native SDPA.

Never treat JIT compilation alone as a pass.
