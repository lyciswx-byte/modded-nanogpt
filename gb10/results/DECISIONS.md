# GB10 decisions

- Kept the branch on `2xGB10` and did not change the H100/master path.
- Kept FA4, FP8, fused CE, Triton Gram, and CUDA Graph paths opt-in; no
  unmeasured SM121 kernel is enabled by default.
- Recorded the portable compile and policy/import results separately from the
  blocked torch and hardware suite.
- Hardware acceptance is deferred until both existing ARM64 CUDA containers
  provide SM121 capability, versions, NCCL transport, and reproducible two-rank
  logs.
- Do not enable FA4 by default: the local GB10 run is blocked by Hugging Face
  authentication while loading the third-party bundle. Keep FP8 linear opt-in
  pending two-node convergence evidence.
- Corrected Triton FP8 masked-load literals and passed the fused FP8 CE
  forward/gradient gate in the current container.

- Post-`b09d028`: keep FA4 and FP8 Linear opt-in while restoring a
  cuBLASLt-friendly column-major FP8 right operand, aligning ragged batch
  rows to 16, and adapting mixed QK/V (64,128) for FA4 by zero-padding Q/K.
- Keep an **exact quantized-reference** FP8 GEMM gate as well as bounded
  E4M3 RMS/tail error against BF16. Elementwise allclose over ~80k outputs
  is not an appropriate sole approximation test. Full training convergence
  and real SM121 runtime remain separate acceptance criteria.
- Do NOT guess the reason for the third-party FA4 failure: retain chained
  exceptions, gather the actual CuTe traceback and verify dependencies
  on the same GB10 container. No silent SDPA fallback when `fa4` is selected.
