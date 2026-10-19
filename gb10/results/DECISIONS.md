# GB10 decisions

- Kept the branch on `2xGB10` and did not change the H100/master path.
- Kept FA4, FP8, fused CE, Triton Gram, and CUDA Graph paths opt-in; no
  unmeasured SM121 kernel is enabled by default.
- Recorded the portable compile and policy/import results separately from the
  blocked torch and hardware suite.
- Two real ARM64 SM121 containers and a successful two-rank 8-step BF16 eager
  smoke have now been reported. Full training, FA4 and graph/FP8 convergence
  remain unverified. NCCL used Socket rather than proven RDMA.
- Do not enable FA4 by default: the Hub route initially returned HTTP 401,
  and the available local bundle subsequently failed on the relocated
  CUTLASS `cute.core.ThrMma` annotation. A scoped compatibility shim is now
  available but awaits real SM121 re-test. Keep FP8 linear opt-in pending
  full training convergence evidence.
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
- The supplied local bundle was tested and the exact dependency mismatch is
  now recorded: the installed CuTe/CUTLASS lacks `cutlass.cute.core.ThrMma`.

- Apply a strictly scoped compatibility mapping from legacy
  `cutlass.cute.core.ThrMma` to NVIDIA's real `cutlass.cute.ThrMma`
  and related moved types when FA4 is opted in, rather than fabricating
  a kernel or replacing the entire CUDA stack. Re-test the original
  package; a further CuTe JIT failure must remain fatal.
- Correct 8-step reported token budget to the *stage-0 batch*, not
  maximum per-rank tokens from stage 3; log actual planned/completed
  global tokens during training.
- Surface container RDMA visibility separately from NCCL transport:
  do not claim that Socket on a ConnectX interface is RoCE.
