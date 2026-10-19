# GB10 decisions

- Kept the branch on `2xGB10` and did not change the H100/master path.
- Kept FA4, FP8, fused CE, Triton Gram, and CUDA Graph paths opt-in; no
  unmeasured SM121 kernel is enabled by default.
- Recorded the portable compile and policy/import results separately from the
  blocked torch and hardware suite.
- Hardware acceptance is deferred until both existing ARM64 CUDA containers
  provide SM121 capability, versions, NCCL transport, and reproducible two-rank
  logs.
- Do not enable FP8 linear or FA4 defaults: the local GB10 run found FP8
  projection parity failure and FA4 bundle execution failure.
- Corrected Triton FP8 masked-load literals and passed the fused FP8 CE
  forward/gradient gate in the current container.
- Re-executed the mandatory portable checks on `2xGB10` at
  `b09d028e6d3031465be7814d789e78a48e885d19`. The host-level compile and
  dependency-free tests pass; torch-backed tests and two-node acceptance remain
  explicitly blocked outside the CUDA container and second host.

- Retested the supplied local FA4 bundle on both GB10s. It loads successfully,
  but its sliding-window JIT has a CuTe `Int32` versus Python `int` annotation
  mismatch. Kept FA4 disabled by default pending a bundle-side compatibility
  fix and repeated forward/backward parity.
