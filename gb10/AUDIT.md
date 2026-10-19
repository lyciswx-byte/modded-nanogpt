# GB10 two-node implementation audit — 2026-10-18

This is a **code and CI audit**, NOT two-GB10 hardware certification.
The code builds on the refactored #92 trainer, with intentional changes to
batch sizes, table capacity, precision, device layout and kernels.

## Findings and fixes

| Severity | Finding | Symptom | Change / verification |
|---|---|---|---|
| P0 | `track_1_short/perf/kernels/cross_entropy.py` eagerly compiles H100 `compute_capability="90"` with NVRTC while importing the GPT model, even if GB10 later calls the native CE. | SM121 containers can fail **during module import** before training. | GB10 skips compilation and errors if the SM90 CE is accidentally invoked; H100 still eagerly compiles as before. CPU AST regression `gb10/test_import_guards.py`. |
| P1 | ANVIL `XXT/XTX/poly` kernels mirror each computed output tile to the opposite triangle. The proposed SM121 autotune list included **rectangular** M/N shapes. | Overlapping CTAs can write symmetric tiles concurrently / race. | Restrict to symmetric square M/N geometry and verify the launch contract in `gb10/policy.py`, `test_policy.py`. Numerically profile every remaining candidate on SM121. |
| P1 | GB10 validation retained **all** FineWeb batches and row pulls simultaneously; the 65536-token global validation batch default implies 160 batches. | Over-allocation in 128GB unified memory (OS and GPU share physical RAM), spurious OOM or paging. | `gb10/train.py` now creates/evaluates/releases bounded 1–8 batch windows, default 2. CPU test verifies exactly-once coverage. This changes validation *preparation timing* versus the original H100 record; no 39.9s comparison is claimed. |
| P2 | FP8 CE Tile calibration was fixed at vocabulary 14336, although the record uses 10240 / 14336 / 24576 / 50304 categories. | A Tile that is fast in stage 2 may be slower in early or final stages. | `gb10/triton_tiles.py` calibrates each vocabulary separately; `fused_fp8_ce.py` dispatches the corresponding forward/backward block size. Re-tune outside Graph capture. |
| P2 | Container-unsupported `expandable_segments:True` was enforced even when graph pools are enabled. | Potential allocator incompatibility or fragmentation in GB10 Docker/CUDA configuration. | Graph mode defaults to native allocator without overriding the container's explicit setting. |
| P2 | Historical H100 CUDA Graph logic can silently bake host-side BOS/sequence boundaries. | Replays may attend to the wrong document. | Existing `gb10/graph_attention.py` computes segment identity from GPU offsets on every replay; CPU parity plus real-GB10 graph replay test. FA4/CPU-varlen disallowed in current graph-capture mode. |
| P2 | Compilation/tuning inside Graph capture corrupts the capture protocol. | Capture errors or graph explosion at runtime. | Triton Gram and CE tuning occur before capture. Gram tuner raises if asked to autotune inside capture. Each Graph key is pre-captured and sealed; missing key aborts. |
| P2 | Multiple full-step graph pools can crowd the unified memory. | Increasing GPU reserved bytes while host RAM is exhausted. | Graph pool reservation and MemAvailable/cgroup checks, with fail-fast configurable thresholds. |

## Tests done / not done

**Automated GitHub Actions CPU suite:** Python `compileall`, native attention
and CE reference gradients, QK RoPE, MLP bank orientation, packed Graph attention
versus segmented SDPA for multiple QK/V widths, mirrored Tile safety, validation
window coverage and SM90 eager-import AST guards.

**Hardware-specific tests, explicitly NOT performed:** Triton PTX compilation
on GB10 SM121, FA4 mixed-width backward, FP8 CE numerical/optimizer convergence,
real NCCL/RoCE bandwidth, out-of-memory behavior, 8-step/full two-rank training,
end-to-end wall-time or the 3.28 FineWeb validation threshold.

The branch does **not** capture CPU data ingress, two-node NCCL collectives,
variable-length n-gram sparse row exchange, all Adam updates or validation in
one graph. It captures differentiable compute plus ANVIL bank updates only;
the original #92 architecture and training economics cannot be claimed equal
with the default 2×GB10 reduced token/table settings.

## Hardware acceptance sequence

1. Run CPU tests in both Docker containers; check `sm_121`, pinned CUDA/Triton
   versions and NCCL high-speed interface.
2. Run `python -m unittest gb10.test_accelerated gb10.test_tiles gb10.test_graph_attention -v`
   on both **actual GB10s**. Any numerical failure is a blocker.
3. Compare CE Tiles over each of the four vocabulary widths, as well as
   ANVIL banks' XXT/XTX/poly outputs. Run graph replay with live-changing
   document boundaries.
4. Run a two-rank 8-step smoke; then 128 steps; only then 1194 steps.
   Record effective GB10 global token budget, wall-clock phase timings,
   per-node peak GPU allocated/reserved plus CPU MemAvailable.
5. Re-evaluate 21M -> 42M -> 84.6M n-gram capacity with actual quality and
   memory measurements. Restore a comparable token budget and validation
   conventions before comparing quality to the #92 H100 world record.

Fail-closed is deliberate: runtime kernel or numeric errors do **not** silently
fall back to H100 kernels.
