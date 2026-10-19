# GB10 correctness audit

## Portable checks

| Result | Exact command | Observation |
|---|---|---|
| PASS | `python3 -m compileall -q gb10 track_1_short` | completed successfully |
| PASS | `python3 -m unittest discover -s gb10 -p 'test_*.py' -v` (policy/import subset) | 5 CPU tests passed |
| BLOCKED | same unittest command (torch modules) | `ModuleNotFoundError: No module named 'torch'` in accelerated, attention, graph attention, loss, model math and tiles modules |
| PASS | `git diff --check` | no whitespace errors |

## Hardware checks

All SM121 forward/backward, FA4 masked varlen, CUDA Graph replay, NCCL
two-rank smoke, and full training checks are **BLOCKED**. No GPU, container,
seed, tolerance, loss, gradient, or traceback from real GB10 hardware exists.
