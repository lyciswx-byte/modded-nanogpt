# GB10 correctness audit

## Portable checks

| Result | Exact command | Observation |
|---|---|---|
| PASS | `python3 -m compileall -q gb10 track_1_short` | completed successfully |
| PASS | `python3 -m unittest discover -s gb10 -p 'test_*.py' -v` (policy/import subset) | 5 CPU tests passed |
| BLOCKED | same unittest command (torch modules) | `ModuleNotFoundError: No module named 'torch'` in accelerated, attention, graph attention, loss, model math and tiles modules |
| PASS | `git diff --check` | no whitespace errors |

## Hardware checks

Using `modded-nanogpt:92-gb10`, SM121 attention, graph attention, native loss,
model math, tile kernels and fused FP8 CE passed. The exact command was:
`docker run --rm --gpus all --ipc host --network host -v "$PWD":/workspace/modded-nanogpt -w /workspace/modded-nanogpt modded-nanogpt:92-gb10 python -m unittest gb10.test_import_guards gb10.test_policy gb10.test_attention gb10.test_graph_attention gb10.test_loss gb10.test_model_math gb10.test_tiles -v`.

The initial accelerated run exposed and fixed Triton FP8 masked-load typing and
the new `kernels` trust flag. Fused FP8 CE then passed its forward/gradient
test. FA4 remains **BLOCKED/FAIL**: the loader reaches the third-party bundle
but the bundle itself was not successfully executed in this container.
FP8 projection remains **FAIL** on numerical parity and is still opt-in.
Two-node NCCL smoke and full training remain BLOCKED.
