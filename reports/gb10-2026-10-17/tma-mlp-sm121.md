# Hopper TensorDescriptor/TMA MLP adapted to GB10

The original fused kernel is retained: TMA operand loads, FP8 tensor-core
GEMM, fused ReLU²/dpre, both FP8 output layouts and delayed amax. This is not
the cuBLASLt or pointer-based fallback. Select it explicitly with
`NANOGPT_TARGET=gb10 GB10_MLP_BACKEND=fp8 GB10_FP8_MLP_IMPL=hopper`.

## Configuration and native resource measurements

GB10 uses 64×128×64 tiles, 4 warps and one pipeline stage for all variants.
This replaces the oversized 128×256 Hopper tile on GB10; H100 defaults
remain unchanged. Safe defaults apply before eager priming and during
Inductor compilation. The opt-in TMA backend also uses this fused BF16
forward in validation instead of the previous ATen fallback.

Compiler metadata from actual eagerly compiled `linear_relu_square_kernel`
variants, identified by their `FORWARD` and `USE_FP8` constexpr values:

| Variant | Shared memory |
| --- | ---: |
| BF16 forward | 32,768 bytes |
| FP8 forward, both output layouts | 16,896 bytes |
| FP8 backward, both output layouts | 8,200 bytes |
| GB10 per-block limit | 101,376 bytes |

Native environment: aarch64 GB10 `(12,1)`, PyTorch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA 13.4, Triton 3.8.0,
existing `modded-nanogpt:92-gb10` image. No architecture spoofing.

## Checks

`tools/gb10_tma_mlp_probe.py` passed on GB10:

- Independent quantized-reference output and dx/dW1/dW2 checks, with and
  without folded gain/dp, delayed scales and amax. FP8 output tolerance:
  atol 0.025 / rtol 0.01; gradient relative L2 < 0.045.
- Partial tiles: 144 rows and 320 hidden channels. Full model width 768,
  hidden width 2816, including real 8192- and 24576-token training sizes.
- Cold `torch.compile(fullgraph=True)` forward/backward before eager priming.
- CUDA Graph capture/replay with changing FP8 inputs and incoming gradients;
  outputs, all parameter/input gradients, folded gain gradient and delayed
  amax agree with eager execution.
- Actual GPT validation dispatch compiled at 128, 129 and 8192 rows,
  compared to FP32 math (relative L2 < 0.03).
- All eagerly compiled variants fit the hardware shared-memory limit.

The integrated full operator probe with
`--capture-attention --check-fp8-mlp` also passed with the TMA flags.
The default BF16 full operator probe also passed; CPU tests: 11 passed.
Syntax compilation and `git diff --check` passed.

## Isolated benchmark

`tools/gb10_tma_mlp_probe.py --bench`, 8192×768 input and 2816 hidden
channels, 12 GPU-event samples after warmup (median):

| Eager forward + backward | Time |
| --- | ---: |
| BF16 F.linear + autograd | 6.011 ms |
| Adapted fused TMA FP8 | 3.780 ms |

Observed ratio: 1.590×. This compares different numerical precisions and
excludes model context, cache refresh, optimizer, data loading and losses.
It is not a whole-training speedup. The 19,104-step training and final
FineWeb CE have not been rerun for this backend; default remains BF16.

Run commands and selection flags are documented in [GB10.md](../../GB10.md).
