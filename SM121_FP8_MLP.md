# GB10 SM121 FP8 MLP — experimental backend and tile tuning

## Status / honest scope

An independent backend named **`fp8_sm121`** is available on `92-1xGB10`. This implements the same record #92 FP8-cache and custom-gradient contract as the original MLP, without executing its Hopper TensorDescriptor/TMA fused GEMM kernels. It **has not been timed or validated on a physical GB10 in this change**. The previous real-device result (BF16 MLP, 19,104 optimizer steps, FineWeb CE **3.5956**) remains the proven baseline.

This change does **not** implement 8-H100-equivalent gradient accumulation. The default `GB10_MLP_BACKEND=bf16` is intentionally preserved.

### Operator design

The original MLP is `y = relu(x @ W1.T)^2 @ W2`, with a folded residual post-lambda in most training layers. FP8 training uses:

1. E4M3 input and W1 cached weights; **up GEMM** via `torch._scaled_mm` (CUDA cuBLASLt) producing BF16 `pre`.
2. New SM121-friendly, pointer-based Triton **forward epilogue**: ReLU-square in BF16, saturating E4M3 cast, both row-major and transpose layouts, and delayed `post_amax` atomic-max in a **single kernel**. It has no TMA TensorDescriptor.
3. Existing optimized FP8 **down GEMM** (`nanogpt::dp_f8`), reading the same existing W2 column-major cache, with folded post-lambda scale when appropriate.
4. **Backward**: E5M2 quantize/transposed grad, existing `wg2_f8` for dW2; cuBLASLt E5M2 × E4M3 for dpost with `use_fast_accum=False`.
5. New pointer-based Triton **backward epilogue**: `2 * dpost * sqrt(post_f8 * delayed_post_scale)`, BF16 rounding, saturating E5M2 dpre, transpose, and delayed `dpre_amax` in one kernel.
6. Existing `wg1_f8` and `dx_f8` produce dW1/dx. If post-lambda is folded, compute its gradient and apply its scaling exactly as the old custom autograd does.

The original Hopper approach fused GEMM and epilogue in one custom TMA kernel. **This GB10 approach separates GEMM from the fused epilogue**, so the added intermediate BF16 traffic can make it slower. The tradeoff is a platform-compatible candidate with bounded tile sizes, shared-memory usage, and an isolated performance harness. It uses the existing model caches, FP8 refresh graphs, optimizer, CUDA graphs, schedules and sparse embeddings. The SM121 device check in `train_gpt.py` remains active.

The existing `fp8` name still denotes the **original Hopper-style MLP**. The new backend is explicitly named `fp8_sm121`; the two are not aliases.

## Native GB10 commands

On the real GB10 host, build the branch image first:

```bash
git checkout 92-1xGB10
docker build --platform linux/arm64 -f Dockerfile.gb10 -t modded-nanogpt:92-gb10 .
```

**A. Numeric reference and compiled forward/backward**

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -e GB10_MLP_BACKEND=fp8_sm121 \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 \
  python tools/gb10_fp8_mlp_probe.py --check-compile
```

This checks MLP outputs, the dX/dW1/dW2/post-lambda gradients, stored delayed amax, and fullgraph compilation against independent FP8 GEMM/quantization reference math, including full model dimensions. **CUDA Graph capture/replay of the complete model still needs an actual training smoke/full run**.

**B. Compare BF16 and new FP8 MLP, then sweep six SM121 tile options**

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -e GB10_MLP_BACKEND=fp8_sm121 \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 \
  python tools/gb10_fp8_mlp_probe.py --sweep --tokens 8192 --iters 10
```

Sweep repeats **all reference checks** in isolated interpreter processes with `tile_m` in `16,32,64` and `tile_n` in `64,128`. Reported median is **single MLP fwd+bwd only**: no n-gram, attention, data, optimizer, weight refresh or full training, and no graph replay. Also sweep `--tokens 16384` and `--tokens 24576` to choose a configuration appropriate for the different batch stages, then check under CUDA graphs.

Tune statically (at process launch):

```bash
-e GB10_MLP_BACKEND=fp8_sm121 \
-e GB10_FP8_MLP_TILE_M=32 \
-e GB10_FP8_MLP_TILE_N=128
```

The tile values are read at module import and **must not change after torch.compile or during CUDA Graph capture**. Defaults are M=32,N=128; they are starting points, **not measured optima**.

**C. Existing complete operator probe**

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -e GB10_MLP_BACKEND=fp8_sm121 \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 \
  python tools/gb10_kernel_probe.py --capture-attention --check-fp8-mlp
```

**D. Only if all above succeed: training**

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -e GB10_MLP_BACKEND=fp8_sm121 -e TRAIN_SEED=1234 \
  -v "$PWD:/workspace/modded-nanogpt" -w /workspace/modded-nanogpt \
  modded-nanogpt:92-gb10 bash run.sh
```

Do not assume a full-run result exists. Record the actual final CE, wall time, token/s, memory, and any errors. For a controlled BF16-vs-FP8 end-to-end comparison use **identical** FineWeb data and seed, default batch sizes, training schedule and profiling settings. A short bounded training smoke requires respecting the repository's tail-averaging minimum-step assertion (see `GB10.md`).

## Performance / correctness gates

- Maintain the correct `relu(x @ W1.T)^2 @ W2` and optional folded lambda semantics, plus all three weight/input gradients and the scalar post-lambda gradient.
- Validate FP8 STE-style quantization, delayed per-tensor scales, clipping to E4M3/E5M2 finite max, and amax updates; original QKV and CE FP8 paths remain unchanged.
- The standalone probe's tolerances compare **the actual FP8 GEMM/quantized intermediate contract**, not ideal FP32 gradients. Re-check if the underlying NGC PyTorch/CUDA version changes.
- The full `torch.compile` / training CUDA Graph path must execute without allocation/aliasing/capture errors; smoke training must have finite gradients and correct optimizer cadence.
- Measure full training separately. Even if this MLP microbenchmark is faster, a gain in the full model is **not guaranteed** because of cache refresh, memory bandwidth, activation traffic and graph launch overhead.
- If FP8 MLP quality worsens relative to BF16, keep BF16 default and investigate scale histories / rounding before promoting the backend.
- If `torch._scaled_mm` itself is the dominant bottleneck on SM121, profile native CUTLASS/CuTe FP8 GEMM kernels and replace the GEMM behind a well-tested interface. No claim of an SM121-native fused TensorCore/TMA GEMM is made for the current code.

Relevant source: `track_1_short/perf/kernels/mlp_sm121.py` (new Triton epilogues, autograd); `track_1_short/perf/kernels/mlp.py` (existing FP8 GEMM and cache quantizers); `tools/gb10_fp8_mlp_probe.py` (reference, compile, tune); `track_1_short/model/gpt.py` (backend dispatch); `track_1_short/perf/cuda_graphs/fp8_refresh_graphs.py` (refresh graphs).
