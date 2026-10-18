"""Portable SM121 FP8 fused MLP activation kernels and precompile-time tile search.

Used ONLY by the opt-in GB10 FP8 MLP. Unlike the original Hopper kernel, this
path uses masked Triton tl.load/tl.store + FP8 tensor-core tl.dot, not
TensorDescriptor / Hopper-specific TMA descriptors. The GEMM output epilogue
writes FP8 activation/gradient in both layouts needed by the existing
torch._scaled_mm backward. It preserves the original delayed-scale contract.

Tile selection takes place eagerly before torch.compile or CUDA Graph capture:
the compiled forward/backward only read an immutable, preselected tuple.
This is a compatibility and tuning implementation, NOT a claim of measured GB10
speedup. Set GB10_FP8_MLP_AUTOTUNE=0 for deterministic conservative tiles.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources


@dataclass(frozen=True)
class MlpTile:
    m: int
    n: int
    k: int
    warps: int
    stages: int


SAFE_TILE = MlpTile(64, 128, 64, 4, 2)
CANDIDATES = (
    SAFE_TILE,
    MlpTile(32, 128, 64, 4, 2),
    MlpTile(64, 64, 64, 4, 2),
    MlpTile(64, 128, 128, 4, 2),
    MlpTile(64, 256, 64, 8, 2),
    MlpTile(128, 128, 64, 4, 2),
    MlpTile(128, 256, 64, 8, 2),
    MlpTile(64, 128, 64, 4, 3),
)
# This dictionary is written only during the eager, explicit prime call.
_TILES: dict[tuple[bool, int, int, int], MlpTile] = {}


def selected_tile(forward: bool, m: int, n: int, k: int) -> MlpTile:
    """Pure, trace-safe lookup. Unprimed shapes fall back to a conservative tile."""
    return _TILES.get((forward, m, n, k), SAFE_TILE)


def autotune_enabled() -> bool:
    v = os.environ.get("GB10_FP8_MLP_AUTOTUNE", "1")
    if v not in ("0", "1"):
        raise ValueError("GB10_FP8_MLP_AUTOTUNE must be 0 or 1")
    return v == "1"


@triton.jit
def _fp8_relu_square_sm121(
    a_ptr, b_ptr, aux_ptr, out_ptr, out_t_ptr,
    dequant_ptr, post_scale_ptr, dpre_scale_ptr, amax_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    FORWARD: tl.constexpr,
):
    # Matrix operands are contiguous [M,K] and [N,K], respectively.
    # Group launch tiles along N, so independent program instances have no
    # output overlap; atomic_max only touches the scalar amax for this MLP slot.
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        kk = ki * BK + rk
        aa = tl.load(a_ptr + rm[:, None] * K + kk[None, :],
                     (rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        bb = tl.load(b_ptr + rn[:, None] * K + kk[None, :],
                     (rn[:, None] < N) & (kk[None, :] < K), other=0.0)
        acc = tl.dot(aa, tl.trans(bb), acc)
    acc = acc * tl.load(dequant_ptr)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if FORWARD:
        # The reference Hopper epilogue rounds 'pre' to BF16 before ReLU^2.
        pre = acc.to(tl.bfloat16)
        relu = tl.maximum(pre, 0).to(tl.bfloat16)
        post = (relu * relu).to(tl.bfloat16)
        post_fp32 = post.to(tl.float32)
        s = tl.load(post_scale_ptr)
        quant = tl.minimum(post_fp32 / s, 448.0).to(tl.float8e4nv)
        mag = tl.max(tl.max(tl.where(mask, post_fp32, 0.0), axis=1), axis=0)
    else:
        # The FP8 post was emitted by the previous forward epilogue.
        # Match its delayed scale and BF16 dpre rounding, as for the Hopper
        # kernel: dpre = 2 * (dout @ W2) * sqrt(post).
        post_fp8 = tl.load(aux_ptr + rm[:, None] * N + rn[None, :],
                           mask, other=0.0).to(tl.float32)
        post_scale = tl.load(post_scale_ptr)
        dpre = (2.0 * acc * tl.sqrt(post_fp8 * post_scale)).to(tl.bfloat16)
        dpre_float = dpre.to(tl.float32)
        s = tl.load(dpre_scale_ptr)
        quant = tl.maximum(tl.minimum(dpre_float / s, 57344.0), -57344.0).to(tl.float8e5)
        mag = tl.max(tl.max(tl.where(mask, tl.abs(dpre_float), 0.0), axis=1), axis=0)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], quant, mask)
    tl.store(out_t_ptr + rn[:, None] * M + rm[None, :], tl.trans(quant),
             (rn[:, None] < N) & (rm[None, :] < M))
    tl.atomic_max(amax_ptr, mag, sem="relaxed")


def _grid(m: int, n: int, tile: MlpTile):
    return (triton.cdiv(m, tile.m), triton.cdiv(n, tile.n))


def _launch(a, b, aux, out, out_t, dequant, post_scale, dpre_scale, amax,
            forward: bool, tile: MlpTile):
    m, k = a.shape
    n = b.shape[0]
    _fp8_relu_square_sm121[_grid(m, n, tile)](
        a, b, aux, out, out_t, dequant, post_scale, dpre_scale, amax,
        M=m, N=n, K=k, BM=tile.m, BN=tile.n, BK=tile.k,
        FORWARD=forward, num_warps=tile.warps, num_stages=tile.stages,
    )


def fused_activation(a_f8: torch.Tensor, b_f8: torch.Tensor,
                     dequant_scale: torch.Tensor, post_scale: torch.Tensor,
                     post_amax: torch.Tensor, *, aux: torch.Tensor | None = None,
                     dpre_scale: torch.Tensor | None = None,
                     dpre_amax: torch.Tensor | None = None):
    """Mirror the two FP8 modes of mlp.linear_relu_square.

    Forward: return (None, None, fp8_post[M,N], fp8_post_t[N,M]).
    Backward: return (fp8_dpre[M,N], fp8_dpre_t[N,M]).
    """
    assert a_f8.ndim == b_f8.ndim == 2
    assert a_f8.is_contiguous() and b_f8.is_contiguous()
    m, k = a_f8.shape
    n, kb = b_f8.shape
    assert kb == k and m > 0 and n > 0
    forward = aux is None
    if forward:
        assert a_f8.dtype == b_f8.dtype == torch.float8_e4m3fn
        assert post_amax is not None
        out_dtype = torch.float8_e4m3fn
        amax = post_amax
    else:
        assert a_f8.dtype == torch.float8_e5m2 and b_f8.dtype == torch.float8_e4m3fn
        assert aux.dtype == torch.float8_e4m3fn and aux.shape == (m, n)
        assert dpre_scale is not None and dpre_amax is not None
        out_dtype = torch.float8_e5m2
        amax = dpre_amax
    out = torch.empty((m, n), dtype=out_dtype, device=a_f8.device)
    out_t = torch.empty((n, m), dtype=out_dtype, device=a_f8.device)
    # All scale tensors are read live inside CUDA-graph replay (no host scalar capture).
    tile = selected_tile(forward, m, n, k)
    _launch(a_f8, b_f8, aux if aux is not None else out,
            out, out_t, dequant_scale, post_scale,
            dpre_scale if dpre_scale is not None else post_scale, amax, forward, tile)
    return (None, None, out, out_t) if forward else (out, out_t)


def prime_sm121_fp8_mlp(shapes: list[tuple[int, int, int]]):
    """Eagerly pick tile configurations for each (M,N,K), before compiling.

    Run on the actual GB10 with real training shapes. Unsupported/out-of-resource
    candidates are excluded; SAFE_TILE must be valid to proceed. Timings are
    local kernel-only CUDA Graph medians, not training speedups.
    """
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("SM121 FP8 MLP is GB10-only; expected compute capability (12, 1)")
    tune = autotune_enabled()
    if tune:
        from triton.testing import do_bench_cudagraph
    for m, n, k in sorted(set(shapes)):
        for forward in (True, False):
            aa_dtype = torch.float8_e4m3fn if forward else torch.float8_e5m2
            aa = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(aa_dtype)
            bb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            aux = torch.zeros(m, n, device="cuda", dtype=torch.float8_e4m3fn)
            out_dtype = torch.float8_e4m3fn if forward else torch.float8_e5m2
            out = torch.empty((m, n), device="cuda", dtype=out_dtype)
            out_t = torch.empty((n, m), device="cuda", dtype=out_dtype)
            one = torch.ones(1, device="cuda", dtype=torch.float32)
            amax = torch.zeros(1, device="cuda", dtype=torch.float32)

            def run(tile):
                _launch(aa, bb, aux, out, out_t, one, one, one,
                        amax, forward, tile)

            chosen, best_ms = None, float("inf")
            for tile in CANDIDATES if tune else (SAFE_TILE,):
                try:
                    run(tile)
                    torch.cuda.synchronize()
                    ms = do_bench_cudagraph(lambda tile=tile: run(tile), rep=40) if tune else 0.0
                except (OutOfResources, RuntimeError, AssertionError) as exc:
                    print(f"[sm121-fp8-mlp] rejected {tile}: {str(exc)[:160]}")
                    continue
                if ms < best_ms:
                    chosen, best_ms = tile, ms
            if chosen is None:
                raise RuntimeError(f"No valid SM121 FP8 MLP tile for {forward=} {m=} {n=} {k=}")
            _TILES[(forward, m, n, k)] = chosen
            timing = f"{best_ms:.4f} ms" if tune else "autotune disabled"
            print(f"[sm121-fp8-mlp] {'fwd' if forward else 'bwd'} "
                  f"M={m} N={n} K={k} tile={chosen} ({timing})")
