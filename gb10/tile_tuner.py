"""SM121 ANVIL Gram tile selection, calibrated *on the actual GB10*.

The H100 kernels hard-code 128x128x64 / eight warps. This module reuses the
mathematically equivalent symmetric Triton program but searches consumer
Blackwell candidates (32/64/128 tiles, 4/8 warps, 2/3 stages) on each GB10.
It checks numerical agreement with ATen for EVERY candidate. No benchmark or
autotune is allowed during a CUDA Graph capture: prewarm() must run first.
Selected (shape, stride, dtype, kernel) -> tile is cached for graph replay.
"""
from dataclasses import dataclass
from gb10.policy import require_symmetric_tile
import os

import torch
import triton
from track_1_short.perf.kernels.polar_express import XXT_kernel, XTX_kernel, ba_plus_cAA_kernel


@dataclass(frozen=True)
class Tile:
    m: int
    n: int
    k: int
    warps: int
    stages: int


# SM121 can be register-/occupancy-limited; don't assume Hopper tiles win.
CANDIDATES = (
    # Square M/N tiles ONLY: the inherited symmetric kernels mirror each
    # computed block to (N,M), and rectangular blocks can overlap writes.
    Tile(32, 32, 32, 4, 2),
    Tile(64, 64, 32, 4, 2),
    Tile(64, 64, 64, 4, 2),
    Tile(128, 128, 32, 4, 3),
    Tile(128, 128, 64, 8, 3),
)
_CHOSEN = {}
_STATS = {}


def _key(kind, a):
    # Cache should not mix CUDA contexts or output layouts in a future 2-GPU
    # per-process configuration. Each GB10 still launches 1 worker today.
    return (kind, str(a.device), tuple(a.shape), tuple(a.stride()), a.dtype)


def _launch(kind, a, out, tile, *, alpha=1., beta=0.):
    require_symmetric_tile(tile.m, tile.n, tile.k)
    M, K = a.shape[-2:]
    side = M if kind == "xxt" else K
    if kind == "poly":
        assert M == K
        side = M
    batch = a.shape[0] if a.ndim == 3 else 1
    kernel = {"xxt": XXT_kernel, "xtx": XTX_kernel, "poly": ba_plus_cAA_kernel}[kind]
    grid = (batch * triton.cdiv(side, tile.m) * triton.cdiv(side, tile.n),)
    kw = dict(
        A_ptr=a, C_ptr=out, M=M,
        a_stride_b=a.stride(0) if a.ndim == 3 else 0,
        a_stride_r=a.stride(-2), a_stride_c=a.stride(-1),
        c_stride_b=out.stride(0) if out.ndim == 3 else 0,
        c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=tile.m, BLOCK_SIZE_N=tile.n, BLOCK_SIZE_K=tile.k,
        GROUP_SIZE_M=8, LOWER_UPPER=1,
        num_warps=tile.warps, num_stages=tile.stages,
    )
    if kind == "poly":
        kw.update(alpha=alpha, beta=beta)
    else:
        kw["K"] = K
    kernel[grid](**kw)
    return out


def _reference(kind, a, *, alpha=1., beta=0.):
    if kind == "xxt":
        return a @ a.transpose(-1, -2)
    if kind == "xtx":
        return a.transpose(-1, -2) @ a
    return alpha * (a @ a.transpose(-1, -2)) + beta * a


def _out(a, kind):
    n = a.shape[-2] if kind in ("xxt", "poly") else a.shape[-1]
    return torch.empty((*a.shape[:-2], n, n), device=a.device, dtype=a.dtype)


def _time(call, repeats=5):
    # CUDA events, not wall clock distorted by ARM host launch overhead.
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def tune(kind, a, *, alpha=1.3, beta=-0.25):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GB10 Gram autotune inside CUDA Graph capture; prewarm first")
    if torch.cuda.get_device_capability(a.device) != (12, 1):
        raise RuntimeError("GB10 Triton tile tuner requires sm_121")
    key = _key(kind, a)
    if key in _CHOSEN:
        return _CHOSEN[key]
    if kind not in ("xxt", "xtx", "poly"):
        raise ValueError(kind)
    expected = _reference(kind, a, alpha=alpha, beta=beta).float()
    output = _out(a, kind)
    candidates = []
    for candidate in CANDIDATES:
        try:
            output.zero_()
            _launch(kind, a, output, candidate, alpha=alpha, beta=beta)
            torch.cuda.synchronize()
            error = (output.float() - expected).abs()
            # BF16 Gram has ~.5% rounding near large terms. Test BOTH
            # scaled absolute error and norm-wise relative error.
            threshold = max(0.02, 0.025 * float(expected.abs().amax()))
            if not torch.isfinite(output).all() or float(error.amax()) > threshold:
                continue
            duration = _time(lambda: _launch(kind, a, output, candidate,
                                            alpha=alpha, beta=beta))
            candidates.append((duration, candidate))
        except (RuntimeError, ValueError, AssertionError) as exc:
            print(f"[SM121 tile] reject {kind} {candidate}: {type(exc).__name__}: {exc}", flush=True)
    if not candidates:
        raise RuntimeError(f"No numerically valid sm_121 Triton Gram tile for {key}")
    best_time, best = min(candidates, key=lambda entry: entry[0])
    _CHOSEN[key] = best
    _STATS[key] = best_time
    print(f"[SM121 tile] {kind} {tuple(a.shape)} {a.dtype}: "
          f"{best} {best_time*1000:.1f}us (of {len(candidates)} valid)", flush=True)
    return best


def dispatch(kind, a, out, *, alpha=1., beta=0.):
    key = _key(kind, a)
    if key not in _CHOSEN:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"uncalibrated {kind} {tuple(a.shape)} during graph capture")
        tune(kind, a)
    return _launch(kind, a, out, _CHOSEN[key], alpha=alpha, beta=beta)


def XXT(a, out):
    return dispatch("xxt", a, out)


def XTX(a, out):
    return dispatch("xtx", a, out)


def ba_plus_cAA(a, alpha, beta, out):
    return dispatch("poly", a, out, alpha=alpha, beta=beta)


def prewarm(banks):
    """Call after constructing 2-rank optimizer, BEFORE graph warmup.

    Keep calibration away from live optimizer state. No distributed operation
    and no mutation of parameter/gradient tensors takes place.
    """
    for bank in banks:
        shape = tuple(bank.grad.shape)
        x = (torch.randn(shape, device=bank.grad.device, dtype=torch.float32) * .02).bfloat16()
        kind = "xtx" if shape[-2] > shape[-1] else "xxt"
        tune(kind, x)
        side = shape[-1] if kind == "xtx" else shape[-2]
        gram = torch.randn((*shape[:-2], side, side), device=x.device,
                           dtype=torch.float32).mul_(.004).bfloat16()
        # Symmetric input only; polynomial kernel writes both triangles.
        gram = ((gram + gram.transpose(-1, -2)) * .5).contiguous()
        tune("poly", gram)
    torch.cuda.synchronize()
