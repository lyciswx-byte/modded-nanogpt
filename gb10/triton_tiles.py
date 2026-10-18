"""SM121 pointwise FP8 and fused-CE Triton tile microbenchmark.

Must be executed OUTSIDE CUDA graphs once per worker, before full-step capture.
Searches block geometry on actual GB10 rather than copying H100 sizes. Errors
in a candidate reject it; no unsupported partial kernels are made default.
Chosen values are process-local and are frozen for the current training run.
"""
import os
from dataclasses import dataclass

import torch
import triton

# Cold defaults, used by correctness tests that have not prewarmed.
_QUANT = 1024
_CE = 1024
_FROZEN = False


def quant_block():
    return _QUANT


def ce_block():
    return _CE


def _time(call, reps=8):
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    first.record()
    for _ in range(reps):
        call()
    last.record()
    last.synchronize()
    return first.elapsed_time(last) / reps


def calibrate():
    global _QUANT, _CE, _FROZEN
    if _FROZEN:
        return (_QUANT, _CE)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SM121 tile calibration cannot run during CUDA graph capture")
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("SM121 calibration requires a GB10")
    # Lazily import after the runtime has finished creating its CUDA context.
    from gb10.fp8_linear import _quantize
    from gb10.fused_fp8_ce import _partials
    x = torch.randn((256 * 768,), device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([.001], device="cuda", dtype=torch.float32)
    q = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)
    for size in (256, 512, 1024, 2048):
        try:
            run = lambda: _quantize[(triton.cdiv(x.numel(),size),)](
                x, q, scale, x.numel(), size, num_warps=4,
            )
            run()
            torch.cuda.synchronize()
            # Quant kernel must agree with saturated reference to nearest FP8.
            expected = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            if not torch.equal(q.float(), expected.float()):
                print(f"[SM121] reject quant tile {size}: numerical mismatch",flush=True)
                continue
            latency = _time(run)
            if "_quant_perf" not in locals() or latency < _quant_perf:
                _quant_perf = latency
                _QUANT = size
        except Exception as exc:
            print(f"[SM121] quant tile {size} failed: {exc}",flush=True)
    if "_quant_perf" not in locals():
        raise RuntimeError("No valid SM121 FP8 quant Tile")
    rows, vocab = 64, 14336
    qlogits = torch.zeros((rows, vocab), dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.tensor([.01], device="cuda", dtype=torch.float32)
    best_ce = None
    for size in (256, 512, 1024, 2048):
        chunks = triton.cdiv(vocab, size)
        part = torch.empty((rows, chunks), device="cuda", dtype=torch.float32)
        try:
            def run():
                _partials[(rows, chunks)](qlogits, scale, part,
                                         vocab, chunks, size, num_warps=4)
            run()
            torch.cuda.synchronize()
            expected = torch.exp(
                23 * torch.sigmoid((qlogits.float() * scale + 5)/7.5) - 23
            ).sum(-1)
            if not torch.allclose(part.sum(-1), expected, rtol=.002, atol=.01):
                print(f"[SM121] reject CE tile {size}: numerical mismatch",flush=True)
                continue
            latency = _time(run)
            if best_ce is None or latency < best_ce[0]:
                best_ce = (latency, size)
        except Exception as exc:
            print(f"[SM121] CE tile {size} failed: {exc}",flush=True)
    if best_ce is None:
        raise RuntimeError("No valid SM121 CE tile")
    _CE = best_ce[1]
    _FROZEN = True
    print(f"[SM121 tiles] FP8 quant={_QUANT} {_quant_perf*1000:.1f}us, "
          f"fused CE={_CE} {best_ce[0]*1000:.1f}us; frozen before graph capture",
          flush=True)
    return (_QUANT, _CE)
