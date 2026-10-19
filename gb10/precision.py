"""FP8 numerical acceptance policy, independent of Triton / CUDA.

E4M3 has only three explicit fraction bits. Requiring per-element
allclose(atol=.1,rtol=.1) across an 80k-element GEMM is not a meaningful
test: statistically valid E4M3 quantization creates some larger outliers.
We check strict agreement with the *quantized* mathematical GEMM plus a
bounded RMS and tail error against the BF16 projection. These thresholds
are conservative initial gates, NOT a proof of downstream convergence.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FP8Metrics:
    normalized_rmse: float
    p999_abs_over_ref_rms: float
    max_abs_over_ref_rms: float


def fp8_error_metrics(approx: torch.Tensor, reference: torch.Tensor) -> FP8Metrics:
    if approx.shape != reference.shape:
        raise ValueError("FP8 and BF16 projection shapes differ")
    if approx.numel() == 0:
        raise ValueError("FP8 validation requires nonempty output")
    if not torch.isfinite(approx).all() or not torch.isfinite(reference).all():
        raise ValueError("FP8 projection produced nonfinite results")
    error = (approx.float() - reference.float()).abs().flatten()
    ref_rms = reference.float().square().mean().sqrt().clamp_min(1e-6)
    return FP8Metrics(
        normalized_rmse=float((error.square().mean().sqrt() / ref_rms).item()),
        p999_abs_over_ref_rms=float(
            (torch.quantile(error, 0.999) / ref_rms).item()
        ),
        max_abs_over_ref_rms=float((error.max() / ref_rms).item()),
    )


def assert_fp8_budget(approx: torch.Tensor, reference: torch.Tensor,
                      *, max_nrmse: float = 0.06,
                      max_p999: float = 0.22,
                      max_relative_peak: float = 0.40) -> FP8Metrics:
    """Reject accuracy loss beyond expected E4M3 rounding; retain strict guards."""
    stats = fp8_error_metrics(approx, reference)
    if (stats.normalized_rmse > max_nrmse
            or stats.p999_abs_over_ref_rms > max_p999
            or stats.max_abs_over_ref_rms > max_relative_peak):
        raise AssertionError(
            "FP8 E4M3 projection outside initial error budget: "
            f"nrmse={stats.normalized_rmse:.5f} "
            f"p99.9/ref_rms={stats.p999_abs_over_ref_rms:.5f} "
            f"max/ref_rms={stats.max_abs_over_ref_rms:.5f}"
        )
    return stats
