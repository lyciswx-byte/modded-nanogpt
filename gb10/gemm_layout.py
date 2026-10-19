"""CPU-testable SM121 FP8 cuBLASLt operand layout and shape normalization.

No GPU or Triton import in this module. FP8 matmul is tile-aligned: pad the
flattened leading dimension with zeros, but trim outputs before returning.
Column-major second operand is a transposed row-major W[out,in] view.
"""
import torch
import torch.nn.functional as F


def prepare_fp8_gemm_input(x: torch.Tensor, alignment: int = 16):
    """Return zero-padded [M,K], original M; preserve PyTorch autograd."""
    if x.ndim != 2:
        raise ValueError("FP8 GEMM input must be [M,K]")
    if alignment < 1 or (alignment & (alignment - 1)):
        raise ValueError("alignment must be a positive power of two")
    rows = x.shape[0]
    padded_rows = (rows + alignment - 1) // alignment * alignment
    return F.pad(x, (0, 0, 0, padded_rows - rows)), rows


def column_major_rhs(quantized_weight: torch.Tensor):
    """Return B[in,out] with column-major stride instead of forcing row-major."""
    if quantized_weight.ndim != 2:
        raise ValueError("weight must be rank 2")
    return quantized_weight.transpose(0, 1)
