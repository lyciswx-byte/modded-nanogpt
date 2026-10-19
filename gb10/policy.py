"""Pure-Python contracts shared by the GB10 tuner and validation loader.

Keep these CPU-testable: nothing here imports torch, Triton or CUDA.
"""


def require_symmetric_tile(m: int, n: int, k: int) -> None:
    """Symmetric XXT/XTX/poly kernels write both C[m,n] and C[n,m].

    M/N rectangular tiles can cause diagonal blocks to be written by two
    programs at once with different masking. Never accept such a candidate.
    """
    if not all(isinstance(t, int) and t > 0 for t in (m, n, k)):
        raise ValueError("Triton tile dimensions must be positive integers")
    if m != n:
        raise ValueError("symmetric-mirror Triton kernels require square M/N tiles")
    if m & (m - 1) or k & (k - 1):
        raise ValueError("Triton tile sizes must be powers of two")


def validation_windows(count: int, window: int):
    """Bounded, exactly-once validation windows for both NCCL ranks.

    Rank count and the global validation configuration determine identical
    windows on each node; no rank-dependent early exit is allowed.
    """
    if count < 0 or not 1 <= window <= 8:
        raise ValueError("validation count must be >=0 and window in 1..8")
    for start in range(0, count, window):
        yield range(start, min(start + window, count))
