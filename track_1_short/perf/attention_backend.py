"""Architecture-selectable variable-length attention for record #92.

FA3's prebuilt sm90a binary must never be loaded on GB10 (sm121). FA2 is the
conservative GB10 default; FA4 is explicitly opt-in because the published
Blackwell CuTeDSL support does not imply compatibility with every SM121 kernel.
All backend imports are lazy, so CPU configuration tests need no GPU extension.

The original model has Q/K head widths 64 or 128 and V widths 64 or 128.
Pad to a common width for portable backends, retaining the explicit softmax
scale, then slice V's output back to the original width. Padding Q/K with zeros
does not change QK logits; padding V adds only zero output channels.
"""
import hashlib
import os
import pathlib

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

FA3_REPO = "devenpzak/flash-attn3-12864"
FA3_REVISION = "64c1e6d1f2780e7931839f41426ddcdb564a7cb9"
FA3_BINARY_SHA256 = "9beee90dbef9c9c221e677dad9f932108b70c46aa35841fc0862281d5b84c09b"

BACKENDS = ("flex", "fa2", "fa3", "fa4")


def configured_backend() -> str:
    default = "flex" if os.environ.get("NANOGPT_TARGET") == "gb10" else "fa3"
    name = os.environ.get("NANOGPT_ATTENTION_BACKEND", default).lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown attention backend {name!r}; choose from {BACKENDS}")
    return name


def load_attention_kernel(name: str | None = None):
    """Return the backend's varlen function, without touching the GPU."""
    name = configured_backend() if name is None else name
    if name == "flex":
        # Native PyTorch Triton backend; avoids SM90A binary and third-party ABI.
        return None
    if name == "fa3":
        if os.environ.get("NANOGPT_TARGET") == "gb10":
            raise RuntimeError("The pinned FA3 binary targets H100/sm90a, not GB10/sm121. Use fa2 or opt-in fa4.")
        from kernels import get_kernel
        fa3 = get_kernel(FA3_REPO, revision=FA3_REVISION, trust_remote_code=True)
        binary = next(pathlib.Path(fa3.__file__).parent.glob("_flash_attn3_cuda_*.so"))
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        assert digest == FA3_BINARY_SHA256, f"unexpected FA3 binary {binary}"
        return fa3.flash_attn_interface.flash_attn_varlen_func
    try:
        if name == "fa2":
            from flash_attn import flash_attn_varlen_func
            return flash_attn_varlen_func
        if name == "fa4":
            from flash_attn.cute import flash_attn_varlen_func
            return flash_attn_varlen_func
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Cannot import {name} on this platform. Use the GB10 Dockerfile and "
            "run tools/gb10_kernel_probe.py before training."
        ) from exc
    raise ValueError(f"Unknown attention backend: {name}")


# Import only once, outside forward/torch.compile; importantly never import the FA3
# H100-specific extension in GB10 mode.
_VARLEN_KERNEL = None


def prime_attention_backend():
    """Resolve backend before torch.compile or CUDA-graph capture."""
    global _VARLEN_KERNEL
    if _VARLEN_KERNEL is None:
        _VARLEN_KERNEL = load_attention_kernel()
    return configured_backend()


def _flex_varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           seqlens: torch.Tensor, scale: float, left_window: int) -> torch.Tensor:
    """Native torch.compile/FlexAttention fallback with per-document causal windows.

    Builds a token-to-document map from the *live* CUDA cu_seqlens. This is
    correct for packed batches and has a real autograd backward, but building
    block masks can be expensive at larger microbatch sizes. It is the portable
    bring-up path for SM121 rather than a claim of H100-level performance.
    """
    tokens = q.shape[0]
    positions = torch.arange(tokens, device=q.device, dtype=seqlens.dtype)
    # right=True assigns a boundary token to the sequence beginning there.
    documents = torch.searchsorted(seqlens[1:], positions, right=True)
    offsets = positions - seqlens[documents]
    # Flex block masks evaluate some padded q/kv indices; guard the indexing.
    block = 128
    padding = (-tokens) % block
    documents = F.pad(documents, (0, padding), value=-1)
    offsets = F.pad(offsets, (0, padding), value=0)

    def mask_mod(batch, head, q_idx, kv_idx):
        return ((documents[q_idx] == documents[kv_idx])
                & (documents[q_idx] >= 0)
                & (offsets[kv_idx] <= offsets[q_idx])
                & (offsets[kv_idx] >= offsets[q_idx] - left_window))

    block_mask = create_block_mask(mask_mod, B=None, H=None,
                                   Q_LEN=tokens, KV_LEN=tokens,
                                   device=q.device, BLOCK_SIZE=block)
    out = flex_attention(q.transpose(0, 1).unsqueeze(0),
                         k.transpose(0, 1).unsqueeze(0),
                         v.transpose(0, 1).unsqueeze(0),
                         block_mask=block_mask, scale=scale)
    return out[0].transpose(0, 1).contiguous()


def varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     seqlens: torch.Tensor, max_len: int, scale: float,
                     left_window: int) -> torch.Tensor:
    """[T,H,Dqk] x 2 and [T,H,Dv] -> [T,H,Dv], packed causal local attention."""
    global _VARLEN_KERNEL
    backend = configured_backend()
    if backend != "flex" and _VARLEN_KERNEL is None:
        _VARLEN_KERNEL = load_attention_kernel()

    value_dim = v.shape[-1]
    qk_dim = q.shape[-1]
    assert k.shape[-1] == qk_dim
    if backend != "fa3":
        # FA2/FA4 builds may reject unequal Q/K and V head dimensions. Padding
        # preserves attention math (including scaling) and gradients.
        padded_dim = max(qk_dim, value_dim)
        if padded_dim > qk_dim:
            q = F.pad(q, (0, padded_dim - qk_dim))
            k = F.pad(k, (0, padded_dim - qk_dim))
        if padded_dim > value_dim:
            v = F.pad(v, (0, padded_dim - value_dim))
    if backend == "flex":
        y = _flex_varlen_attention(q, k, v, seqlens, scale, left_window)
    else:
        y = _VARLEN_KERNEL(
            q.contiguous(), k.contiguous(), v.contiguous(),
            cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
            max_seqlen_q=max_len, max_seqlen_k=max_len, causal=True,
            softmax_scale=scale, window_size=(left_window, 0),
        )
    # The original FA3 extension supports mixed head widths directly.
    return y[..., :value_dim] if y.shape[-1] != value_dim else y
