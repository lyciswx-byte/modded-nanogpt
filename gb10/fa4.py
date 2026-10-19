"""SM121 FA4 adapter, not the H100's patched FA3 binary.

The GB10 does not have SM100 tcgen05/TMEM. Use the downstream FA4/CuTe
SM120 warp-mma path *only* after smoke testing this workload's unequal
QK/V widths, varlen metadata, sliding windows and backward gradients.
FA4 is opt-in via GB10_ATTN_BACKEND=fa4; baseline remains native SDPA.
"""
import functools
import inspect
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

from gb10.cute_compat import install_legacy_cute_annotations, cutlass_version


FA4_MODEL = "SecondNatureComputing/flash-attn-4-sm120"


@functools.lru_cache(maxsize=1)
def load_fa4():
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("SM121 FA4 adapter only supports the GB10 compute capability (12, 1)")
    # Third-party CuTe DSL annotations may still reference the pre-4.6
    # cute.core namespace. Alias only classes that are present in NVIDIA's
    # supported top-level cute API. No JIT code or kernels are rewritten.
    aliased = list(install_legacy_cute_annotations())
    # CuTe 4.8 renamed the old fragment constructor used by the SM120 bundle.
    # The old call creates register-memory storage, so this alias preserves
    # semantics while keeping the compatibility change scoped to opt-in FA4.
    try:
        import cutlass.cute as cute
        if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
            cute.make_fragment = cute.make_rmem_tensor
            aliased.append("make_fragment=make_rmem_tensor")
        # CUTLASS 4.8 renamed the NVVM atomic result keyword from `res` to
        # `results`; the supplied FA4 bundle still uses the former spelling.
        from cutlass._mlir.dialects import nvvm
        atomicrmw = nvvm.atomicrmw
        if not getattr(atomicrmw, "_gb10_compat", False):
            def atomicrmw_compat(*args, **kwargs):
                if "res" in kwargs and "results" not in kwargs:
                    kwargs["results"] = kwargs.pop("res")
                if "results" in kwargs and not isinstance(kwargs["results"], (tuple, list)):
                    kwargs["results"] = [kwargs["results"]]
                return atomicrmw(*args, **kwargs)
            atomicrmw_compat._gb10_compat = True
            nvvm.atomicrmw = atomicrmw_compat
            aliased.append("nvvm.atomicrmw(res=results)")
    except ImportError:
        pass
    if aliased:
        print(f"[GB10 FA4] CUTLASS {cutlass_version()}: "
              f"compatibility annotations {', '.join(aliased)}", flush=True)
    local_root = os.environ.get("GB10_FA4_LOCAL_PATH")
    if local_root:
        package_dir = os.path.join(local_root, "build", "torch-cuda")
        init_py = os.path.join(package_dir, "__init__.py")
        if not os.path.isfile(init_py):
            raise RuntimeError(f"GB10_FA4_LOCAL_PATH has no build/torch-cuda package: {local_root}")
        spec = importlib.util.spec_from_file_location(
            "flash_attn_4_sm120", init_py,
            submodule_search_locations=[package_dir],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create loader for local FA4: {init_py}")
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(spec.name)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            fn = module.flash_attn_varlen_func
        except Exception as exc:
            # A failed CuTe import must not poison a later corrected retry.
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
            raise RuntimeError(
                f"Local SM121 FA4 bundle failed to import from {init_py} "
                f"with nvidia-cutlass-dsl={cutlass_version()}; "
                "the original exception is chained."
            ) from exc
        return fn
    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise RuntimeError("FA4 requires 'kernels', nvidia-cutlass-dsl and apache-tvm-ffi; "
                           "see gb10/README.md") from exc
    # The SM120 bundle contains an alternative warp-MMA implementation,
    # not an SM100 tcgen05/TMEM binary or SM90 WGMMA.
    # kernels>=0.10 requires explicit trust for this third-party bundle.
    # Keep this opt-in path fail-closed; the repository and version are fixed
    # above and no fallback to the H100 FA3 loader is permitted.
    try:
        kernel = get_kernel(FA4_MODEL, version=1, trust_remote_code=True)
        fn = kernel.flash_attn_varlen_func
    except Exception as exc:
        raise RuntimeError(
            "SM121 FA4 bundle could not be loaded in this CUDA/ARM64 container. "
            "Check kernels/nvidia-cutlass-dsl/apache-tvm-ffi versions and the "
            "original chained exception; the FA4 backend remains opt-in."
        ) from exc
    try:
        sig = inspect.signature(fn)
        parameters = sig.parameters
        needed = {"cu_seqlens_q", "cu_seqlens_k", "window_size", "softmax_scale"}
        if not needed.issubset(parameters) and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            raise RuntimeError(f"FA4 missing required varlen/sliding-window API: {sig}")
    except (TypeError, ValueError):
        # A pybind interface may not publish a Python signature; the smoke
        # checker in test_fa4.py will call the actual function.
        pass
    return fn


def pad_mixed_qkv_for_fa4(q, k, v):
    """Preserve the original QK dot product when FA4 requires equal head dims.

    #92 uses (QK,V)=(64,128), (64,64) and (128,128). Padding Q/K with
    zeros from 64 to 128 leaves Q@K.T and its gradients mathematically
    unchanged, while making FA4's (Dq,Dk,Dv) shape uniform. Keep the
    ORIGINAL softmax scale: changing it to 1/sqrt(128) is incorrect.
    """
    if q.shape[:-1] != k.shape[:-1] or q.shape[:-1] != v.shape[:-1]:
        raise ValueError("mixed-head tensors must share [tokens, heads]")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K head dims differ")
    d_qk, d_v = q.shape[-1], v.shape[-1]
    if (d_qk, d_v) not in ((64, 64), (64, 128), (128, 128)):
        raise ValueError(f"Unsupported #92 mixed QK/V dims: {(d_qk,d_v)}")
    if d_qk == d_v:
        return q, k, v
    pad = d_v - d_qk
    return F.pad(q, (0, pad)), F.pad(k, (0, pad)), v


def packed_fa4(q, k, v, cum_seqlens, left_window, scale):
    """Varlen exact causal sliding-window FA4, with Dqk!=Dv preserved."""
    assert q.device.type == "cuda"
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[:2] == k.shape[:2] == v.shape[:2]
    assert q.shape[-1] in (64, 128) and v.shape[-1] in (64, 128)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    # The loader emits padded terminal endpoints. FA4 requires strictly
    # increasing offsets, so remove trailing duplicates on the CPU.
    cpu = cum_seqlens.cpu() if cum_seqlens.device.type != "cpu" else cum_seqlens
    # The validation loader may also begin with duplicate zero offsets
    # when the first token is BOS, not only trailing padding at T.
    ends = []
    for endpoint in cpu.tolist():
        if not ends or endpoint > ends[-1]:
            ends.append(endpoint)
    assert ends[0] == 0 and ends[-1] == q.shape[0], (ends[-1], q.shape[0])
    max_seqlen = max(b - a for a, b in zip(ends, ends[1:]))
    cu = torch.tensor(ends, dtype=torch.int32, device=q.device)
    fn = load_fa4()
    q_compatible, k_compatible, v_compatible = pad_mixed_qkv_for_fa4(q, k, v)
    # Recent CuTe FA4 bundles annotate the JIT window arguments as Int32.
    # Python ints work with older bundles but fail during SM121 compilation.
    # Keep the conversion local to the optional backend so the eager/graph
    # paths do not acquire a CUTLASS dependency.
    try:
        from cutlass.base_dsl.typing import Int32
        fa4_window = (Int32(int(left_window)), Int32(0))
    except (ImportError, TypeError, ValueError):
        fa4_window = (int(left_window), 0)
    try:
        y = fn(q_compatible.contiguous(), k_compatible.contiguous(), v_compatible.contiguous(),
           cu_seqlens_q=cu, cu_seqlens_k=cu,
           max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
               # Encode causal masking as a local window with right=0. Newer
               # bundles otherwise normalize causal=True back to Python int 0
               # before the CuTe JIT sees the argument.
               causal=False, softmax_scale=float(scale),
               window_size=fa4_window)
    except Exception as exc:
        raise RuntimeError(
            f"FA4 SM121 execution failed for QK={q.shape[-1]}, V={v.shape[-1]}, "
            f"tokens={q.shape[0]}, heads={q.shape[1]}, window={left_window}. "
            "Original exception retained; use GB10_ATTN_BACKEND=sdpa while diagnosing."
        ) from exc
    if isinstance(y, (tuple, list)):
        y = y[0]
    assert y.shape == v.shape, (y.shape, v.shape)
    return y


def dispatch(q, k, v, seqlens, window, scale):
    import os
    backend = os.environ.get("GB10_ATTN_BACKEND", "sdpa").lower()
    if backend == "graph":
        # GPU-resident dynamic document masks: same captured graph supports
        # different BOS boundaries at fixed token and cu_seqlens shapes.
        from gb10.graph_attention import graph_sdpa
        return graph_sdpa(q, k, v, seqlens, window, scale)
    if backend == "fa4":
        return packed_fa4(q, k, v, seqlens, window, scale)
    if backend == "fa4-safebwd":
        # Diagnostic only. Checks FA4 forward numerics, then recomputes a
        # trusted SDPA backward. Do NOT confuse this with a fixed FA4 kernel.
        from gb10.fa4_safebwd import forward_fa4_backward_sdpa
        return forward_fa4_backward_sdpa(q, k, v, seqlens, window, scale)
    if backend == "hybrid":
        # FA4's launch overhead is measurable for tiny segments on SM121.
        # The threshold must be selected from actual GB10 forward+backward
        # timings, not Hopper benchmarks. No automatic fallback on FA4 error.
        cpu = seqlens.cpu() if seqlens.device.type != "cpu" else seqlens
        raw = cpu.tolist()
        positive = [b-a for a,b in zip(raw, raw[1:]) if b>a]
        threshold = int(os.environ.get("GB10_FA4_MIN_SEQLEN", "384"))
        if positive and max(positive) >= threshold:
            return packed_fa4(q, k, v, seqlens, window, scale)
    if backend in ("sdpa", "hybrid"):
        from gb10.attention import segmented_sdpa
        return segmented_sdpa(q, k, v, seqlens, window, scale)
    raise ValueError("GB10_ATTN_BACKEND must be 'sdpa', 'fa4', 'fa4-safebwd', 'hybrid' or 'graph'")
