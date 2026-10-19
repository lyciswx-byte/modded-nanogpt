"""Compatibility for CuTe DSL annotation API migration (CUTLASS >=4.6).

Some externally bundled FA4 code still refers to cutlass.cute.core.ThrMma,
ThrCopy, etc. The maintained CUTLASS API exports these types from cutlass.cute.
Expose the *same type objects* under the legacy namespace ONLY when the user
has explicitly selected FA4. This repairs evaluated Python annotations; it
does not modify PTX, JIT strategy, memory layouts, or kernel semantics.

A subsequent import/JIT failure remains fatal and includes its original
exception; this shim is not a compatibility guarantee for an entire FA4 fork.
"""
import importlib
import importlib.metadata
import os


# Only API names that were moved from cute.core to cute. Do not invent classes.
_MOVED = ("ThrMma", "TiledMma", "ThrCopy", "TiledCopy")


def cutlass_version():
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def install_legacy_cute_annotations():
    """Return names aliased into legacy core; refuse missing required ThrMma.

    Applied once at opt-in FA4 load time, before importing the external
    module. Aliases point at NVIDIA's real current classes, not placeholders.
    """
    if os.environ.get("GB10_FA4_CUTE_COMPAT", "1") != "1":
        return ()
    try:
        cute = importlib.import_module("cutlass.cute")
        core = importlib.import_module("cutlass.cute.core")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "FA4 requires a working nvidia-cutlass-dsl installation; "
            f"detected version={cutlass_version()}"
        ) from exc
    changed = []
    for name in _MOVED:
        if hasattr(core, name):
            continue
        replacement = getattr(cute, name, None)
        if replacement is not None:
            setattr(core, name, replacement)
            changed.append(name)
    if not hasattr(core, "ThrMma"):
        raise RuntimeError(
            "FA4 legacy cute.core.ThrMma cannot be mapped to "
            f"cutlass.cute.ThrMma (CUTLASS DSL {cutlass_version()}). "
            "Inspect installed package version; do not stub it with a fake type."
        )
    return tuple(changed)
