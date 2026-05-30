"""
CUDALinkBootstrap.py — Library-mode sys.path injector + bare-name alias registry.

TEXT DAT NAME: CUDALinkBootstrap

This module must be the **first import** in CUDAIPCExtension.py.  It runs before
any sibling Text DAT is imported by bare name, so the aliases it registers in
sys.modules are visible to all later imports:
  TDConfig     → Env
  TDSender     → ActivationBarrier, Exporter, NVTXShim, SHMProtocol
  TDReceiver   → CUDAIPCWrapper, CUDARuntimeTypes, NVTXShim, SHMProtocol
  (all others pulled in transitively by the installed package's own relative imports)

Two deployment modes
---------------------
Library mode (this module's purpose):
    Set CUDALINK_LIB_PATH to a folder created by:
        pip install --target <folder> "dist/cuda_link-<ver>-py3-none-any.whl"
    This module injects that folder onto sys.path, imports the cuda_link package,
    and registers all 14 mirror module names in sys.modules as aliases to the
    installed submodules.  The 14 mirror Text DATs (Env, SHMProtocol, Exporter, …)
    can then be removed from the COMP.

Fallback / classic mode:
    If CUDALINK_LIB_PATH is not set or cuda_link is not importable, this module
    no-ops and prints a notice.  All 14 mirror Text DATs must be present in the COMP
    as before (the original "paste all DATs" deployment story is fully preserved).

Drift guard:
    tests/test_td_bootstrap.py verifies that _ALIAS_MAP keys and values stay in sync
    with the PAIRS list in scripts/sync_td_wrapper.py.  If a new mirror module is added,
    update PAIRS first; the test will then fail here until this dict is updated too.
"""

from __future__ import annotations

import importlib
import os
import sys

# ---------------------------------------------------------------------------
# Alias map: bare PascalCase TD name  →  cuda_link submodule import path
#
# Must stay in sync with PAIRS in scripts/sync_td_wrapper.py.
# Key  = derived td_exporter stem (importable bare name inside TD's COMP namespace)
# Value = fully-qualified cuda_link submodule to alias it to
#
# tests/test_td_bootstrap.py::test_alias_map_covers_all_pairs enforces this.
# ---------------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    # byte_identical pairs (no relative imports in canonical source)
    # CUDARuntimeTypes MUST come before CUDAIPCWrapper: cuda_ipc_wrapper.py imports from
    # CUDARuntimeTypes at module load time, so the alias must be registered first to ensure
    # all ctypes argtypes use the same cudaIpcMemHandle_t class in all modes.
    "Env": "cuda_link._env",
    "FrameProfile": "cuda_link._profile",
    "CUDARuntimeTypes": "cuda_link.cuda_runtime_types",
    "CUDAIPCWrapper": "cuda_link.cuda_ipc_wrapper",
    "CUDAGraphs": "cuda_link.cuda_graphs",
    "NVMLObserver": "cuda_link.nvml_observer",
    "SHMProtocol": "cuda_link.shm_protocol",
    "ActivationBarrier": "cuda_link.activation_barrier",
    # rewrite_relative pairs (canonical source uses relative imports)
    "NVTXShim": "cuda_link._nvtx",
    "ExporterPort": "cuda_link._exporter_port",
    "ImporterPort": "cuda_link._importer_port",
    "CUDAAdapters": "cuda_link._cuda_adapters",
    "Exporter": "cuda_link.exporter",
    "Importer": "cuda_link.importer",
}


def _bootstrap() -> bool:
    """Inject sys.path and register bare-name aliases.  Returns True on success."""
    lib_path = os.environ.get("CUDALINK_LIB_PATH", "").strip()
    if lib_path and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    # Verify cuda_link is importable.  This also triggers the package __init__, which is
    # torch-safe: __init__.py re-exports torch/numpy/cupy only as guarded *_AVAILABLE flags.
    try:
        importlib.import_module("cuda_link")
    except ImportError:
        return False

    # Register each mirror name as an alias to the installed submodule.
    # Skip names already present (e.g. a sibling Text DAT loaded before us — unlikely but safe).
    failed: list[str] = []
    for bare_name, submodule_path in _ALIAS_MAP.items():
        if bare_name in sys.modules:
            continue
        try:
            sys.modules[bare_name] = importlib.import_module(submodule_path)
        except ImportError:
            failed.append(f"{bare_name} ({submodule_path})")

    if failed:
        # Partial bootstrap — warn without crashing.  Missing names fall through to
        # sibling Text DATs if present; absent Text DATs will raise ImportError later
        # at the glue-file level, which gives a clear error message.
        print(f"[CUDALinkBootstrap] WARNING: could not alias: {', '.join(failed)}")
        return False

    return True


# Run at Text DAT load time.
_active = _bootstrap()

if _active:
    print("[CUDALinkBootstrap] Library mode active — cuda_link submodules aliased as bare module names.")
else:
    print(
        "[CUDALinkBootstrap] Fallback mode — using sibling Text DAT mirrors. "
        "Set CUDALINK_LIB_PATH to enable library mode."
    )
