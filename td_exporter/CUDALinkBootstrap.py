"""
CUDALinkBootstrap.py — Project-anchored cuda_link resolver + bare-name alias registry.

TEXT DAT NAME: CUDALinkBootstrap

This module must be the **first import** in CUDAIPCExtension.py.  It runs before
any sibling Text DAT is imported by bare name, so the aliases it registers in
sys.modules are visible to all later imports:
  TDConfig     → Env
  TDSender     → ActivationBarrier, Exporter, NVTXShim, SHMProtocol
  TDReceiver   → CUDAIPCWrapper, CUDARuntimeTypes, NVTXShim, SHMProtocol
  (all others pulled in transitively by the installed package's own relative imports)

Resolution is layered (ADR-0014).  The first layer that yields a cuda_link whose
``__version__`` equals MIRROR_VERSION wins.  A mismatching install is skipped WITHOUT
being imported, and a cuda_link that is already loaded from somewhere else -- or an
import hook that redirects the name elsewhere -- aborts resolution outright instead of
silently mixing two installations:
  (a) an explicitly supplied folder             _bootstrap(basefolder=...)
  (b) the ``Libpath`` custom parameter           on this COMP or any ancestor
  (c) <project.folder>/cuda_link, /StreamDiffusion, then <project.folder> itself
  (d) CUDALINK_LIB_PATH                          (ADR-0003 compatibility)
  (e) whatever sys.path already provides         (TD Preferences module path, pip)
Each root is probed as <root>/venv/Lib/site-packages, <root>/.venv/Lib/site-packages,
<root>/Lib/site-packages (the root is itself a venv) and <root> itself (a
``pip install --target`` folder).

Two deployment modes
---------------------
Library mode (this module's purpose):
    Install cuda_link once (install_td_library.cmd, or
    ``pip install --target <folder> dist/cuda_link-<ver>-py3-none-any.whl``) and point
    the COMP's ``Libpath`` parameter -- or CUDALINK_LIB_PATH -- at that folder.  The
    resolved package is imported and all 15 mirror module names are registered in
    sys.modules as aliases to its submodules.  The 15 mirror Text DATs (Env,
    SHMProtocol, Exporter, …) can then be removed from the COMP.

Fallback / classic mode:
    If no layer resolves, this module no-ops, records why in ``last_error`` (the
    extension surfaces it as a yellow status), and all 15 mirror Text DATs must be
    present in the COMP as before (the original "paste all DATs" deployment story).

Drift guards:
    tests/td/test_td_bootstrap.py verifies that _ALIAS_MAP keys and values stay in sync
    with PAIRS in scripts/sync_td_wrapper.py, and that MIRROR_VERSION equals
    cuda_link.__version__.  The sync script rewrites the stamp -- never edit it by hand.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import re
import sys
from collections.abc import Iterator

# Stamped by scripts/sync_td_wrapper.py from src/cuda_link/__init__.py::__version__.
# The bootstrap only activates an install whose __version__ equals this value, so the
# mirrors shipped inside the .tox and the package they alias can never drift apart.
MIRROR_VERSION = "1.12.2"

logger = logging.getLogger("cuda_link.td.bootstrap")

# ---------------------------------------------------------------------------
# Alias map: bare PascalCase TD name  →  cuda_link submodule import path
#
# Must stay in sync with PAIRS in scripts/sync_td_wrapper.py.
# Key  = derived td_exporter stem (importable bare name inside TD's COMP namespace)
# Value = fully-qualified cuda_link submodule to alias it to
#
# tests/td/test_td_bootstrap.py::test_alias_map_covers_all_pairs enforces this.
# Listed in dependency order (Env first, CUDARuntimeTypes before CUDAIPCWrapper).  Since
# ADR-0002's relative-import rule every mirrored module imports its siblings relatively,
# so the order is belt-and-braces rather than load-bearing.
# ---------------------------------------------------------------------------
_ALIAS_MAP: dict[str, str] = {
    "Env": "cuda_link._env",
    "FrameProfile": "cuda_link._profile",
    "CUDARuntimeTypes": "cuda_link.cuda_runtime_types",
    "CUDAIPCWrapper": "cuda_link.cuda_ipc_wrapper",
    "CUDAGraphs": "cuda_link.cuda_graphs",
    "NVMLObserver": "cuda_link.nvml_observer",
    "SHMProtocol": "cuda_link.shm_protocol",
    "ActivationBarrier": "cuda_link.activation_barrier",
    "Doorbell": "cuda_link._doorbell",
    "NVTXShim": "cuda_link._nvtx",
    "ExporterPort": "cuda_link._exporter_port",
    "ImporterPort": "cuda_link._importer_port",
    "CUDAAdapters": "cuda_link._cuda_adapters",
    "Exporter": "cuda_link.exporter",
    "Importer": "cuda_link.importer",
}

# Public state read by CUDAIPCExtension (and by StreamDiffusionTD, which shares this shape).
last_error = ""
resolved_site_packages = ""  # diagnostics: which folder won
_active = False

_VERSION_RE = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)


class _RivalInstallError(RuntimeError):
    """A different cuda_link is already loaded in this process.

    Must abort resolution immediately rather than fall through to the next layer --
    falling through risks silently accepting whichever install happens to already be
    loaded even though the caller asked for a specific, different one.
    """


# --------------------------------------------------------------------------- paths


def _normalized(path: object) -> str:
    try:
        return os.path.normcase(os.path.realpath(str(path)))
    except (OSError, ValueError, TypeError):
        return ""


def _within(child: object, parent: object) -> bool:
    """True when child is parent or lives under it.  Never raises.

    os.path.commonpath() raises ValueError for paths on different drives; string
    comparison on normalised paths does not.  Equality returns True, so the guard
    cannot fire on the already-loaded path.
    """
    child_n, parent_n = _normalized(child), _normalized(parent)
    if not child_n or not parent_n:
        return False
    return child_n == parent_n or child_n.startswith(parent_n + os.sep)


def _expand(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    expander = globals().get("tdu")  # TD-only: expands $VAR and project-relative paths
    if expander is not None:
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            text = expander.expandPath(text)
    return os.path.expandvars(text)


def _package_dir(module: object) -> str:
    path = getattr(module, "__file__", "") or ""
    return os.path.dirname(path) if path else ""


def _read_version(site_packages: str) -> str:
    """cuda_link.__version__ read from disk -- the candidate install is NOT imported."""
    init = os.path.join(site_packages, "cuda_link", "__init__.py")
    try:
        with open(init, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return ""
    match = _VERSION_RE.search(text)
    return match.group(1) if match else ""


# --------------------------------------------------------------------------- layers


def _libpath_parameter() -> str:
    """``Libpath`` custom parameter on the owning COMP or any ancestor (TD only)."""
    dat = globals().get("me")
    component = dat.parent() if dat is not None else None
    while component is not None:
        parameter = getattr(component.par, "Libpath", None)
        if parameter is not None:
            return str(parameter.eval()).strip()
        component = component.parent()
    return ""


def _project_roots() -> Iterator[str]:
    """Folders next to the .toe that conventionally hold a per-project install."""
    proj = globals().get("project")
    folder = str(getattr(proj, "folder", "") or "") if proj is not None else ""
    if not folder:
        return
    yield os.path.join(folder, "cuda_link")
    yield os.path.join(folder, "StreamDiffusion")  # StreamDiffusionTD's venv lives here
    yield folder


def _env_libpath() -> str:
    return os.environ.get("CUDALINK_LIB_PATH", "").strip()


def _sys_path_root() -> str:
    """Folder holding the cuda_link that a plain ``import cuda_link`` would pick up."""
    try:
        spec = importlib.util.find_spec("cuda_link")
    except (ImportError, ValueError):
        return ""
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        return ""
    return os.path.dirname(os.path.dirname(origin))


def _layers(basefolder: str | None) -> Iterator[tuple[str, str, bool]]:
    """(label, root, inject-on-sys.path) -- lazy, so later layers (operator walk,
    find_spec) never run once an earlier layer has already resolved."""
    yield "folder argument", basefolder or "", True
    yield "Libpath parameter", _libpath_parameter(), True
    for root in _project_roots():
        yield "project folder", root, True
    yield "CUDALINK_LIB_PATH", _env_libpath(), True
    yield "sys.path", _sys_path_root(), False


def _site_package_candidates(root: str) -> list[str]:
    """A project holding a venv, a venv root, and a pre-resolved site-packages / --target dir.

    ``Lib/site-packages`` is the Windows venv layout, the only one TouchDesigner runs on.
    """
    expanded = _expand(root)
    if not expanded:
        return []
    return [
        os.path.join(expanded, "venv", "Lib", "site-packages"),
        os.path.join(expanded, ".venv", "Lib", "site-packages"),
        os.path.join(expanded, "Lib", "site-packages"),
        expanded,
    ]


def _has_package(site_packages: str) -> bool:
    return os.path.isfile(os.path.join(site_packages, "cuda_link", "__init__.py"))


# --------------------------------------------------------------------------- activate


def _check_origin(module: object, package_dir: str, *, fresh: bool = False) -> object:
    """Return *module* if it may coexist with the install at *package_dir*, else raise.

    *fresh* marks a module this call has just imported: a wrong origin then means an
    import hook ahead of sys.path (an editable install's redirecting finder, for
    example) is serving the name -- nothing was loaded before we asked.
    """
    origin = _package_dir(module)
    if not origin:
        return module  # TD Text DAT module (no file on disk) -- not a rival install
    if _within(origin, package_dir):
        return module  # same installation (equality included) -- never fires spuriously
    if "cuda_link" not in _normalized(origin).split(os.sep):
        return module  # unrelated module owning the bare name (classic mirror DAT)
    if fresh:
        raise _RivalInstallError(
            f"importing cuda_link resolved to {origin} instead of the selected installation "
            f"{package_dir}; an import hook ahead of sys.path (for example an editable "
            f"'pip install -e' of cuda-link) is redirecting it. Remove that install or point "
            f"the component at it."
        )
    raise _RivalInstallError(
        f"CUDA-Link is already loaded from {origin}; the selected installation is "
        f"{package_dir}. Restart TouchDesigner to switch installations."
    )


def _activate(site_packages: str, *, inject: bool = True) -> bool:
    """Import cuda_link from *site_packages* and register the bare-name aliases."""
    global last_error, _active, resolved_site_packages
    package_dir = os.path.join(site_packages, "cuda_link")
    before = set(sys.modules)

    loaded = sys.modules.get("cuda_link")
    if loaded is not None:
        _check_origin(loaded, package_dir)
    for name in _ALIAS_MAP:  # preflight before mutating anything
        if name in sys.modules:
            _check_origin(sys.modules[name], package_dir)

    if inject:
        if site_packages in sys.path:
            sys.path.remove(site_packages)
        sys.path.insert(0, site_packages)
    try:
        # Triggers the package __init__, which is torch-safe: it re-exports torch/numpy/
        # cupy only as guarded *_AVAILABLE flags.
        _check_origin(importlib.import_module("cuda_link"), package_dir, fresh=True)
        for name, target in _ALIAS_MAP.items():
            if name in sys.modules:
                # Already owned by a sibling Text DAT loaded before us (preflight above
                # proved it is not a rival package copy) -- leave it; never swap a module
                # other code may already hold references into.
                continue
            sys.modules[name] = _check_origin(importlib.import_module(target), package_dir, fresh=True)  # type: ignore[assignment]
    except BaseException:
        if inject and site_packages in sys.path:  # do not leave a dead path entry behind
            sys.path.remove(site_packages)
        for key in list(sys.modules):  # drop only what this call imported
            if key not in before and (key == "cuda_link" or key.startswith("cuda_link.") or key in _ALIAS_MAP):
                del sys.modules[key]
        raise

    last_error = ""
    resolved_site_packages = site_packages
    _active = True
    return True


def _abort(error: BaseException) -> bool:
    """A rival install was detected -- stop resolving, do not try later layers."""
    global last_error, _active, resolved_site_packages
    last_error = str(error)
    resolved_site_packages = ""
    _active = False
    logger.warning("%s", last_error)
    return False


def _bootstrap(basefolder: str | None = None) -> bool:
    """Resolve, version-check, import and alias cuda_link.  Returns True on success.

    On failure ``last_error`` names every layer that was tried and why it was skipped;
    the extension shows it as a yellow status and the COMP falls back to its mirrors.
    """
    global last_error, _active, resolved_site_packages
    notes: list[str] = []

    for label, root, inject in _layers(basefolder):
        if not str(root).strip():
            notes.append(f"{label}: not set")  # quiet deferral, not a warning
            continue
        candidates = [p for p in _site_package_candidates(root) if _has_package(p)]
        if not candidates:
            notes.append(f"{label}: no cuda_link under {_expand(root)}")
            continue
        for site_packages in candidates:
            found = _read_version(site_packages)
            if found != MIRROR_VERSION:
                notes.append(
                    f"{label}: cuda_link {found or '?'} under {site_packages} "
                    f"does not match the component's {MIRROR_VERSION}"
                )
                continue
            try:
                return _activate(site_packages, inject=inject)
            except _RivalInstallError as error:
                return _abort(error)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                notes.append(f"{label}: {error}")

    last_error = "CUDA-Link could not be resolved -- " + "; ".join(notes)
    resolved_site_packages = ""
    _active = False
    return False


# Run at Text DAT load time.
_bootstrap()

if _active:
    print(f"[CUDALinkBootstrap] Library mode active — cuda_link {MIRROR_VERSION} from {resolved_site_packages}")
else:
    print(f"[CUDALinkBootstrap] Fallback mode — using sibling Text DAT mirrors. {last_error}")
