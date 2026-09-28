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
being imported.  A cuda_link that is already loaded from somewhere else, an import hook
that redirects the name elsewhere, or a sibling mirror Text DAT imported ahead of this
module all abort resolution outright instead of silently mixing two copies:
  (a) an explicitly supplied folder             _bootstrap(basefolder=...)
  (b) <project.folder>/cuda_link, /StreamDiffusion, /src, then <project.folder> itself
  (c) CUDALINK_LIB_PATH                          (ADR-0003 compatibility)
  (d) whatever sys.path already provides         (TD Preferences module path, pip)
Each root is probed as <root>/venv/Lib/site-packages, <root>/.venv/Lib/site-packages,
<root>/Lib/site-packages (the root is itself a venv) and <root> itself (a
``pip install --target`` folder). For the two manually supplied roots -- (a) and (c) --
a root whose basename is itself ``cuda_link`` is also probed one level up, so pointing
CUDALINK_LIB_PATH at the package folder (rather than its parent) still resolves; the
synthetic (b) roots are exempt; forgiving ``<project.folder>/cuda_link`` the same way
would make it swallow <project.folder> itself, the fallback root tried after
/StreamDiffusion and /src.

Two deployment modes
---------------------
Library mode (this module's purpose):
    Install cuda_link once (install_td_library.cmd, or
    ``pip install --target <folder> dist/cuda_link-<ver>-py3-none-any.whl``) next to the
    .toe, into a venv the project layer probes, or into a Python already on TD's
    sys.path -- see the layer list above; there is no per-COMP parameter to set.  The
    resolved package is imported and all 15 mirror module names are registered in
    sys.modules as aliases to its submodules.  The 15 mirror Text DATs (Env,
    SHMProtocol, Exporter, …) can then be removed from the COMP.

Fallback / classic mode:
    If no layer resolves, this module no-ops and records why in ``last_error``. The COMP
    then falls back to its mirror Text DATs. When none of those mirrors are present as COMP
    siblings, the extension additionally shows a short, actionable line as the COMP's status
    and prints the full ``last_error`` to the Textport, since that shape means library mode
    failed AND classic mode has nothing to fall back to. When the mirrors ARE present (the
    original "paste all DATs" deployment story), the fallback is silent -- no yellow tint,
    no Status line -- because classic mode is working as intended.

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
from typing import Any

# Stamped by scripts/sync_td_wrapper.py from src/cuda_link/__init__.py::__version__.
# The bootstrap only activates an install whose __version__ equals this value, so the
# mirrors shipped inside the .tox and the package they alias can never drift apart.
MIRROR_VERSION = "1.13.0"

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
# Specific reason the last _bootstrap() call gave up, so the extension can show a
# cause-specific Status line instead of always saying "not found": "missing" (no
# candidate anywhere), "mismatch" (a candidate exists but its version doesn't match),
# "rival" (a different cuda_link is already loaded), "broken" (a matching-version
# candidate was found but raised while importing/activating), or "" (active).
failure_kind = ""

_VERSION_RE = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)


class _RivalInstallError(RuntimeError):
    """Library mode would mix two copies of cuda_link in this process.

    Raised when a different cuda_link is already loaded, when an import hook redirects
    the name, or when a bare alias name is already owned by a mirror Text DAT.  Must
    abort resolution immediately rather than fall through to the next layer -- no later
    layer can unload a module that is already imported, and falling through risks
    silently accepting whichever copy happens to be loaded.
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


def _td_name(name: str) -> Any:
    """Look up a TD-injected name (``me``, ``project``, ``tdu``, ...) that this DAT module
    may not have in its own globals.

    TD binds ``me``/``op``/``parent`` directly into a DAT module's ``globals()``, but binds
    ``project``/``tdu``/``td`` **only** into that module's private ``__builtins__`` dict --
    ``globals().get("project")`` always returns None in real TD, even though the name is
    reachable and working from ordinary code in the same module (Python resolves free
    variables through ``__builtins__`` as a last step). Check globals() first so a test
    fixture that does ``monkeypatch.setattr(bootstrap, name, ...)`` (i.e. writes straight
    into module globals) still works unchanged.

    Returns ``Any``, not ``object | None``: these are TD's own COMP/DAT/project objects
    (or None), the same "no stub, treat as Any" idiom this repo already applies to ``td.*``
    itself (see the ``replace-imports-with-any`` note in pyproject.toml) -- a precise
    ``object | None`` return would make every ``.expandPath``/``.parent()`` call site a
    pyrefly ``missing-attribute`` error for an attribute that demonstrably exists at runtime.
    """
    value = globals().get(name)
    if value is not None:
        return value
    scope = globals().get("__builtins__")
    if isinstance(scope, dict):
        return scope.get(name)
    return getattr(scope, name, None)


def _expand(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    expander = _td_name("tdu")  # TD-only: expands $VAR and project-relative paths
    if expander is not None:
        with contextlib.suppress(Exception):  # a broken tdu must not crash the bootstrap
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


def _project_roots() -> Iterator[str]:
    """Folders next to the .toe that conventionally hold a per-project install."""
    proj = _td_name("project")
    folder = str(getattr(proj, "folder", "") or "") if proj is not None else ""
    if not folder:
        return
    yield os.path.join(folder, "cuda_link")
    yield os.path.join(folder, "StreamDiffusion")  # StreamDiffusionTD's venv lives here
    yield os.path.join(folder, "src")  # a repo checkout's src-layout package
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


def _layers(basefolder: str | None) -> Iterator[tuple[str, str, bool, bool]]:
    """(label, root, inject-on-sys.path, forgive-package-dir) -- lazy, so later layers
    (project-root walk, find_spec) never run once an earlier layer has already resolved.

    ``forgive-package-dir`` is only set for the two layers a human can mistype: the
    ``basefolder`` argument and ``CUDALINK_LIB_PATH``. The synthetic project-folder roots
    already include a root named literally ``cuda_link`` (``<project>/cuda_link``); forgiving
    that one too would make it swallow its parent -- ``<project>`` itself, the fallback root
    tried after it -- whenever a project keeps its install directly in the project folder,
    which defeats the roots' own preference order.
    """
    yield "folder argument", basefolder or "", True, True
    for root in _project_roots():
        yield "project folder", root, True, False
    yield "CUDALINK_LIB_PATH", _env_libpath(), True, True
    yield "sys.path", _sys_path_root(), False, False


def _site_package_candidates(root: str, *, forgive_package_dir: bool = False) -> list[str]:
    """A project holding a venv, a venv root, and a pre-resolved site-packages / --target dir.

    Only the Windows venv layout (``Lib/site-packages``) is probed: cuda-link's CUDA IPC
    transport targets TouchDesigner on Windows, so the POSIX ``lib/pythonX.Y/site-packages``
    layout never holds an install for this COMP.

    When *forgive_package_dir* is set and *root* itself is named ``cuda_link`` -- the package
    folder, not its parent -- the parent is probed too. There is no per-COMP parameter to catch
    this mistake anymore, so a manually supplied root (``CUDALINK_LIB_PATH`` or the
    ``basefolder`` argument) pointing one level too deep still resolves rather than silently
    failing.
    """
    expanded = _expand(root)
    if not expanded:
        return []
    candidates = [
        os.path.join(expanded, "venv", "Lib", "site-packages"),
        os.path.join(expanded, ".venv", "Lib", "site-packages"),
        os.path.join(expanded, "Lib", "site-packages"),
        expanded,
    ]
    normalized = os.path.normpath(expanded)
    if forgive_package_dir and os.path.normcase(os.path.basename(normalized)) == "cuda_link":
        parent = os.path.dirname(normalized)
        # Skip a bare relative name's empty parent ("cuda_link" -> "") and a drive root
        # ("C:\cuda_link" -> "C:\", whose own dirname is itself) -- neither is a real
        # parent folder to probe, and the drive root would otherwise get injected onto
        # sys.path wholesale.
        if parent and parent != os.path.dirname(parent):
            candidates.append(parent)
    return candidates


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
            f"'pip install -e' of cuda-link) is redirecting it. Remove that install "
            f"(`pip uninstall cuda-link` in that environment) and restart TouchDesigner."
        )
    raise _RivalInstallError(
        f"CUDA-Link is already loaded from {origin}; the selected installation is "
        f"{package_dir}. Restart TouchDesigner to switch installations."
    )


def _check_alias_owner(name: str, module: object, package_dir: str) -> None:
    """Raise unless the module registered under bare *name* comes from *package_dir*.

    Library mode aliases every bare name to the installed package.  A name that another
    module owns before this runs -- a sibling mirror Text DAT imported ahead of this DAT,
    or a different cuda_link copy -- cannot be aliased without leaving the COMP with two
    copies of that module (two ctypes handle classes, two sets of protocol constants),
    which is exactly the mixed-copy failure ADR-0014 exists to prevent.  Refuse library
    mode instead; the mirrors then serve every name from one consistent copy.
    """
    origin = _package_dir(module)
    if origin and _within(origin, package_dir):
        return  # this same installation, aliased by an earlier run -- leave it
    if origin and "cuda_link" in _normalized(origin).split(os.sep):
        raise _RivalInstallError(
            f"CUDA-Link is already loaded from {origin}; the selected installation is "
            f"{package_dir}. Restart TouchDesigner to switch installations."
        )
    raise _RivalInstallError(
        f"{name} was imported from a sibling mirror Text DAT (or another module) before "
        f"CUDALinkBootstrap ran, so library mode from {package_dir} would mix two copies "
        f"of it. Make CUDALinkBootstrap the first import in the COMP, or remove the mirror "
        f"Text DATs, then restart TouchDesigner."
    )


def _activate(site_packages: str, *, inject: bool = True) -> bool:
    """Import cuda_link from *site_packages* and register the bare-name aliases."""
    global last_error, _active, resolved_site_packages, failure_kind
    package_dir = os.path.join(site_packages, "cuda_link")
    before = set(sys.modules)

    loaded = sys.modules.get("cuda_link")
    if loaded is not None:
        _check_origin(loaded, package_dir)
    for name in _ALIAS_MAP:  # preflight before mutating anything
        if name in sys.modules:
            _check_alias_owner(name, sys.modules[name], package_dir)

    previous_index = sys.path.index(site_packages) if site_packages in sys.path else None
    if inject:
        if previous_index is not None:
            sys.path.remove(site_packages)
        sys.path.insert(0, site_packages)
    try:
        # Triggers the package __init__, which is torch-safe: it re-exports torch/numpy/
        # cupy only as guarded *_AVAILABLE flags.
        _check_origin(importlib.import_module("cuda_link"), package_dir, fresh=True)
        for name, target in _ALIAS_MAP.items():
            if name in sys.modules:
                continue  # preflight proved it is this same installation, already aliased
            sys.modules[name] = _check_origin(importlib.import_module(target), package_dir, fresh=True)  # type: ignore[assignment]
    except BaseException:
        if inject:  # put sys.path back exactly as it was, including a pre-existing entry
            if site_packages in sys.path:
                sys.path.remove(site_packages)
            if previous_index is not None:
                sys.path.insert(previous_index, site_packages)
        for key in list(sys.modules):  # drop only what this call imported
            if key not in before and (key == "cuda_link" or key.startswith("cuda_link.") or key in _ALIAS_MAP):
                del sys.modules[key]
        raise

    if inject:
        # Submodules already resolve through cuda_link.__path__, set at import time, so
        # the injected root no longer needs sys.path[0] for the rest of the process --
        # leaving it there would shadow every other same-named top-level module TD has.
        # Put it back where it was (or at the tail, if it is new) instead.
        if site_packages in sys.path:
            sys.path.remove(site_packages)
        if previous_index is not None:
            sys.path.insert(previous_index, site_packages)
        else:
            sys.path.append(site_packages)

    last_error = ""
    resolved_site_packages = site_packages
    _active = True
    failure_kind = ""
    return True


def _fail(message: str, *, kind: str = "") -> bool:
    """Record why library mode is off; the COMP then stays on its mirror Text DATs."""
    global last_error, _active, resolved_site_packages, failure_kind
    last_error = message
    resolved_site_packages = ""
    _active = False
    failure_kind = kind
    return False


def _native_note() -> str:
    """ "" unless *resolved_site_packages* is a bare ``src`` checkout with no compiled
    native wait backend -- see ADR-0014 and ``cuda_link._native_loader.load_native_backend``.

    A ``src``-layout repo checkout never has a built ``_native_waiter*.pyd`` (it is
    gitignored), so cuda_link silently falls back to the pure-Python wait path. That is a
    legitimate, working configuration -- not an error, and ``_project_roots()`` deliberately
    keeps preferring ``src`` over a bare project-folder install (see the layer-order test
    pinning it) -- but the latency difference is real, so it gets one visible note instead
    of a fully silent fallback. A built install in ``<project>/.venv`` would provide the
    native backend instead.
    """
    if not resolved_site_packages:
        return ""
    normalized = os.path.normpath(resolved_site_packages)
    if os.path.normcase(os.path.basename(normalized)) != "src":
        return ""
    package_dir = os.path.join(normalized, "cuda_link")
    try:
        entries = os.listdir(package_dir)
    except OSError:
        return ""
    if any(name.startswith("_native_waiter") and name.endswith(".pyd") for name in entries):
        return ""
    return (
        "[CUDALinkBootstrap] note: this is a source checkout (src/) with no compiled "
        "_native_waiter -- using the pure-Python wait path. A built install in "
        "<project>/.venv would provide the native backend."
    )


def _bootstrap(basefolder: str | None = None) -> bool:
    """Resolve, version-check, import and alias cuda_link.  Returns True on success.

    On failure ``last_error`` names every layer that was tried and why it was skipped;
    the extension prints it to the Textport and the COMP falls back to its mirrors.
    """
    notes: list[str] = []

    # A matching-version cuda_link already loaded from elsewhere in this process (another
    # COMP's bootstrap run, or a second open project) is reused outright, before any layer
    # is walked. Restarting TouchDesigner could never "fix" the rival error the layer walk
    # would otherwise raise here: both COMPs would just resolve to the same already-loaded
    # copy again on the next load. A version mismatch still falls through to the layer walk,
    # which raises the ordinary rival hard stop -- reusing a *wrong*-version copy would let
    # two different-version copies of the ctypes/protocol code coexist in one process.
    saw_broken = False
    loaded = sys.modules.get("cuda_link")
    if loaded is not None:
        origin_dir = _package_dir(loaded)
        if origin_dir:
            already_at = os.path.dirname(origin_dir)
            # Compare the loaded module's own __version__, not a fresh disk read of
            # already_at's __init__.py: the two can differ (the file on disk was edited
            # or replaced after this process imported it), and it is the loaded copy --
            # the one every alias actually points at -- whose version matters here.
            if getattr(loaded, "__version__", "") == MIRROR_VERSION:
                try:
                    return _activate(already_at, inject=False)
                except _RivalInstallError as error:
                    logger.warning("%s", error)
                    return _fail(str(error), kind="rival")
                except Exception as error:
                    notes.append(f"already-loaded cuda_link: {type(error).__name__}: {error}")
                    saw_broken = True

    saw_mismatch = False
    for label, root, inject, forgive in _layers(basefolder):
        if not str(root).strip():
            notes.append(f"{label}: not set")  # quiet deferral, not a warning
            continue
        candidates = [p for p in _site_package_candidates(root, forgive_package_dir=forgive) if _has_package(p)]
        if not candidates:
            notes.append(f"{label}: no cuda_link under {_expand(root)}")
            continue
        for site_packages in candidates:
            found = _read_version(site_packages)
            if found != MIRROR_VERSION:
                saw_mismatch = True
                notes.append(
                    f"{label}: cuda_link {found or '?'} under {site_packages} "
                    f"does not match the component's {MIRROR_VERSION}"
                )
                continue
            try:
                return _activate(site_packages, inject=inject)
            except _RivalInstallError as error:
                logger.warning("%s", error)
                return _fail(str(error), kind="rival")  # hard stop: no later layer can unload a module
            except Exception as error:
                # Anything the candidate raised while importing (ImportError, but also a
                # SyntaxError in a half-copied install or an AttributeError from a
                # dependency) means a matching-version install was found but is broken,
                # not merely absent; _activate has already rolled back. This runs at Text
                # DAT load time, so an escaping exception would leave ext.CUDAIPCExtension
                # undefined instead of falling back to the mirrors.
                notes.append(f"{label}: {type(error).__name__}: {error}")
                saw_broken = True

    # Precedence: a broken matching-version install outranks a version mismatch, which
    # outranks "nothing found" -- the most actionable diagnosis wins.
    if saw_broken:
        kind = "broken"
    elif saw_mismatch:
        kind = "mismatch"
    else:
        kind = "missing"
    return _fail("CUDA-Link could not be resolved -- " + "; ".join(notes), kind=kind)


# Run at Text DAT load time. This runs at import time inside TD, so an escaping exception
# would leave ext.CUDAIPCExtension undefined instead of falling back to the mirrors.
try:
    _bootstrap()
except Exception as _bootstrap_error:  # noqa: BLE001 -- last resort, see comment above
    # _bootstrap() itself only escapes here on a bug in the resolver, not a candidate's
    # own import failure (those are caught inside the layer walk) -- but from the caller's
    # perspective this is still "something was found/attempted and it broke", so it gets
    # the same "broken" kind rather than the misleading "missing".
    _fail(f"{type(_bootstrap_error).__name__}: {_bootstrap_error}", kind="broken")

if _active:
    print(f"[CUDALinkBootstrap] Library mode active — cuda_link {MIRROR_VERSION} from {resolved_site_packages}")
    _note = _native_note()
    if _note:
        print(_note)
else:
    print(
        "[CUDALinkBootstrap] Library mode off — COMP uses its mirror Text DATs if present (reason in CUDALinkBootstrap.last_error)"
    )
