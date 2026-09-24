"""Test that TD-exporter copies stay in sync with canonical sources.

Two check modes:
  byte_identical   — canonical and derived must be byte-identical.
  rewrite_relative — derived must equal the output of rewrite_relative_imports()
                     applied to the canonical source (deep modules with relative imports).

The authoritative pair list lives in scripts/sync_td_wrapper.PAIRS — keep this
docstring free of the listing so it doesn't drift.

To update any paired file: edit src/cuda_link/<file>.py, then run:

    python scripts/sync_td_wrapper.py

Never edit td_exporter/ paired files directly — this test and the pre-commit
hook will reject the commit.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Import the sync script's PAIRS list and rewrite function directly so the
# test and the script always agree on the transform.
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from sync_td_wrapper import CANONICAL_ONLY, NAMES, PAIRS, rewrite_relative_imports  # noqa: E402

# ---------------------------------------------------------------------------
# Parametrise by mode
# ---------------------------------------------------------------------------

_BYTE_IDENTICAL_PAIRS = [(src, dst) for src, dst, mode in PAIRS if mode == "byte_identical"]
_REWRITE_PAIRS = [(src, dst) for src, dst, mode in PAIRS if mode == "rewrite_relative"]

_BYTE_IDS = [dst.stem for _, dst, mode in PAIRS if mode == "byte_identical"]
_REWRITE_IDS = [dst.stem for _, dst, mode in PAIRS if mode == "rewrite_relative"]


@pytest.mark.parametrize(
    "canonical,derived",
    _BYTE_IDENTICAL_PAIRS,
    ids=_BYTE_IDS,
)
def test_td_exporter_file_is_identical(canonical: Path, derived: Path) -> None:
    """Verify each byte_identical TD-exporter copy is byte-identical to its canonical source."""
    assert canonical.exists(), f"Canonical source not found: {canonical}"
    assert derived.exists(), f"Derived copy not found: {derived}\nRun: python scripts/sync_td_wrapper.py"

    canonical_content = canonical.read_text(encoding="utf-8")
    derived_content = derived.read_text(encoding="utf-8")

    assert canonical_content == derived_content, (
        f"{derived.name} is out of sync with {canonical.name}!\n"
        f"  canonical: {canonical} ({len(canonical_content)} chars)\n"
        f"  derived:   {derived} ({len(derived_content)} chars)\n"
        "\nRun: python scripts/sync_td_wrapper.py"
    )


@pytest.mark.parametrize(
    "canonical,derived",
    _REWRITE_PAIRS,
    ids=_REWRITE_IDS,
)
def test_td_exporter_rewrite_pair_is_in_sync(canonical: Path, derived: Path) -> None:
    """Verify each rewrite_relative TD-exporter copy matches the expected transform output."""
    assert canonical.exists(), f"Canonical source not found: {canonical}"
    assert derived.exists(), f"Derived copy not found: {derived}\nRun: python scripts/sync_td_wrapper.py"

    canonical_content = canonical.read_text(encoding="utf-8")
    expected = rewrite_relative_imports(canonical_content)
    derived_content = derived.read_text(encoding="utf-8")

    assert derived_content == expected, (
        f"{derived.name} is out of sync with the rewrite of {canonical.name}!\n"
        f"  canonical: {canonical} ({len(canonical_content)} chars)\n"
        f"  derived:   {derived} ({len(derived_content)} chars)\n"
        "\nRun: python scripts/sync_td_wrapper.py"
    )


def test_pairs_cover_all_mirrorable_modules() -> None:
    """Every src/cuda_link/*.py must be in PAIRS or CANONICAL_ONLY.

    This invariant prevents a module from being silently omitted from the sync
    script (as happened with _profile.py / FrameProfile.py).  To add a new
    canonical module: register it in PAIRS in scripts/sync_td_wrapper.py, or
    add its filename to CANONICAL_ONLY if it intentionally has no TD twin.
    """
    project_root = Path(__file__).parent.parent.parent
    src_dir = project_root / "src" / "cuda_link"
    paired_canonicals = {src for src, _, _ in PAIRS}

    unregistered = []
    for module_path in sorted(src_dir.glob("*.py")):
        if module_path.name in CANONICAL_ONLY:
            continue
        if module_path in paired_canonicals:
            continue
        unregistered.append(module_path.name)

    assert not unregistered, (
        "The following src/cuda_link/ modules are neither in PAIRS nor in CANONICAL_ONLY:\n"
        + "".join(f"  {name}\n" for name in unregistered)
        + "Add each to PAIRS in scripts/sync_td_wrapper.py (with the appropriate mode),\n"
        "or to CANONICAL_ONLY if it intentionally has no td_exporter twin."
    )


def test_wrapper_contains_key_definitions() -> None:
    """Verify wrapper contains expected CUDA definitions."""
    project_root = Path(__file__).parent.parent.parent
    pip_wrapper = project_root / "src" / "cuda_link" / "cuda_ipc_wrapper.py"
    types_module = project_root / "src" / "cuda_link" / "cuda_runtime_types.py"

    wrapper_content = pip_wrapper.read_text(encoding="utf-8")
    types_content = types_module.read_text(encoding="utf-8")

    # IPC handle structs live in cuda_runtime_types (extracted in Phase 1)
    assert "class cudaIpcMemHandle_t(ctypes.Structure):" in types_content
    assert "class cudaIpcEventHandle_t(ctypes.Structure):" in types_content

    # Runtime API and factory stay in cuda_ipc_wrapper (inherits from CUDAGraphsMixin after Phase 2)
    assert "class CUDARuntimeAPI(" in wrapper_content
    assert "def get_cuda_runtime(" in wrapper_content

    # Check for key methods
    assert "def malloc(" in wrapper_content
    assert "def free(" in wrapper_content
    assert "def malloc_host(" in wrapper_content
    assert "def free_host(" in wrapper_content
    assert "def memcpy_async(" in wrapper_content
    assert "def ipc_get_mem_handle(" in wrapper_content
    assert "def ipc_open_mem_handle(" in wrapper_content
    assert "def ipc_close_mem_handle(" in wrapper_content
    assert "def record_event(" in wrapper_content
    assert "def stream_wait_event(" in wrapper_content


# ---------------------------------------------------------------------------
# Import-shape invariants: each deployment resolves its dependencies against ONE
# installation (ADR-0014).  The sync script can only guarantee that when every
# mirrored dependency is reached through a relative import it can rewrite.
# ---------------------------------------------------------------------------

_ALL_CANONICALS = [src for src, _, _ in PAIRS]
_ALL_IDS = [src.stem for src, _, _ in PAIRS]
_BYTE_IDENTICAL_CANONICALS = [src for src, _, mode in PAIRS if mode == "byte_identical"]
_CANONICAL_ONLY_STEMS = {Path(name).stem for name in CANONICAL_ONLY}
_BARE_TD_NAMES = set(NAMES.values())


def _import_nodes(path: Path) -> list[ast.Import | ast.ImportFrom]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]


@pytest.mark.parametrize("canonical", _ALL_CANONICALS, ids=_ALL_IDS)
def test_paired_source_never_imports_a_mirrored_module_absolutely(canonical: Path) -> None:
    """A paired module reaches its mirrored siblings through relative imports only.

    ``from cuda_link.x import ...`` inside a paired module resolves against whichever
    cuda_link is importable in the process -- not necessarily the installation the COMP
    was resolved against.  In TouchDesigner that silently mixes a sibling Text DAT with a
    system-installed copy of a different version.  A relative import is rewritten by the
    sync script to the bare TD name, so both deployments stay self-contained.  Modules in
    CANONICAL_ONLY have no TD twin and may be imported absolutely (guarded at the call site).
    """
    offenders: list[str] = []
    for node in _import_nodes(canonical):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or not (module == "cuda_link" or module.startswith("cuda_link.")):
                continue
            stem = module.partition("cuda_link.")[2]
            if stem in _CANONICAL_ONLY_STEMS:
                continue
            offenders.append(f"line {node.lineno}: from {module} import ...")
        else:
            for alias in node.names:
                if alias.name == "cuda_link" or alias.name.startswith("cuda_link."):
                    offenders.append(f"line {node.lineno}: import {alias.name}")

    assert not offenders, (
        f"{canonical.name} imports mirrored modules by absolute package path:\n"
        + "".join(f"  {line}\n" for line in offenders)
        + "Use a relative import (from ._env import ...) and register the pair as\n"
        "rewrite_relative in scripts/sync_td_wrapper.py::PAIRS."
    )


@pytest.mark.parametrize("canonical", _ALL_CANONICALS, ids=_ALL_IDS)
def test_paired_source_never_imports_a_mirror_by_its_td_name(canonical: Path) -> None:
    """The package must not import ``Env``, ``CUDARuntimeTypes``, ... by bare TD name.

    That name is the sync script's OUTPUT.  Inside a package context it resolves to
    whatever module of that name sits on sys.path (td_exporter/ is on the test pythonpath),
    which is a second copy of the same code -- ctypes argtypes compare handle classes by
    identity, so a second ``cudaIpcMemHandle_t`` class breaks every IPC call.
    """
    offenders: list[str] = []
    for node in _import_nodes(canonical):
        if isinstance(node, ast.ImportFrom):
            if not node.level and node.module in _BARE_TD_NAMES:
                offenders.append(f"line {node.lineno}: from {node.module} import ...")
        else:
            offenders.extend(f"line {node.lineno}: import {a.name}" for a in node.names if a.name in _BARE_TD_NAMES)

    assert not offenders, (
        f"{canonical.name} imports a mirror by its TD name:\n"
        + "".join(f"  {line}\n" for line in offenders)
        + "Use the relative package import; the sync script derives the TD name."
    )


@pytest.mark.parametrize("canonical", _BYTE_IDENTICAL_CANONICALS, ids=_BYTE_IDS)
def test_byte_identical_source_has_no_relative_imports(canonical: Path) -> None:
    """byte_identical copies are verbatim: a relative import would break in TD's flat namespace.

    Move the pair to rewrite_relative in scripts/sync_td_wrapper.py::PAIRS instead.
    """
    relative = [
        f"line {node.lineno}: from {'.' * node.level}{node.module or ''} import ..."
        for node in _import_nodes(canonical)
        if isinstance(node, ast.ImportFrom) and node.level
    ]
    assert not relative, f"{canonical.name} is byte_identical but uses relative imports:\n" + "\n".join(relative)
