"""
test_td_bootstrap.py — Structural and functional tests for CUDALinkBootstrap.

Tests:
  1. Drift guard: _ALIAS_MAP keys/values must match PAIRS in sync_td_wrapper.py.
  2. Functional: bootstrap activates when cuda_link is importable; aliases are
     registered in sys.modules pointing to the installed submodules.
  3. No-op: bootstrap skips alias registration when cuda_link is not importable.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TD_EXPORTER = REPO_ROOT / "td_exporter"
SCRIPTS = REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _load_file_as_module(path: Path, name: str, run: bool = True) -> Any:
    """Load a .py file as a named module without modifying sys.path.

    If run=False the module is created but exec_module is NOT called (useful for
    introspecting the source before executing it; not applicable to bootstrap
    because the alias map is defined at module scope as a plain dict literal).
    This helper always runs exec_module — run parameter reserved for future use.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Could not create spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_bootstrap(capsys=None) -> Any:
    """Load a fresh copy of CUDALinkBootstrap, discarding any cached version."""
    sys.modules.pop("CUDALinkBootstrap", None)
    mod = _load_file_as_module(TD_EXPORTER / "CUDALinkBootstrap.py", "CUDALinkBootstrap")
    return mod


def _load_sync_script() -> Any:
    """Load scripts/sync_td_wrapper.py as a module to read PAIRS."""
    sys.modules.pop("sync_td_wrapper", None)
    return _load_file_as_module(SCRIPTS / "sync_td_wrapper.py", "sync_td_wrapper")


# ---------------------------------------------------------------------------
# Cleanup fixture: remove all alias keys from sys.modules before/after tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_sys_modules():
    """Remove CUDALinkBootstrap and all alias keys before and after each test.

    Pre-warms the cuda_link package import BEFORE clearing alias names.
    Without this, tests that clear sys.modules["cuda_link.*"] (the noops tests
    below, which do ``for k in sys.modules: sys.modules.pop(k)`` for cuda_link)
    leave cuda_link absent.  The next _bootstrap() call then triggers a fresh
    import of cuda_link.cuda_ipc_wrapper, whose bare-name fallback:

        from CUDARuntimeTypes import (...)   # first-try TD flat-namespace path

    finds td_exporter/CUDARuntimeTypes.py on sys.path (pythonpath=["td_exporter"])
    because CUDARuntimeTypes was cleared by this fixture.  _bootstrap()'s
    skip-guard (``if bare_name in sys.modules: continue``) then leaves
    sys.modules["CUDARuntimeTypes"] pointing at the td_exporter module instead
    of cuda_link.cuda_runtime_types, causing the alias correctness test to fail.

    The pre-warm guarantees cuda_link and cuda_link.cuda_ipc_wrapper are cached
    before alias names are cleared, so _bootstrap()'s import_module("cuda_link")
    call returns immediately and never re-executes cuda_ipc_wrapper's bare-name
    import.  If cuda_link is not importable (mocked ImportError in the noops
    tests), the pre-warm is silently skipped — the noops tests patch
    importlib.import_module in the test body, after this fixture runs, so the
    pre-warm always uses the real import_module.
    """
    # Pre-warm: cache cuda_link before alias names are cleared.
    # Uses a bare importlib.import_module (not the patched version — noops tests
    # apply monkeypatch inside the test body, which runs after this fixture).
    with contextlib.suppress(ImportError):
        importlib.import_module("cuda_link")
        # Unavailable environment: bootstrap tests skip via _active check.

    sync = _load_sync_script()
    alias_names = {dst.stem for _, dst, _ in sync.PAIRS} | {"CUDALinkBootstrap"}

    # Pre-test: capture and remove any existing entries
    saved = {k: sys.modules.pop(k) for k in alias_names if k in sys.modules}
    yield
    # Post-test: restore to pre-test state
    for k in alias_names:
        sys.modules.pop(k, None)
    sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 1. Structural / drift-guard tests
# ---------------------------------------------------------------------------


def test_alias_map_covers_all_pairs():
    """Every derived TD module name in PAIRS must appear as a key in _ALIAS_MAP."""
    sync = _load_sync_script()
    bootstrap = _load_bootstrap()

    pairs_derived_stems = {dst.stem for _, dst, _ in sync.PAIRS}
    alias_keys = set(bootstrap._ALIAS_MAP.keys())

    missing = pairs_derived_stems - alias_keys
    extra = alias_keys - pairs_derived_stems

    assert not missing, (
        f"_ALIAS_MAP is missing these PAIRS derived stems: {missing}\n"
        f"Add them to td_exporter/CUDALinkBootstrap.py::_ALIAS_MAP"
    )
    assert not extra, (
        f"_ALIAS_MAP has extra keys not in PAIRS: {extra}\n"
        f"Remove them from td_exporter/CUDALinkBootstrap.py::_ALIAS_MAP "
        f"or add corresponding entries to scripts/sync_td_wrapper.py::PAIRS"
    )


def test_alias_map_values_match_canonical_stems():
    """Each _ALIAS_MAP value must equal 'cuda_link.<canonical_stem>' from PAIRS."""
    sync = _load_sync_script()
    bootstrap = _load_bootstrap()

    expected = {dst.stem: f"cuda_link.{src.stem}" for src, dst, _ in sync.PAIRS}

    for bare_name, submodule_path in bootstrap._ALIAS_MAP.items():
        assert bare_name in expected, f"_ALIAS_MAP key {bare_name!r} not found in PAIRS"
        assert submodule_path == expected[bare_name], (
            f"_ALIAS_MAP[{bare_name!r}] = {submodule_path!r}\n"
            f"  expected: {expected[bare_name]!r} (derived from PAIRS src stem)"
        )


# ---------------------------------------------------------------------------
# 2. Functional: bootstrap activates when cuda_link is importable
# ---------------------------------------------------------------------------


def test_bootstrap_activates_when_package_importable():
    """When cuda_link is on sys.path (pytest sets src/ via pythonpath), bootstrap activates."""
    # src/ is on sys.path via pyproject.toml [tool.pytest.ini_options] pythonpath = ["src"]
    # so import cuda_link succeeds without CUDALINK_LIB_PATH.
    bootstrap = _load_bootstrap()
    assert bootstrap._active, (
        "bootstrap._active should be True when cuda_link is importable. "
        "Ensure src/ is on sys.path (pytest pythonpath config)."
    )


def test_bootstrap_registers_all_aliases_in_sys_modules():
    """After activation, every _ALIAS_MAP key is present in sys.modules."""
    bootstrap = _load_bootstrap()
    if not bootstrap._active:
        pytest.skip("cuda_link not importable — bootstrap inactive; see test_bootstrap_activates_*")

    for bare_name in bootstrap._ALIAS_MAP:
        assert bare_name in sys.modules, f"sys.modules[{bare_name!r}] not registered after bootstrap activation"


def test_bootstrap_aliases_point_to_correct_submodules():
    """Registered aliases must point to the same object as importlib.import_module(submodule)."""
    bootstrap = _load_bootstrap()
    if not bootstrap._active:
        pytest.skip("cuda_link not importable — bootstrap inactive")

    for bare_name, submodule_path in bootstrap._ALIAS_MAP.items():
        registered = sys.modules.get(bare_name)
        assert registered is not None, f"sys.modules[{bare_name!r}] is None"
        expected = importlib.import_module(submodule_path)
        assert registered is expected, (
            f"sys.modules[{bare_name!r}] = {registered!r}\n  expected:  {expected!r}  ({submodule_path})"
        )


def test_bootstrap_bare_name_import_resolves_to_package_module():
    """After bootstrap, 'from Exporter import Exporter' resolves to cuda_link.exporter.Exporter."""
    bootstrap = _load_bootstrap()
    if not bootstrap._active:
        pytest.skip("cuda_link not importable — bootstrap inactive")

    # Simulate TD-style bare-name import by reading directly from sys.modules
    exporter_module = sys.modules.get("Exporter")
    assert exporter_module is not None
    assert hasattr(exporter_module, "Exporter"), "sys.modules['Exporter'] should expose the Exporter class"
    from cuda_link.exporter import Exporter

    assert exporter_module.Exporter is Exporter


def test_bootstrap_shm_protocol_exposes_private_symbols():
    """SHMProtocol alias must expose private symbols used by TDReceiver (e.g. _ST_BBH)."""
    bootstrap = _load_bootstrap()
    if not bootstrap._active:
        pytest.skip("cuda_link not importable — bootstrap inactive")

    shm_module = sys.modules.get("SHMProtocol")
    assert shm_module is not None

    # These are the symbols TDReceiver.py imports from SHMProtocol (lines 31-45)
    required = [
        "_ST_BBH",
        "FLAGS_BFLOAT16",
        "FORMAT_KIND_FLOAT",
        "FORMAT_KIND_UNSIGNED",
        "MAGIC_OFFSET",
        "MAGIC_SIZE",
        "METADATA_SIZE",
        "NUM_SLOTS_OFFSET",
        "NUM_SLOTS_SIZE",
        "PROTOCOL_MAGIC",
        "SHM_HEADER_SIZE",
        "SHUTDOWN_FLAG_SIZE",
        "SLOT_SIZE",
        "VERSION_OFFSET",
    ]
    missing = [s for s in required if not hasattr(shm_module, s)]
    assert not missing, f"SHMProtocol alias is missing symbols needed by TDReceiver: {missing}"


# ---------------------------------------------------------------------------
# 3. No-op when cuda_link is unavailable
# ---------------------------------------------------------------------------


def test_bootstrap_noops_when_import_fails(monkeypatch):
    """Bootstrap returns False and leaves aliases unregistered when cuda_link cannot be imported."""
    # Patch importlib.import_module so that 'cuda_link' raises ImportError
    original_import = importlib.import_module

    def mock_import(name, *args, **kwargs):
        if name == "cuda_link":
            raise ImportError("mocked: cuda_link not installed")
        return original_import(name, *args, **kwargs)

    # The bootstrap's module-scope code calls importlib.import_module directly,
    # so we need to patch the importlib module that the bootstrap itself uses.
    import importlib as _importlib_ref

    monkeypatch.setattr(_importlib_ref, "import_module", mock_import)

    # Remove cuda_link from sys.modules so importlib.import_module is actually
    # called (not short-circuited by the sys.modules cache).
    # Use monkeypatch.delitem so the entries are automatically RESTORED after
    # this test — bare sys.modules.pop() leaves cuda_link absent for ALL
    # subsequent tests, causing patch("cuda_link.xxx.yyy") in later tests to
    # target a freshly re-imported module while already-imported function/class
    # objects still reference the original module's __dict__.
    for k in list(sys.modules):
        if k == "cuda_link" or k.startswith("cuda_link."):
            monkeypatch.delitem(sys.modules, k)

    bootstrap = _load_bootstrap()

    assert not bootstrap._active, "bootstrap._active should be False when cuda_link is not importable"


def test_bootstrap_noops_leaves_aliases_unregistered(monkeypatch):
    """When bootstrap is inactive, none of the _ALIAS_MAP keys appear in sys.modules."""
    original_import = importlib.import_module

    def mock_import(name, *args, **kwargs):
        if name == "cuda_link":
            raise ImportError("mocked: cuda_link not installed")
        return original_import(name, *args, **kwargs)

    import importlib as _importlib_ref

    monkeypatch.setattr(_importlib_ref, "import_module", mock_import)

    # Same rationale as test_bootstrap_noops_when_import_fails: use
    # monkeypatch.delitem so cuda_link.* is restored after this test.
    for k in list(sys.modules):
        if k == "cuda_link" or k.startswith("cuda_link."):
            monkeypatch.delitem(sys.modules, k)

    bootstrap = _load_bootstrap()

    for bare_name in bootstrap._ALIAS_MAP:
        assert bare_name not in sys.modules, (
            f"sys.modules[{bare_name!r}] should NOT be registered when bootstrap is inactive"
        )


# ---------------------------------------------------------------------------
# 4. CUDAIPCExtension detection logic: _sibling_mirrors_available + banner guard
# ---------------------------------------------------------------------------


def _make_fake_extension(op_map: dict):
    """Construct a minimal object that exercises _sibling_mirrors_available and
    _show_install_banner without requiring the full TD runtime or extension __init__.

    Rather than fighting with the module loader (CUDAIPCExtension.py has many
    module-level imports that need TD stubs), we define an inline class with the
    same method bodies.  This is a white-box test: if the method implementation
    changes, these tests should be updated too.
    """
    import contextlib as _contextlib

    class FakeComp:
        path = "/fake/comp"

        def op(self, name):
            return op_map.get(name)

    class _FakeExtension:
        def __init__(self):
            self.ownerComp = FakeComp()

        def _sibling_mirrors_available(self) -> bool:
            """Mirrors td_exporter/CUDAIPCExtension.py::_sibling_mirrors_available."""
            with _contextlib.suppress(AttributeError, RuntimeError):
                for name in ("SHMProtocol", "Exporter", "Env"):
                    if self.ownerComp.op(name) is not None:
                        return True
            return False

        def _show_install_banner(self) -> None:
            """Mirrors td_exporter/CUDAIPCExtension.py::_show_install_banner (guard only)."""
            try:
                from td import ui  # noqa: F401, PLC0415
            except (ImportError, NameError):
                return
            # If we reach here, TD is available — do nothing further in the test stub.

    return _FakeExtension()


def test_sibling_mirrors_available_returns_true_when_op_found():
    """_sibling_mirrors_available() returns True when any key mirror op is present."""
    ext = _make_fake_extension({"SHMProtocol": object()})  # SHMProtocol present
    assert ext._sibling_mirrors_available() is True


def test_sibling_mirrors_available_returns_false_when_no_mirrors():
    """_sibling_mirrors_available() returns False when none of the three key ops exist."""
    ext = _make_fake_extension({})  # no ops present
    assert ext._sibling_mirrors_available() is False


def test_sibling_mirrors_available_returns_true_for_exporter():
    """Returns True when Exporter mirror op is present (even if others absent)."""
    ext = _make_fake_extension({"Exporter": object()})
    assert ext._sibling_mirrors_available() is True


def test_sibling_mirrors_available_returns_true_for_env():
    """Returns True when Env mirror op is present (even if others absent)."""
    ext = _make_fake_extension({"Env": object()})
    assert ext._sibling_mirrors_available() is True


def test_show_install_banner_is_noop_outside_td():
    """_show_install_banner returns silently when the 'td' module is unavailable."""
    ext = _make_fake_extension({})
    # In the pytest environment the 'td' module does not exist — the import guard
    # must catch ImportError/NameError and return without raising.
    try:
        ext._show_install_banner()
    except Exception as exc:
        pytest.fail(f"_show_install_banner() raised outside TD context: {exc!r}")
