"""
test_td_bootstrap.py — Structural and functional tests for CUDALinkBootstrap.

Tests:
  1. Drift guard: _ALIAS_MAP keys/values must match PAIRS in sync_td_wrapper.py.
  2. Functional: bootstrap activates when cuda_link is importable; aliases are
     registered in sys.modules pointing to the installed submodules.
  3. No-op: bootstrap skips alias registration when cuda_link is not importable.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import importlib.machinery
import importlib.util
import sys
import types
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

    Pre-warms the cuda_link package import BEFORE clearing alias names so that
    ``_bootstrap()``'s ``import_module("cuda_link")`` hits the module cache and
    never re-executes the package under a half-cleared ``sys.modules``.  The
    package imports its own dependencies relatively since ADR-0002 was enforced
    for every mirrored file, so the historical hazard (a bare
    ``from CUDARuntimeTypes import ...`` fallback resolving against the
    td_exporter copy on ``pythonpath``) is gone; the pre-warm is kept as cheap
    insurance for the noops tests, which pop every ``cuda_link*`` entry.  If
    cuda_link is not importable the pre-warm is silently skipped — the noops
    tests patch importlib.import_module in the test body, after this fixture.
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


# ---------------------------------------------------------------------------
# 5. Project-anchored resolver: layered lookup, version stamp, rival-install guard
# ---------------------------------------------------------------------------


def _make_fake_install(root: Path, version: str, alias_map: dict[str, str]) -> str:
    """A minimal cuda_link package under *root*: __init__ carrying __version__ plus one
    empty module per _ALIAS_MAP target, so every alias import succeeds with no real code."""
    package = root / "cuda_link"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    for target in alias_map.values():
        stem = target.split(".")[-1]
        (package / f"{stem}.py").write_text("# fake alias target for tests\n", encoding="utf-8")
    return str(root)


def _forget_loaded_cuda_link(bootstrap: Any) -> None:
    """Drop every cuda_link.* module and every bare alias so the next _bootstrap() call
    resolves from scratch (the module-scope run at load time already aliased src/)."""
    for key in list(sys.modules):
        if key == "cuda_link" or key.startswith("cuda_link."):
            del sys.modules[key]
    for key in bootstrap._ALIAS_MAP:
        sys.modules.pop(key, None)
    importlib.invalidate_caches()


def _is_editable_finder(finder: object) -> bool:
    """True for the redirecting finder an editable install parks on sys.meta_path
    (scikit-build-core ``_cuda_link_editable``, setuptools ``__editable___*_finder``,
    hatchling ``_editable_impl_*``) -- it serves cuda_link from src/ ahead of sys.path."""
    cls = finder if isinstance(finder, type) else type(finder)
    return "editable" in f"{cls.__module__}.{cls.__qualname__}".lower()


class _RedirectingFinder:
    """Stand-in for that editable finder: serves ``cuda_link`` from one fixed folder no
    matter what sys.path says, exactly like ``pip install -e .`` does in CI."""

    def __init__(self, site_packages: str) -> None:
        self._search = [site_packages]

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> Any:
        if fullname == "cuda_link":
            return importlib.machinery.PathFinder.find_spec(fullname, self._search)
        return None


class _FakeProject:
    """Stand-in for TD's ``project`` global: only ``.folder`` is read by the bootstrap."""

    def __init__(self, folder: str) -> None:
        self.folder = folder


class _SysPathLayer:
    """Stand-in for layer (e) installed by ``isolated_resolver``: quiet by default so the
    src/ checkout on pytest's pythonpath never resolves behind a test's back; set
    ``enabled`` to run the real ``_sys_path_root`` (find_spec) again."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.enabled = False

    def __call__(self) -> str:
        return self._real() if self.enabled else ""


@pytest.fixture
def isolated_resolver(monkeypatch):
    """Fresh bootstrap module with every implicit layer silenced.

    Snapshots sys.path and the loaded cuda_link.* modules (after the module-scope run
    has imported all of them from src/) and restores both afterwards, so activating a
    fake install inside a test never leaks into the rest of the suite.  Layer (b) project
    folder is quiet on its own outside TD (no ``project`` global); (c) CUDALINK_LIB_PATH
    is cleared and (d) sys.path is silenced by a ``_SysPathLayer`` stand-in, until a test
    enables one explicitly.

    An editable install of this repo (``pip install -e .``, as branch-protection.yml
    does) adds a finder ahead of sys.path that would hijack every activation below;
    it is hidden here and covered on its own by
    test_import_hook_ahead_of_sys_path_is_refused.
    """
    bootstrap = _load_bootstrap()
    saved_modules = {k: v for k, v in sys.modules.items() if k == "cuda_link" or k.startswith("cuda_link.")}
    saved_path = list(sys.path)
    monkeypatch.setattr(sys, "meta_path", [f for f in sys.meta_path if not _is_editable_finder(f)])
    monkeypatch.delenv("CUDALINK_LIB_PATH", raising=False)
    monkeypatch.setattr(bootstrap, "_sys_path_root", _SysPathLayer(bootstrap._sys_path_root))
    yield bootstrap
    for key in list(sys.modules):
        if key == "cuda_link" or key.startswith("cuda_link."):
            del sys.modules[key]
    sys.modules.update(saved_modules)
    sys.path[:] = saved_path
    importlib.invalidate_caches()


def test_explicit_folder_activates_a_matching_install(isolated_resolver, tmp_path):
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=site) is True
    assert bootstrap._active is True
    assert bootstrap.last_error == ""
    assert bootstrap.resolved_site_packages == site
    origin = Path(str(getattr(sys.modules["cuda_link"], "__file__", ""))).resolve()
    assert origin.parent == (tmp_path / "lib" / "cuda_link").resolve()
    for bare_name, target in bootstrap._ALIAS_MAP.items():
        assert sys.modules[bare_name] is sys.modules[target]


def test_venv_layout_is_probed_under_the_given_root(isolated_resolver, tmp_path):
    bootstrap = isolated_resolver
    site_dir = tmp_path / "proj" / "venv" / "Lib" / "site-packages"
    site = _make_fake_install(site_dir, bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=str(tmp_path / "proj")) is True
    assert bootstrap.resolved_site_packages == site


def test_venv_root_itself_is_probed(isolated_resolver, tmp_path):
    """A root may name the venv itself (install_td_library.py --venv D:/proj/.venv)."""
    bootstrap = isolated_resolver
    venv = tmp_path / "proj" / ".venv"
    site = _make_fake_install(venv / "Lib" / "site-packages", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=str(venv)) is True
    assert bootstrap.resolved_site_packages == site


def test_successful_activation_moves_the_injected_root_off_sys_path_head(isolated_resolver, tmp_path):
    """Submodules already resolve through cuda_link.__path__, set at import time, so a
    successful activation must not leave the injected root shadowing every other
    same-named top-level module at sys.path[0] for the rest of the process."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=site) is True
    assert site in sys.path
    assert sys.path[0] != site
    assert sys.path[-1] == site


def test_successful_activation_restores_its_original_sys_path_index(isolated_resolver, tmp_path):
    """A root the host already put on sys.path (TD Preferences, an earlier activation)
    goes back to that same index after activation, rather than staying at 0 or moving
    to the tail."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _forget_loaded_cuda_link(bootstrap)
    sys.path.insert(2, site)
    snapshot_len = len(sys.path)

    assert bootstrap._activate(site, inject=True) is True
    assert sys.path.index(site) == 2
    assert len(sys.path) == snapshot_len


def test_failed_activation_is_rolled_back_before_the_next_candidate(isolated_resolver, tmp_path):
    """An install whose alias import blows up half-way must leave no cuda_link.* or alias
    entries behind: the next candidate under the same root then activates instead of
    being refused as a rival, and with no other candidate the COMP falls back cleanly."""
    bootstrap = isolated_resolver
    root = tmp_path / "proj"
    broken = _make_fake_install(root / "venv" / "Lib" / "site-packages", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    (Path(broken) / "cuda_link" / "importer.py").write_text("raise ImportError('boom')\n", encoding="utf-8")
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=broken) is False
    assert bootstrap._active is False
    assert "boom" in bootstrap.last_error
    assert not [k for k in sys.modules if k == "cuda_link" or k.startswith("cuda_link.")]
    assert not [k for k in bootstrap._ALIAS_MAP if k in sys.modules]
    assert broken not in sys.path

    good = _make_fake_install(root, bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    assert bootstrap._bootstrap(basefolder=str(root)) is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == good
    origin = Path(str(getattr(sys.modules["cuda_link"], "__file__", ""))).resolve()
    assert origin.parent == (root / "cuda_link").resolve()


def test_failed_activation_restores_a_pre_existing_sys_path_entry(isolated_resolver, tmp_path):
    """Rollback must not strip a path entry the host put there (TD Preferences module path,
    an earlier activation): sys.path ends up exactly as it was before the attempt."""
    bootstrap = isolated_resolver
    broken = _make_fake_install(tmp_path / "prefs", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    (Path(broken) / "cuda_link" / "importer.py").write_text("raise ImportError('boom')\n", encoding="utf-8")
    _forget_loaded_cuda_link(bootstrap)
    sys.path.append(broken)  # the fixture restores sys.path afterwards
    snapshot = list(sys.path)

    assert bootstrap._bootstrap(basefolder=broken) is False
    assert "boom" in bootstrap.last_error
    assert sys.path == snapshot


def test_activate_success_path_tolerates_sys_path_already_stripped(isolated_resolver, tmp_path):
    """If something during import already removed *site_packages* from sys.path (the
    package's own __init__ mutating sys.path, here -- simulating another thread or a
    dependency doing the same), the post-import cleanup must not raise ValueError trying
    to remove it a second time (fix 10)."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    init_file = Path(site) / "cuda_link" / "__init__.py"
    init_file.write_text(
        f'__version__ = "{bootstrap.MIRROR_VERSION}"\n'
        "import os as _os\n"
        "import sys as _sys\n"
        "_here = _os.path.normpath(_os.path.dirname(_os.path.dirname(__file__)))\n"
        "_sys.path[:] = [p for p in _sys.path if _os.path.normpath(p) != _here]\n",
        encoding="utf-8",
    )
    _forget_loaded_cuda_link(bootstrap)

    # The point of this test is that _activate does not raise (ValueError from
    # sys.path.remove on an already-missing entry) -- it must still succeed and put
    # site_packages back at the tail exactly as it would for any other fresh injection.
    assert bootstrap._bootstrap(basefolder=site) is True, bootstrap.last_error
    assert sys.path.count(site) == 1


def test_any_exception_from_a_candidate_is_recorded_instead_of_escaping(isolated_resolver, tmp_path):
    """A candidate that raises something other than ImportError (a SyntaxError in a
    half-copied install, an AttributeError from a dependency) is skipped like any other
    failure.  The module-scope run would otherwise propagate it out of
    ``import CUDALinkBootstrap`` and leave ext.CUDAIPCExtension undefined."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    (Path(site) / "cuda_link" / "shm_protocol.py").write_text("def broken(:\n", encoding="utf-8")
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=site) is False
    assert bootstrap._active is False
    assert "SyntaxError" in bootstrap.last_error
    assert "cuda_link" not in sys.modules
    assert site not in sys.path


def test_broken_candidate_reports_failure_kind_broken_not_missing(isolated_resolver, tmp_path):
    """A matching-version candidate that raises while importing (a syntax error in a
    half-copied install, here) was FOUND, not missing -- failure_kind must say "broken"
    so the Status line never claims cuda_link was never found when it actually was, just
    unusable (fix 3)."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    (Path(site) / "cuda_link" / "shm_protocol.py").write_text("def broken(:\n", encoding="utf-8")
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=site) is False
    assert bootstrap.failure_kind == "broken"


def test_mirror_dat_imported_ahead_of_the_bootstrap_refuses_library_mode(isolated_resolver, tmp_path, monkeypatch):
    """A bare alias name already owned by a sibling Text DAT (a module with no file on disk)
    is never mixed with the installed package: library mode is refused outright, the DAT's
    module is left untouched and nothing from the install is imported."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _forget_loaded_cuda_link(bootstrap)
    mirror = types.ModuleType("CUDARuntimeTypes")  # TD DAT modules carry no __file__
    monkeypatch.setitem(sys.modules, "CUDARuntimeTypes", mirror)

    assert bootstrap._bootstrap(basefolder=site) is False
    assert bootstrap._active is False
    assert "CUDARuntimeTypes" in bootstrap.last_error
    assert "mirror Text DAT" in bootstrap.last_error
    assert sys.modules["CUDARuntimeTypes"] is mirror
    assert "cuda_link" not in sys.modules
    assert not [k for k in bootstrap._ALIAS_MAP if k in sys.modules and k != "CUDARuntimeTypes"]
    assert site not in sys.path


def test_project_folder_layer_prefers_a_cuda_link_folder_next_to_the_toe(isolated_resolver, tmp_path, monkeypatch):
    """Layer (c), first root: <project.folder>/cuda_link (a ``pip install --target`` folder)
    wins over a StreamDiffusion venv in the same project."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    target = _make_fake_install(proj / "cuda_link", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _make_fake_install(
        proj / "StreamDiffusion" / "venv" / "Lib" / "site-packages", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP
    )
    monkeypatch.setattr(bootstrap, "project", _FakeProject(str(proj)), raising=False)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == target
    origin = Path(str(getattr(sys.modules["cuda_link"], "__file__", ""))).resolve()
    assert origin.parent == (proj / "cuda_link" / "cuda_link").resolve()


def test_project_folder_layer_prefers_streamdiffusion_venv_over_the_folder_itself(
    isolated_resolver, tmp_path, monkeypatch
):
    """Layer (c), second root: <project.folder>/StreamDiffusion (StreamDiffusionTD's venv)
    wins over an install sitting directly in the project folder."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    sdtd = _make_fake_install(
        proj / "StreamDiffusion" / "venv" / "Lib" / "site-packages", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP
    )
    _make_fake_install(proj, bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setattr(bootstrap, "project", _FakeProject(str(proj)), raising=False)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == sdtd


def test_project_folder_layer_falls_back_to_the_folder_itself(isolated_resolver, tmp_path, monkeypatch):
    """Layer (c), last root: an install dropped straight into the project folder."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    here = _make_fake_install(proj, bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setattr(bootstrap, "project", _FakeProject(str(proj)), raising=False)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == here


def test_sys_path_layer_activates_in_place_without_touching_sys_path(isolated_resolver, tmp_path, monkeypatch):
    """Layer (e): a cuda_link that a plain ``import cuda_link`` already finds (TD Preferences
    module path, pip) is activated where it is; sys.path is left exactly as it was."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "prefs", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.syspath_prepend(site)
    bootstrap._sys_path_root.enabled = True
    _forget_loaded_cuda_link(bootstrap)
    snapshot = list(sys.path)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == site
    assert sys.path == snapshot
    origin = Path(str(getattr(sys.modules["cuda_link"], "__file__", ""))).resolve()
    assert origin.parent == (tmp_path / "prefs" / "cuda_link").resolve()
    for bare_name, target in bootstrap._ALIAS_MAP.items():
        assert sys.modules[bare_name] is sys.modules[target]


def test_same_version_already_loaded_install_is_reused_without_rival_error(isolated_resolver, tmp_path):
    """A second COMP's bootstrap run (or a second open project) must reuse a
    matching-version cuda_link already loaded from elsewhere in this process, rather
    than hard-stopping as a rival -- restarting TouchDesigner could never 'fix' this
    case, since both COMPs would resolve to the same already-loaded copy again on the
    next load."""
    bootstrap = isolated_resolver
    loaded = sys.modules["cuda_link"]  # src/cuda_link, imported by the module-scope run
    other_site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)

    assert bootstrap._bootstrap(basefolder=other_site) is True
    assert bootstrap._active is True
    assert bootstrap.last_error == ""
    assert sys.modules["cuda_link"] is loaded, "must reuse what's already loaded, not reimport from other_site"
    assert other_site not in sys.path


def test_already_loaded_reuse_ignores_a_stale_disk_read(isolated_resolver, monkeypatch):
    """The reuse shortcut must key off the loaded module's own ``__version__`` attribute
    -- what every alias in this process actually points at -- not a fresh disk read of
    its ``__init__.py``. Proven by making ``_read_version`` lie and confirming reuse
    still succeeds (fix 2)."""
    bootstrap = isolated_resolver
    loaded = sys.modules["cuda_link"]  # src/cuda_link, imported by the module-scope run
    assert loaded.__version__ == bootstrap.MIRROR_VERSION  # sanity: real repo is in sync
    monkeypatch.setattr(bootstrap, "_read_version", lambda _site: "0.0.1-does-not-match")

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap._active is True
    assert sys.modules["cuda_link"] is loaded, "must reuse the in-memory module regardless of _read_version's result"


def test_rival_install_is_refused_when_the_loaded_version_does_not_match(isolated_resolver, tmp_path, monkeypatch):
    """Only a matching-version already-loaded install is reused (see the test above); a
    mismatched version must still hard-stop as a rival -- reusing it would let two
    different-version copies of the ctypes/protocol code coexist in one process."""
    bootstrap = isolated_resolver
    loaded = sys.modules["cuda_link"]  # src/cuda_link, real on-disk version
    monkeypatch.setattr(bootstrap, "MIRROR_VERSION", "9.9.9")  # does not match what's loaded
    site = _make_fake_install(tmp_path / "lib", "9.9.9", bootstrap._ALIAS_MAP)  # matches the new stamp

    assert bootstrap._bootstrap(basefolder=site) is False
    assert bootstrap._active is False
    assert "already loaded" in bootstrap.last_error
    assert "Restart TouchDesigner" in bootstrap.last_error
    assert sys.modules["cuda_link"] is loaded, "a rival install must never replace the loaded package"
    assert site not in sys.path


def test_import_hook_ahead_of_sys_path_is_refused(isolated_resolver, tmp_path, monkeypatch):
    bootstrap = isolated_resolver
    selected = _make_fake_install(tmp_path / "selected", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    hooked = _make_fake_install(tmp_path / "hooked", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setattr(sys, "meta_path", [_RedirectingFinder(hooked), *sys.meta_path])
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=selected) is False
    assert bootstrap._active is False
    assert "import hook" in bootstrap.last_error
    assert "hooked" in bootstrap.last_error and "selected" in bootstrap.last_error
    assert "cuda_link" not in sys.modules, "a refused import must not leave the redirected package behind"
    assert selected not in sys.path


def test_version_mismatch_is_rejected_before_import(isolated_resolver, tmp_path):
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", "0.0.1", bootstrap._ALIAS_MAP)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=site) is False
    assert bootstrap._active is False
    assert "0.0.1" in bootstrap.last_error
    assert bootstrap.MIRROR_VERSION in bootstrap.last_error
    assert "cuda_link" not in sys.modules, "a mismatching install must be rejected without importing it"
    assert site not in sys.path


def test_project_folder_layer_resolves_when_project_lives_only_in_builtins(isolated_resolver, tmp_path, monkeypatch):
    """TD binds ``project`` only into the DAT module's private ``__builtins__`` dict, never
    into module globals -- unlike the fixture above, which writes ``bootstrap.project``
    straight into globals and so cannot catch a resolver that only checks globals().get().
    This must go red on a bootstrap that reads ``globals().get("project")``."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    here = _make_fake_install(proj, bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    fake_builtins = {**vars(builtins), "project": _FakeProject(str(proj))}
    monkeypatch.setitem(vars(bootstrap), "__builtins__", fake_builtins)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == here


def test_project_folder_layer_prefers_src_over_the_folder_itself(isolated_resolver, tmp_path, monkeypatch):
    """Layer (b), third root: <project.folder>/src (a repo checkout's src-layout package)
    wins over an install sitting directly in the project folder -- the gap that used to
    require a ``Libpath`` value on these example projects."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    src = _make_fake_install(proj / "src", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    _make_fake_install(proj, bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setattr(bootstrap, "project", _FakeProject(str(proj)), raising=False)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == src


def test_native_note_flags_a_src_checkout_with_no_compiled_backend(isolated_resolver, tmp_path, monkeypatch):
    """A ``src``-layout checkout never ships a built ``_native_waiter*.pyd`` (gitignored),
    so the native wait backend silently falls back to pure Python -- ``_native_note()``
    must say so instead of staying completely silent (fix 4). Resolution order itself is
    unchanged: see test_project_folder_layer_prefers_src_over_the_folder_itself above."""
    bootstrap = isolated_resolver
    src = _make_fake_install(tmp_path / "proj" / "src", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setattr(bootstrap, "resolved_site_packages", src)

    note = bootstrap._native_note()
    assert "_native_waiter" in note
    assert ".venv" in note


def test_native_note_is_empty_once_a_native_waiter_pyd_exists(isolated_resolver, tmp_path, monkeypatch):
    """Once a compiled ``_native_waiter*.pyd`` sits next to the ``src`` install (a locally
    built extension, still under a ``src`` root), the note must not fire."""
    bootstrap = isolated_resolver
    src = _make_fake_install(tmp_path / "proj" / "src", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    (Path(src) / "cuda_link" / "_native_waiter.cp311-win_amd64.pyd").write_bytes(b"")
    monkeypatch.setattr(bootstrap, "resolved_site_packages", src)

    assert bootstrap._native_note() == ""


def test_native_note_is_empty_when_the_root_is_not_named_src(isolated_resolver, tmp_path, monkeypatch):
    """A built venv install (or any root not literally named ``src``) never gets the note,
    even with no ``.pyd`` present -- it is not the source-checkout case this note targets."""
    bootstrap = isolated_resolver
    here = _make_fake_install(tmp_path / "proj", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setattr(bootstrap, "resolved_site_packages", here)

    assert bootstrap._native_note() == ""


def test_relative_env_var_is_expanded_via_tdu_from_builtins(isolated_resolver, tmp_path, monkeypatch):
    """``tdu`` is TD-only and, like ``project``, lives only in ``__builtins__`` -- a relative
    ``CUDALINK_LIB_PATH`` value must still be expanded against the project folder through it."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    site = _make_fake_install(proj / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)

    class _FakeTdu:
        def expandPath(self, text: str) -> str:
            return text if Path(text).is_absolute() else str(proj / text)

    fake_builtins = {**vars(builtins), "tdu": _FakeTdu()}
    monkeypatch.setitem(vars(bootstrap), "__builtins__", fake_builtins)
    monkeypatch.setenv("CUDALINK_LIB_PATH", "lib")
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == site


def test_expand_path_raising_leaves_last_error_set_instead_of_escaping(isolated_resolver, tmp_path, monkeypatch):
    """A broken ``tdu`` (any exception from expandPath, not just AttributeError/TypeError/
    ValueError) must not crash the bootstrap: the layer is skipped like any other
    unresolvable root, and last_error is set instead of the exception escaping."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    _make_fake_install(proj / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)

    class _BrokenTdu:
        def expandPath(self, text: str) -> str:
            raise RuntimeError("boom: tdu is broken")

    fake_builtins = {**vars(builtins), "tdu": _BrokenTdu()}
    monkeypatch.setitem(vars(bootstrap), "__builtins__", fake_builtins)
    monkeypatch.setenv("CUDALINK_LIB_PATH", "lib")
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is False
    assert bootstrap._active is False
    assert bootstrap.last_error != ""
    assert "cuda_link" not in sys.modules


def test_env_var_layer_still_resolves_an_install(isolated_resolver, tmp_path, monkeypatch):
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setenv("CUDALINK_LIB_PATH", site)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True
    assert bootstrap.resolved_site_packages == site


def test_env_var_pointing_at_the_package_dir_itself_still_resolves(isolated_resolver, tmp_path, monkeypatch):
    """CUDALINK_LIB_PATH may name the cuda_link package folder itself rather than its
    parent -- the exact mistake a ``Libpath`` value used to make on these example
    projects. There is no per-COMP parameter to catch it anymore, so the site-package
    probe itself must forgive a root one level too deep."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setenv("CUDALINK_LIB_PATH", str(Path(site) / "cuda_link"))
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == site


def test_project_folder_layer_beats_cudalink_lib_path(isolated_resolver, tmp_path, monkeypatch):
    """Cross-layer precedence: layer (b), the project folder, wins over layer (c),
    CUDALINK_LIB_PATH, when both hold a valid matching install."""
    bootstrap = isolated_resolver
    proj = tmp_path / "proj"
    project_install = _make_fake_install(proj, bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    env_install = _make_fake_install(tmp_path / "env_lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.setattr(bootstrap, "project", _FakeProject(str(proj)), raising=False)
    monkeypatch.setenv("CUDALINK_LIB_PATH", env_install)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == project_install


def test_cudalink_lib_path_beats_sys_path(isolated_resolver, tmp_path, monkeypatch):
    """Cross-layer precedence: layer (c), CUDALINK_LIB_PATH, wins over layer (d),
    whatever sys.path already provides, when both hold a valid matching install."""
    bootstrap = isolated_resolver
    env_install = _make_fake_install(tmp_path / "env_lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    sys_path_install = _make_fake_install(tmp_path / "prefs", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)
    monkeypatch.syspath_prepend(sys_path_install)
    bootstrap._sys_path_root.enabled = True
    monkeypatch.setenv("CUDALINK_LIB_PATH", env_install)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is True, bootstrap.last_error
    assert bootstrap.resolved_site_packages == env_install


def test_forgiven_drive_root_parent_is_skipped(isolated_resolver):
    """A forgiven root's parent can be a drive root ('C:\\cuda_link' -> 'C:\\', whose own
    dirname is itself) -- that is not a real parent folder to probe, and appending it
    would inject the whole drive root onto sys.path."""
    bootstrap = isolated_resolver
    candidates = bootstrap._site_package_candidates("C:\\cuda_link", forgive_package_dir=True)
    assert "C:\\" not in candidates


def test_forgiven_empty_parent_is_skipped(isolated_resolver):
    """A bare relative name ('cuda_link', no separator) has an empty dirname -- the CWD,
    not a real parent folder -- and must not be added as a candidate."""
    bootstrap = isolated_resolver
    candidates = bootstrap._site_package_candidates("cuda_link", forgive_package_dir=True)
    assert "" not in candidates


def test_forgiven_package_dir_check_is_case_insensitive(isolated_resolver, tmp_path):
    """Windows paths are case-insensitive; a mixed-case 'CUDA_Link' root must still be
    forgiven up to its parent."""
    bootstrap = isolated_resolver
    root = tmp_path / "lib" / "CUDA_Link"
    candidates = bootstrap._site_package_candidates(str(root), forgive_package_dir=True)
    assert str(root.parent) in candidates


def test_leftover_libpath_par_on_a_comp_is_ignored(isolated_resolver, tmp_path, monkeypatch):
    """Example `.toe` files saved during 1.13.0 development may still carry a ``Libpath``
    custom par (it was withdrawn before ever shipping -- see ADR-0014). The resolver must
    not read it: a par present on ``me``'s
    parent COMP, even one pointing at a matching install, must not activate library mode,
    and must never be mentioned in last_error."""
    bootstrap = isolated_resolver
    site = _make_fake_install(tmp_path / "lib", bootstrap.MIRROR_VERSION, bootstrap._ALIAS_MAP)

    class _FakePar:
        def eval(self) -> str:
            return site

    class _FakeComp:
        par = types.SimpleNamespace(Libpath=_FakePar())

        def parent(self) -> None:
            return None

    class _FakeDat:
        def parent(self) -> _FakeComp:
            return _FakeComp()

    monkeypatch.setattr(bootstrap, "me", _FakeDat(), raising=False)
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap() is False
    assert "Libpath" not in bootstrap.last_error
    assert "cuda_link" not in sys.modules


def test_every_layer_failing_reports_each_layer_and_stays_inactive(isolated_resolver, tmp_path):
    bootstrap = isolated_resolver
    _forget_loaded_cuda_link(bootstrap)

    assert bootstrap._bootstrap(basefolder=str(tmp_path / "nowhere")) is False
    assert bootstrap._active is False
    assert bootstrap.resolved_site_packages == ""
    assert bootstrap.last_error.startswith("CUDA-Link could not be resolved")
    assert "nowhere" in bootstrap.last_error
    assert "CUDALINK_LIB_PATH: not set" in bootstrap.last_error
    assert "cuda_link" not in sys.modules


def test_mirror_version_stamp_matches_package_version():
    bootstrap = _load_bootstrap()
    assert _load_sync_script().read_package_version() == bootstrap.MIRROR_VERSION, (
        "td_exporter/CUDALinkBootstrap.py::MIRROR_VERSION is stale. Run: python scripts/sync_td_wrapper.py"
    )


def test_sync_script_stamp_is_idempotent_on_current_bootstrap():
    sync = _load_sync_script()
    text = (TD_EXPORTER / "CUDALinkBootstrap.py").read_text(encoding="utf-8")
    assert sync.stamp_mirror_version(text, sync.read_package_version()) == text
    assert sync.stamp_mirror_version(text, "9.9.9") != text
