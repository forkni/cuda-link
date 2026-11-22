"""Tests for scripts/install_cpp_top.py against a throwaway Plugins folder (no TD, no GPU)."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import install_cpp_top as ict  # noqa: E402

_NOW = dt.datetime(2026, 11, 17, 18, 30, 0)


@pytest.fixture
def dll_dir(tmp_path: Path) -> Path:
    d = tmp_path / "build"
    d.mkdir()
    for n in ict.DLL_NAMES:
        (d / n).write_bytes(n.encode())
    (d / "CMakeCache.txt").write_text(
        "CUDAToolkit_BIN_DIR:PATH=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def plugins(tmp_path: Path) -> Path:
    p = tmp_path / "Documents" / "Derivative" / "Plugins"
    p.mkdir(parents=True)
    return p


def test_install_copies_both_dlls_and_writes_version(dll_dir: Path, plugins: Path) -> None:
    assert ict.install(plugins, dll_dir) == 0
    dest = plugins / ict.VENDOR_FOLDER
    assert sorted(p.name for p in dest.iterdir()) == sorted((*ict.DLL_NAMES, ict.VERSION_FILE))
    text = (dest / ict.VERSION_FILE).read_text(encoding="utf-8")
    assert "protocol v0.6 (magic 0x43495036)" in text
    assert "cuda toolkit 12.8" in text
    assert "td header OP_CommonAPIVersion " in text
    assert "cuda-link unknown" not in text


def test_install_refuses_missing_dll(dll_dir: Path, plugins: Path) -> None:
    (dll_dir / "CudaLinkInTOP.dll").unlink()
    assert ict.install(plugins, dll_dir) == 2
    assert not (plugins / ict.VENDOR_FOLDER).exists()


def test_install_refuses_foreign_files_in_vendor_folder(dll_dir: Path, plugins: Path) -> None:
    dest = plugins / ict.VENDOR_FOLDER
    dest.mkdir()
    (dest / "Other.dll").write_bytes(b"x")
    assert ict.install(plugins, dll_dir) == 2


def test_install_dry_run_changes_nothing(dll_dir: Path, plugins: Path) -> None:
    assert ict.install(plugins, dll_dir, dry_run=True) == 0
    assert not (plugins / ict.VENDOR_FOLDER).exists()


def test_install_twice_replaces_in_place(dll_dir: Path, plugins: Path) -> None:
    assert ict.install(plugins, dll_dir) == 0
    (dll_dir / "CudaLinkOutTOP.dll").write_bytes(b"new build")
    assert ict.install(plugins, dll_dir) == 0
    assert (plugins / ict.VENDOR_FOLDER / "CudaLinkOutTOP.dll").read_bytes() == b"new build"


def test_uninstall_moves_folder_out_of_plugins(dll_dir: Path, plugins: Path) -> None:
    ict.install(plugins, dll_dir)
    assert ict.uninstall(plugins, now=_NOW) == 0
    assert not (plugins / ict.VENDOR_FOLDER).exists()
    moved = plugins.parent / "CudaLink.removed-20261117-183000"
    assert (moved / "CudaLinkOutTOP.dll").is_file()
    assert moved.parent != plugins  # never left inside Plugins: TD would still load it


def test_uninstall_when_absent(plugins: Path) -> None:
    assert ict.uninstall(plugins) == 1


def test_status(dll_dir: Path, plugins: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ict.status(plugins) == 1
    ict.install(plugins, dll_dir)
    stray = plugins / "CudaLink.off"
    stray.mkdir()
    (stray / "CudaLinkInTOP.dll").write_bytes(b"x")
    capsys.readouterr()
    assert ict.status(plugins) == 0
    out = capsys.readouterr().out
    assert "CudaLinkOutTOP.dll: present" in out
    assert "CudaLink.off" in out  # a renamed copy inside Plugins is reported: TD loads it


def test_main_refuses_while_td_runs(dll_dir: Path, plugins: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ict, "td_running", lambda: True)
    assert ict.main(["install", "--plugins-dir", str(plugins), "--dll-dir", str(dll_dir)]) == 3
    assert ict.main(["install", "--plugins-dir", str(plugins), "--dll-dir", str(dll_dir), "--force"]) == 0


def test_documents_dir_is_absolute() -> None:
    assert ict.documents_dir().is_absolute()


def test_build_info_wins_over_the_repo_files(dll_dir: Path) -> None:
    (dll_dir / ict.tp.BUILD_INFO_FILE).write_text(
        "cuda-link 9.9.9\nbuild abc1234\ncuda toolkit 12.6\n", encoding="utf-8"
    )
    record = ict.repo_record(dll_dir, now=_NOW)
    assert (record.version, record.build, record.cuda_toolkit) == ("9.9.9", "abc1234", "12.6")  # type: ignore[attr-defined]


def test_a_build_info_td_header_that_is_not_a_number_uses_the_repo_header(dll_dir: Path) -> None:
    # What cpp_top wrote before the BUILD_INFO parse fix: the whole escaped C++ line.
    (dll_dir / ict.tp.BUILD_INFO_FILE).write_text(
        "cuda-link 2.0.0\nbuild abc1234\ntd header OP_CommonAPIVersion const int OP_CommonAPIVersion = 1\\;\n",
        encoding="utf-8",
    )
    record = ict.repo_record(dll_dir, now=_NOW)
    assert record.td_header == "1"  # type: ignore[attr-defined]  # cpp_top/vendor/td/2025/CPlusPlus_Common.h


def _site_package(tmp_path: Path) -> Path:
    """<site-packages>/cuda_link as the cp311 wheel installs it."""
    package = tmp_path / "site-packages" / "cuda_link"
    dll_dir = package / ict.tp.PLUGIN_DIR_NAME
    dll_dir.mkdir(parents=True)
    for n in ict.DLL_NAMES:
        (dll_dir / n).write_bytes(n.encode())
    (dll_dir / ict.tp.BUILD_INFO_FILE).write_text("cuda-link 2.0.0\nbuild ba5e5ab\n", encoding="utf-8")
    (package / "shm_protocol.py").write_text("PROTOCOL_MAGIC: int = 0x43495036\n", encoding="utf-8")
    return package


def test_dll_dir_of_an_installed_wheel_uses_the_wheel_record(tmp_path: Path, plugins: Path) -> None:
    package = _site_package(tmp_path)
    assert (
        ict.main(["install", "--plugins-dir", str(plugins), "--dll-dir", str(package / "td_plugins"), "--force"]) == 0
    )
    record = ict.tp.read_record(plugins)
    assert (record.version, record.build) == ("2.0.0", "ba5e5ab")
    assert record.protocol == "v0.6 (magic 0x43495036)"
    assert record.installed.endswith(f"from {package / 'td_plugins'}")


def test_a_relative_dll_dir_is_resolved(tmp_path: Path, plugins: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # "--dll-dir ." from inside td_plugins: Path(".").name is "", so it must be resolved first.
    package = _site_package(tmp_path)
    monkeypatch.chdir(package / "td_plugins")
    assert ict.main(["install", "--plugins-dir", str(plugins), "--dll-dir", ".", "--force"]) == 0
    record = ict.tp.read_record(plugins)
    assert record.build == "ba5e5ab"
    assert record.installed.endswith(f"from {(package / 'td_plugins').resolve()}")


def test_from_wheel(tmp_path: Path, plugins: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _site_package(tmp_path)
    monkeypatch.setattr(ict, "installed_package_dir", lambda: package)
    assert ict.main(["install", "--from-wheel", "--plugins-dir", str(plugins), "--force"]) == 0
    assert ict.tp.read_record(plugins).version == "2.0.0"


def test_from_wheel_without_dlls_or_package(tmp_path: Path, plugins: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ict, "installed_package_dir", lambda: None)
    assert ict.install_from_wheel(plugins) == 2
    package = _site_package(tmp_path)
    (package / "td_plugins" / "CudaLinkInTOP.dll").unlink()
    monkeypatch.setattr(ict, "installed_package_dir", lambda: package)
    assert ict.install_from_wheel(plugins) == 2
    assert not (plugins / ict.VENDOR_FOLDER).exists()
