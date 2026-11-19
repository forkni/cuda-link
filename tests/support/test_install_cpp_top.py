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
