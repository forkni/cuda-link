"""Contract: the install record that cuda_link.td_plugin writes is the one the .tox reads.

Writer: ``cuda_link.td_plugin`` (used by scripts/install_cpp_top.py and
scripts/install_td_library.py). Reader: ``CUDAIPCExtension.native_tops_status``, the
``Nativetops`` row of the .tox Info page. The .tox cannot import the package (it may run
without it), so it keeps its own small reader; this test keeps the two in step (ADR-0016).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import CUDAIPCExtension
import pytest

from cuda_link import td_plugin


@pytest.fixture
def build(tmp_path: Path) -> Path:
    d = tmp_path / "build"
    d.mkdir()
    for name in td_plugin.DLL_FILES:
        (d / name).write_bytes(b"MZ")
    (d / td_plugin.BUILD_INFO_FILE).write_text("cuda-link 2.0.0\nbuild ba5e5ab\ncuda toolkit 12.6\n", encoding="utf-8")
    return d


def _installed(build: Path, plugins: Path, **record: str) -> Path:
    rec = td_plugin.make_record(build, protocol="v0.6 (magic 0x43495036)", now=dt.datetime(2026, 11, 22), **record)
    assert td_plugin.install(plugins, build, rec, say=lambda _msg: None) == 0
    return plugins / td_plugin.VENDOR_FOLDER / td_plugin.RECORD_FILE


def test_the_tox_reads_what_the_installer_writes(build: Path, tmp_path: Path) -> None:
    path = _installed(build, tmp_path / "Plugins")
    assert CUDAIPCExtension.native_tops_status("2.0.0", path) == ("2.0.0 (build ba5e5ab)", "")


def test_the_tox_warns_on_another_version(build: Path, tmp_path: Path) -> None:
    path = _installed(build, tmp_path / "Plugins", version="2.0.1")
    text, warning = CUDAIPCExtension.native_tops_status("2.0.0", path)
    assert text == "2.0.1 (build ba5e5ab) - differs from 2.0.0"
    assert "install the same version of both" in warning


def test_a_record_without_build_shows_the_cuda_toolkit(build: Path, tmp_path: Path) -> None:
    (build / td_plugin.BUILD_INFO_FILE).unlink()
    path = _installed(build, tmp_path / "Plugins", version="2.0.0", cuda_toolkit="12.8")
    assert CUDAIPCExtension.native_tops_status("2.0.0", path) == ("2.0.0 (CUDA 12.8)", "")


def test_both_sides_use_the_same_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(td_plugin, "documents_dir", lambda: tmp_path)
    monkeypatch.setattr(CUDAIPCExtension, "_documents_dir", lambda: tmp_path)
    expected = td_plugin.default_plugins_dir() / td_plugin.VENDOR_FOLDER / td_plugin.RECORD_FILE
    assert CUDAIPCExtension.native_tops_version_file() == expected
