"""Tests for cuda_link.td_plugin: the shipped TouchDesigner TOP DLLs and the install record."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from cuda_link import td_plugin


@pytest.fixture
def fake_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / td_plugin.PLUGIN_DIR_NAME
    monkeypatch.setattr(td_plugin, "plugin_dir", lambda: d)
    return d


def test_plugin_dir_is_inside_the_package() -> None:
    assert td_plugin.plugin_dir() == Path(td_plugin.__file__).resolve().parent / "td_plugins"


def test_no_dlls_gives_none(fake_dir: Path) -> None:
    assert td_plugin.plugin_path("out") is None
    assert td_plugin.plugin_path("in") is None
    assert td_plugin.has_plugins() is False


def test_shipped_dlls_give_absolute_paths(fake_dir: Path) -> None:
    fake_dir.mkdir()
    for name in td_plugin.DLL_NAMES.values():
        (fake_dir / name).write_bytes(b"MZ")
    assert td_plugin.plugin_path("out") == fake_dir / "CudaLinkOutTOP.dll"
    assert td_plugin.plugin_path("in") == fake_dir / "CudaLinkInTOP.dll"
    assert td_plugin.has_plugins() is True


def test_one_dll_is_not_enough(fake_dir: Path) -> None:
    fake_dir.mkdir()
    (fake_dir / "CudaLinkOutTOP.dll").write_bytes(b"MZ")
    assert td_plugin.plugin_path("out") is not None
    assert td_plugin.has_plugins() is False


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="'out' or 'in'"):
        td_plugin.plugin_path("both")  # type: ignore[arg-type]


# --- the install record (VERSION.txt, ADR-0016) ---

_NOW = dt.datetime(2026, 11, 22, 10, 0, 0)


@pytest.fixture
def dll_dir(tmp_path: Path) -> Path:
    """A cpp_top build folder: both DLLs and the BUILD_INFO.txt sidecar."""
    d = tmp_path / "build"
    d.mkdir()
    for name in td_plugin.DLL_FILES:
        (d / name).write_bytes(name.encode())
    (d / td_plugin.BUILD_INFO_FILE).write_text(
        "cuda-link 2.0.0\nbuild ba5e5ab\ncuda toolkit 12.6\ntd header OP_CommonAPIVersion 1\n", encoding="utf-8"
    )
    return d


def test_record_round_trip() -> None:
    record = td_plugin.InstallRecord(
        version="2.0.0",
        build="ba5e5ab",
        protocol="v0.6 (magic 0x43495036)",
        td_header="1",
        cuda_toolkit="12.6",
        installed="2026-11-22T10:00:00 from C:/build",
        sha256={"CudaLinkOutTOP.dll": "ab" * 32, "CudaLinkInTOP.dll": "cd" * 32},
    )
    text = record.text()
    assert text.splitlines()[0] == "cuda-link 2.0.0"  # the .tox reader's contract
    assert td_plugin.InstallRecord.parse(text) == record


def test_parse_keeps_unknown_for_missing_lines() -> None:
    record = td_plugin.InstallRecord.parse("cuda-link 1.14.1\nsome later line\n")
    assert record.version == "1.14.1"
    assert (record.build, record.protocol, record.sha256) == ("unknown", "unknown", {})


def test_make_record_reads_build_info_and_hashes(dll_dir: Path) -> None:
    record = td_plugin.make_record(dll_dir, protocol="v0.6 (magic 0x43495036)", now=_NOW)
    assert (record.version, record.build, record.cuda_toolkit) == ("2.0.0", "ba5e5ab", "12.6")
    assert record.td_header == "1"
    assert record.installed == f"2026-11-22T10:00:00 from {dll_dir}"
    assert record.sha256 == {n: td_plugin.sha256_file(dll_dir / n) for n in td_plugin.DLL_FILES}


def test_make_record_explicit_values_win(dll_dir: Path) -> None:
    record = td_plugin.make_record(dll_dir, version="2.0.1", cuda_toolkit="12.8", now=_NOW)
    assert (record.version, record.cuda_toolkit, record.build) == ("2.0.1", "12.8", "ba5e5ab")


def test_make_record_without_build_info(dll_dir: Path) -> None:
    (dll_dir / td_plugin.BUILD_INFO_FILE).unlink()
    assert td_plugin.read_build_info(dll_dir) == {}
    record = td_plugin.make_record(dll_dir, now=_NOW)
    assert (record.version, record.build, record.cuda_toolkit) == ("unknown", "unknown", "unknown")


def test_install_writes_a_record_that_reads_back(dll_dir: Path, tmp_path: Path) -> None:
    plugins = tmp_path / "Plugins"
    assert td_plugin.read_record(plugins) is None
    record = td_plugin.make_record(dll_dir, protocol="v0.6 (magic 0x43495036)", td_header="1", now=_NOW)
    assert td_plugin.install(plugins, dll_dir, record, say=lambda _msg: None) == 0
    assert td_plugin.read_record(plugins) == record
    assert sorted(record.sha256) == sorted(td_plugin.DLL_FILES)


def test_status_and_uninstall(dll_dir: Path, tmp_path: Path) -> None:
    plugins = tmp_path / "Plugins"
    lines: list[str] = []
    assert td_plugin.status(plugins, say=lines.append) == 1
    td_plugin.install(plugins, dll_dir, say=lines.append)
    lines.clear()
    assert td_plugin.status(plugins, say=lines.append) == 0
    assert "cuda-link 2.0.0" in "\n".join(lines)
    assert td_plugin.uninstall(plugins, now=_NOW, say=lines.append) == 0
    assert (tmp_path / "CudaLink.removed-20261122-100000" / td_plugin.RECORD_FILE).is_file()
    assert td_plugin.read_record(plugins) is None


class _Done:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_td_running_reads_tasklist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td_plugin.os, "name", "nt")
    monkeypatch.setattr(td_plugin.subprocess, "run", lambda *a, **k: _Done("TouchDesigner.exe  1234 Console"))
    assert td_plugin.td_running() is True
    monkeypatch.setattr(td_plugin.subprocess, "run", lambda *a, **k: _Done("INFO: No tasks are running"))
    assert td_plugin.td_running() is False


def test_td_running_is_false_when_tasklist_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_a: object, **_k: object) -> None:
        raise OSError("no tasklist")

    monkeypatch.setattr(td_plugin.os, "name", "nt")
    monkeypatch.setattr(td_plugin.subprocess, "run", fail)
    assert td_plugin.td_running() is False
