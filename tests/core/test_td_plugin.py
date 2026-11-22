"""Tests for cuda_link.td_plugin: where the shipped TouchDesigner TOP DLLs are."""

from __future__ import annotations

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
