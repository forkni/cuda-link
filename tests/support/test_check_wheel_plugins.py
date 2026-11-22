"""Tests for scripts/check_wheel_plugins.py on synthetic wheels (no build, no TD, no GPU)."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import check_wheel_plugins as cwp  # noqa: E402

_VERSION = "2.0.0.dev0"
_PLATFORM = f"cuda_link-{_VERSION}-cp311-cp311-win_amd64.whl"
_ANY = f"cuda_link-{_VERSION}-py3-none-any.whl"


def _dll(version: str = _VERSION) -> bytes:
    """Bytes shaped like a DLL .rdata: the version as a NUL-terminated string among others."""
    return b"MZ\x00\x00CUDA-Link %s (build %s)\x00" + version.encode() + b"\x00bd1d7e1\x00"


def _wheel(tmp_path: Path, name: str, plugins: dict[str, bytes]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cuda_link/__init__.py", "")
        for file_name, data in plugins.items():
            zf.writestr(cwp.PLUGIN_PREFIX + file_name, data)
    return path


def _both(version: str = _VERSION) -> dict[str, bytes]:
    return {n: _dll(version) for n in cwp.DLL_NAMES}


def test_platform_wheel_with_both_stamped_dlls_passes(tmp_path: Path) -> None:
    assert cwp.check_wheel(_wheel(tmp_path, _PLATFORM, _both()), require_plugins=True) == []


def test_platform_wheel_without_dlls_passes_unless_required(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, _PLATFORM, {})
    assert cwp.check_wheel(wheel) == []
    assert "release wheels need both DLLs" in cwp.check_wheel(wheel, require_plugins=True)[0]


def test_any_wheel_must_not_carry_dlls(tmp_path: Path) -> None:
    assert cwp.check_wheel(_wheel(tmp_path, _ANY, {}), require_plugins=True) == []
    problems = cwp.check_wheel(_wheel(tmp_path, _ANY, _both()))
    assert problems == [f"{_ANY}: a py3-none-any wheel must not carry {cwp.PLUGIN_PREFIX}"]


def test_one_dll_missing_fails(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, _PLATFORM, {"CudaLinkOutTOP.dll": _dll()})
    assert "expected" in cwp.check_wheel(wheel)[0]


def test_extra_file_fails(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, _PLATFORM, {**_both(), "cudart64_12.dll": b"x"})
    assert "expected" in cwp.check_wheel(wheel)[0]


def test_dll_of_another_version_fails(tmp_path: Path) -> None:
    plugins = {"CudaLinkOutTOP.dll": _dll(), "CudaLinkInTOP.dll": _dll("1.14.1")}
    problems = cwp.check_wheel(_wheel(tmp_path, _PLATFORM, plugins))
    assert problems == [f"{_PLATFORM}: CudaLinkInTOP.dll does not carry version {_VERSION}"]


def test_version_must_be_a_whole_string(tmp_path: Path) -> None:
    # "2.0.0" must not match inside "2.0.0.dev0", nor the other way round.
    wheel = _wheel(tmp_path, "cuda_link-2.0.0-cp311-cp311-win_amd64.whl", _both())
    assert len(cwp.check_wheel(wheel)) == 2


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = _wheel(tmp_path, _PLATFORM, _both())
    assert cwp.main([str(good), "--require-plugins"]) == 0
    bad = _wheel(tmp_path, _ANY, _both())
    assert cwp.main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert f"[OK] {_PLATFORM}" in out and f"[FAIL] {_ANY}" in out


def test_bad_file_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a wheel file name"):
        cwp.wheel_tags(tmp_path / "cuda_link.whl")
