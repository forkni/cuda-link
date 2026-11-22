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


def _wheel(tmp_path: Path, name: str, plugins: dict[str, bytes], init: str = "") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cuda_link/__init__.py", init)
        for file_name, data in plugins.items():
            zf.writestr(cwp.PLUGIN_PREFIX + file_name, data)
    return path


def _build_info(version: str = _VERSION, td_header: str = "1") -> bytes:
    return (
        f"cuda-link {version}\nbuild bd1d7e1\ncuda toolkit 12.6\ntd header OP_CommonAPIVersion {td_header}\n"
    ).encode()


def _both(version: str = _VERSION) -> dict[str, bytes]:
    """Both DLLs and the BUILD_INFO.txt sidecar, as cpp_top writes them."""
    files = {n: _dll(version) for n in cwp.DLL_NAMES}
    files[cwp.BUILD_INFO] = _build_info(version)
    return files


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
    plugins = {**_both(), "CudaLinkInTOP.dll": _dll("1.14.1")}
    problems = cwp.check_wheel(_wheel(tmp_path, _PLATFORM, plugins))
    assert problems == [f"{_PLATFORM}: CudaLinkInTOP.dll does not carry version {_VERSION}"]


def test_version_must_be_a_whole_string(tmp_path: Path) -> None:
    # A "2.0.0.dev0" stamp in a 2.0.0 wheel fails (both DLLs and BUILD_INFO.txt) ...
    wheel = _wheel(tmp_path, "cuda_link-2.0.0-cp311-cp311-win_amd64.whl", _both())
    assert len(cwp.check_wheel(wheel)) == 3


def test_version_must_be_a_whole_string_other_way(tmp_path: Path) -> None:
    # ... and a "2.0.0" stamp in a 2.0.0.dev0 wheel fails too.
    assert len(cwp.check_wheel(_wheel(tmp_path, _PLATFORM, _both("2.0.0")))) == 3


def test_raw_package_version_wins_over_the_file_name(tmp_path: Path) -> None:
    # The file name is PEP 440 normalized (2.0.0rc1); the DLL carries __version__ as written.
    name = "cuda_link-2.0.0rc1-cp311-cp311-win_amd64.whl"
    wheel = _wheel(tmp_path, name, _both("2.0.0-rc.1"), init='__version__ = "2.0.0-rc.1"\n')
    assert cwp.check_wheel(wheel, require_plugins=True) == []
    assert cwp.check_wheel(_wheel(tmp_path, name, _both("2.0.0rc1"), init='__version__ = "2.0.0-rc.1"\n'))


def test_main_expands_a_glob_pattern(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _wheel(tmp_path, _PLATFORM, _both())
    _wheel(tmp_path, _ANY, {})
    assert cwp.main([str(tmp_path / "*.whl"), "--require-plugins"]) == 0
    out = capsys.readouterr().out
    assert f"[OK] {_PLATFORM}" in out and f"[OK] {_ANY}" in out


def test_main_reports_a_pattern_with_no_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cwp.main([str(tmp_path / "*.whl")]) == 1
    assert "[FAIL]" in capsys.readouterr().out


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


def test_missing_build_info_fails(tmp_path: Path) -> None:
    plugins = {n: _dll() for n in cwp.DLL_NAMES}
    assert "expected" in cwp.check_wheel(_wheel(tmp_path, _PLATFORM, plugins))[0]


def test_build_info_of_another_version_fails(tmp_path: Path) -> None:
    plugins = {**_both(), cwp.BUILD_INFO: _build_info("1.14.1")}
    problems = cwp.check_wheel(_wheel(tmp_path, _PLATFORM, plugins))
    assert problems == [f"{_PLATFORM}: {cwp.BUILD_INFO} is not for version {_VERSION}"]


@pytest.mark.parametrize(
    "td_header",
    ["const int OP_CommonAPIVersion = 1\\;", "unknown", ""],
    ids=["unparsed-cmake-line", "unknown", "empty"],
)
def test_build_info_without_a_numeric_td_header_fails(tmp_path: Path, td_header: str) -> None:
    # The first one is what cpp_top wrote before the parse fix: the whole escaped header line.
    plugins = {**_both(), cwp.BUILD_INFO: _build_info(td_header=td_header)}
    problems = cwp.check_wheel(_wheel(tmp_path, _PLATFORM, plugins))
    assert problems == [f"{_PLATFORM}: {cwp.BUILD_INFO} has no numeric td header line"]
