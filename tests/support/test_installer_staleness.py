"""Test the installer's stale-wheel detection.

`scripts/install_td_library.py` reuses any existing `dist/*.whl` and only builds
when none is found. `_wheel_is_stale()` closes the gap where a wheel predates a
committed source change and would otherwise be silently reused — see the
"stale core wheel trap" plan. These tests exercise the mtime-comparison helpers
directly against a throwaway tmp_path tree so they don't touch the real repo's
dist/ or src/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
import install_td_library as itl  # noqa: E402


def _touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_wheel_is_stale_when_source_newer(tmp_path: Path) -> None:
    wheel = tmp_path / "cuda_link-1.0.0-py3-none-any.whl"
    src = tmp_path / "src" / "cuda_link" / "importer.py"
    now = time.time()

    _touch(wheel, now)
    _touch(src, now + 10)  # source committed after the wheel was built

    assert itl._wheel_is_stale(wheel, (str(src.relative_to(tmp_path)),), _root=tmp_path)


def test_wheel_is_not_stale_when_wheel_newer(tmp_path: Path) -> None:
    wheel = tmp_path / "cuda_link-1.0.0-py3-none-any.whl"
    src = tmp_path / "src" / "cuda_link" / "importer.py"
    now = time.time()

    _touch(src, now)
    _touch(wheel, now + 10)  # wheel rebuilt after the source change

    assert not itl._wheel_is_stale(wheel, (str(src.relative_to(tmp_path)),), _root=tmp_path)


def test_newest_source_mtime_ignores_artifact_dirs(tmp_path: Path) -> None:
    real_src = tmp_path / "src" / "cuda_link" / "importer.py"
    stale_build_copy = tmp_path / "src" / "cuda_link" / "build" / "importer.py"
    now = time.time()

    _touch(real_src, now)
    _touch(stale_build_copy, now + 1000)  # would win if artifact dirs weren't skipped

    newest = itl._newest_source_mtime(("src/cuda_link",), _root=tmp_path)
    assert newest == real_src.stat().st_mtime


def test_newest_source_mtime_missing_root_returns_zero(tmp_path: Path) -> None:
    assert itl._newest_source_mtime(("does/not/exist",), _root=tmp_path) == 0.0


def test_find_wheel_ignores_same_tag_wheel_with_wrong_version(tmp_path: Path) -> None:
    old = tmp_path / "dist" / "cuda_link-1.12.2-py3-none-any.whl"
    _touch(old, time.time() + 1000)  # newer than any 1.13.0 wheel would be

    assert itl._find_wheel("py3-none-any", "1.13.0", _root=tmp_path) is None


def test_find_wheel_returns_the_matching_version(tmp_path: Path) -> None:
    stale = tmp_path / "dist" / "cuda_link-1.12.2-py3-none-any.whl"
    wanted = tmp_path / "dist" / "cuda_link-1.13.0-py3-none-any.whl"
    _touch(stale, time.time())
    _touch(wanted, time.time())

    assert itl._find_wheel("py3-none-any", "1.13.0", _root=tmp_path) == wanted


def test_wheel_version_parses_an_arbitrary_tag(tmp_path: Path) -> None:
    """PEP 427: the version is always the second '-'-delimited field, regardless of the
    tag -- must parse a tag this project doesn't currently ship, like cp312 (fix 6)."""
    path = tmp_path / "cuda_link-1.13.0-cp312-cp312-win_amd64.whl"
    assert itl._wheel_version(path) == "1.13.0"


def test_wheel_version_still_parses_known_tags(tmp_path: Path) -> None:
    assert itl._wheel_version(tmp_path / "cuda_link-1.13.0-py3-none-any.whl") == "1.13.0"
    assert itl._wheel_version(tmp_path / f"cuda_link-1.13.0-{itl._NATIVE_WHEEL_TAG}.whl") == "1.13.0"


def test_wheel_version_returns_none_for_a_non_wheel_filename(tmp_path: Path) -> None:
    assert itl._wheel_version(tmp_path / "cuda_link-1.13.0.tar.gz") is None


def test_wheel_version_returns_none_for_an_empty_version_field(tmp_path: Path) -> None:
    """A version-less name splits to ["cuda_link", ""] -- must be None, not "", so the
    `is not None` check in resolve_wheel does not treat it as a known version."""
    assert itl._wheel_version(tmp_path / "cuda_link-.whl") is None


def test_resolve_wheel_stale_with_build_skips_download_and_rebuilds(monkeypatch: object, tmp_path: Path) -> None:
    """--build against a stale local wheel must go straight to a local rebuild, not
    re-download whatever the last GitHub Release published -- that would silently
    reinstall the same (still-stale-relative-to-src) bits --build was meant to refresh
    (fix 5)."""
    wheel = tmp_path / "dist" / "cuda_link-1.13.0-py3-none-any.whl"
    src = tmp_path / "src" / "cuda_link" / "importer.py"
    now = time.time()
    _touch(wheel, now)
    _touch(src, now + 10)  # source committed after the wheel was built -> stale

    monkeypatch.setattr(itl, "_installed_version", lambda: "1.13.0")
    monkeypatch.setattr(itl, "_CORE_SOURCE_ROOTS", (str(src.parent.relative_to(tmp_path)),))
    download_calls: list[object] = []
    monkeypatch.setattr(itl, "_download_release_wheel", lambda version, tag, dry_run: download_calls.append(1))
    rebuilt = tmp_path / "dist" / "cuda_link-1.13.0-py3-none-any-REBUILT.whl"
    build_calls: list[object] = []

    def _fake_build(tag: str, version: str, dry_run: bool) -> Path:
        build_calls.append(1)
        return rebuilt

    monkeypatch.setattr(itl, "_build_wheel", _fake_build)

    result = itl.resolve_wheel(target_version=None, override=None, dry_run=True, allow_build=True, _root=tmp_path)

    assert not download_calls, "--build must not fall back to downloading a stale wheel's replacement"
    assert build_calls
    assert result == rebuilt


def test_resolve_wheel_does_not_reuse_an_old_dist_wheel(monkeypatch: object, tmp_path: Path) -> None:
    """Only a wrong-version wheel sits in dist/; resolve_wheel must fall through
    to the download step rather than silently reusing it (the installer bug that
    let a mode-4 run install 1.12.2 while printing a 1.13.0 verify line)."""
    old = tmp_path / "dist" / "cuda_link-1.12.2-py3-none-any.whl"
    _touch(old, time.time())

    downloaded = tmp_path / "dist" / "cuda_link-1.13.0-py3-none-any.whl"
    monkeypatch.setattr(itl, "_installed_version", lambda: "1.13.0")
    monkeypatch.setattr(itl, "_download_release_wheel", lambda version, tag, dry_run: downloaded)

    result = itl.resolve_wheel(target_version=None, override=None, dry_run=True, allow_build=False, _root=tmp_path)

    assert result == downloaded


def test_resolve_wheel_no_wheel_error_names_the_exact_expected_filename(monkeypatch: object, tmp_path: Path) -> None:
    """The "no wheel available" error must name the exact file it looked for, not a glob
    that no longer matches how _find_wheel resolves a wheel (fix 8)."""
    monkeypatch.setattr(itl, "_installed_version", lambda: "1.13.0")
    monkeypatch.setattr(itl, "_download_release_wheel", lambda version, tag, dry_run: None)

    with pytest.raises(SystemExit) as excinfo:
        itl.resolve_wheel(target_version=None, override=None, dry_run=True, allow_build=False, _root=tmp_path)

    assert "cuda_link-1.13.0-py3-none-any.whl" in str(excinfo.value)
