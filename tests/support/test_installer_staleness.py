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
