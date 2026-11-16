"""The golden SHM fixtures for cpp_top/tests/topcore_tests.cpp match shm_protocol.py.

topcore_tests (C++, CI) byte-compares ring_writer output against these files. This test
closes the other side: a change to shm_protocol.py that alters the wire bytes leaves the
checked-in fixtures stale, and fails here until they are regenerated with
`python scripts/gen_golden_shm.py` (and the C++ side is updated to match).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_GEN = _REPO_ROOT / "scripts" / "gen_golden_shm.py"

pytestmark = pytest.mark.skipif(not _GEN.is_file(), reason="scripts/gen_golden_shm.py not present")


def _load_generator():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("_gen_golden_shm", _GEN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixtures_are_current() -> None:
    gen = _load_generator()
    assert gen.stale_fixtures() == [], "run: python scripts/gen_golden_shm.py"


def test_ring_fixture_sizes_match_pinned_totals() -> None:
    gen = _load_generator()
    fixtures = gen.build_fixtures()
    assert len(fixtures["ring_n2.bin"]) == 305
    assert len(fixtures["ring_n3.bin"]) == 433
    assert len(fixtures["ring_n4.bin"]) == 561


def test_stale_fixture_is_reported(tmp_path: Path) -> None:
    gen = _load_generator()
    for name, data in gen.build_fixtures().items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "ring_n2.bin").write_bytes(b"\x00" * 305)
    (tmp_path / "meta_formats.bin").unlink()
    assert sorted(gen.stale_fixtures(tmp_path)) == ["meta_formats.bin", "ring_n2.bin"]
