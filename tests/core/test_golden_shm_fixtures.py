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


@pytest.fixture(scope="module")
def gen():  # noqa: ANN201
    """The generator module, loaded once (it loads its own copy of shm_protocol.py)."""
    spec = importlib.util.spec_from_file_location("_gen_golden_shm", _GEN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixtures_are_current(gen) -> None:  # noqa: ANN001
    assert gen.stale_fixtures() == [], "run: python scripts/gen_golden_shm.py"


def test_ring_fixture_sizes_match_layout(gen) -> None:  # noqa: ANN001
    # The byte totals themselves are pinned in test_shm_protocol.py::test_layout_total_size.
    fixtures = gen.build_fixtures()
    for n in gen.SLOT_COUNTS:
        assert len(fixtures[f"ring_n{n}.bin"]) == gen.sp.SHMLayout(n).total_size


def test_stale_fixture_is_reported(gen, tmp_path: Path) -> None:  # noqa: ANN001
    for name, data in gen.build_fixtures().items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "ring_n2.bin").write_bytes(b"stale")
    (tmp_path / "meta_formats.bin").unlink()
    assert sorted(gen.stale_fixtures(tmp_path)) == ["meta_formats.bin", "ring_n2.bin"]


def test_frozen_v05_fixture_is_kept(gen) -> None:  # noqa: ANN001
    """ring_v05_n2.bin is a frozen cuda-link 1.x segment for the C++ v0.5-message check."""
    data = (gen.FIXTURE_DIR / "ring_v05_n2.bin").read_bytes()
    assert len(data) == 305
    assert int.from_bytes(data[0:4], "little") == gen.sp.PROTOCOL_MAGIC_V05
    assert "ring_v05_n2.bin" not in gen.build_fixtures()  # never regenerated
