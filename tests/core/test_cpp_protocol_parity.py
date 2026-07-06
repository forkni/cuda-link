"""
T3 (minimum-viable): Python <-> C++ wire-protocol constant parity.

cpp_top/src/core/shm_layout.h is a hand-maintained "verbatim transcription" of
src/cuda_link/shm_protocol.py (see shm_layout.h's own header comment): the C++ side has
no build-time or test-time link back to the Python constants, so a constant edited on
one side and forgotten on the other silently desyncs the wire format between the Sender
(Python) and the C++ custom TOPs -- a corruption bug that would only surface as garbled
pixels or a rejected magic number at runtime, never a build failure.

This test parses shm_layout.h's `constexpr ... NAME = VALUE;` declarations with a regex
(no C++ toolchain needed) and asserts each has an equal-valued twin in
cuda_link.shm_protocol, plus cross-checks the derived SHMLayout.total_size formula
against the golden byte counts already pinned in test_shm_protocol.py
(test_layout_total_size: 2-slot=305, 3-slot=433, 4-slot=561).

BITS_PER_BYTE is deliberately C++-only (no Python twin -- Python never needs a bits/8
divisor as a named constant); it is asserted present and equal to 8, not cross-checked
against a Python name.

Skips gracefully if shm_layout.h is absent (e.g. a source checkout without the cpp_top
tree), so the default Python test suite stays green everywhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from cuda_link import shm_protocol  # noqa: E402

_SHM_LAYOUT_H = _REPO_ROOT / "cpp_top" / "src" / "core" / "shm_layout.h"

# Matches e.g. "constexpr uint32_t MAGIC_OFFSET = 0;" or "constexpr uint16_t FLAGS_BGRA = 0x0004;"
# -- deliberately excludes non-constant declarations (methods, SHMLayout's private field, etc.)
# by requiring a bare NAME = INTEGER-LITERAL; form.
_CONSTEXPR_RE = re.compile(
    r"constexpr\s+u?int(?:8|16|32|64)_t\s+(\w+)\s*=\s*(0[xX][0-9a-fA-F]+|\d+)\s*;",
)

# Constants that must match byte-for-byte between shm_layout.h and shm_protocol.py.
# (BITS_PER_BYTE is intentionally excluded -- see module docstring.)
_SHARED_CONSTANT_NAMES: tuple[str, ...] = (
    "PROTOCOL_MAGIC",
    "MAGIC_OFFSET",
    "MAGIC_SIZE",
    "VERSION_OFFSET",
    "VERSION_SIZE",
    "NUM_SLOTS_OFFSET",
    "NUM_SLOTS_SIZE",
    "WRITE_IDX_OFFSET",
    "WRITE_IDX_SIZE",
    "SHM_HEADER_SIZE",
    "SLOT_SIZE",
    "IPC_HANDLE_SIZE",
    "SHUTDOWN_FLAG_SIZE",
    "METADATA_SIZE",
    "TIMESTAMP_SIZE",
    "FORMAT_KIND_SIGNED",
    "FORMAT_KIND_UNSIGNED",
    "FORMAT_KIND_FLOAT",
    "FLAGS_BFLOAT16",
    "FLAGS_MONO_ALPHA",
    "FLAGS_BGRA",
)


def _parse_cpp_constants(text: str) -> dict[str, int]:
    return {name: int(value, 0) for name, value in _CONSTEXPR_RE.findall(text)}


@pytest.fixture(scope="module")
def cpp_constants() -> dict[str, int]:
    if not _SHM_LAYOUT_H.is_file():
        pytest.skip(f"cpp_top C++ tree not present in this checkout: {_SHM_LAYOUT_H}")
    return _parse_cpp_constants(_SHM_LAYOUT_H.read_text(encoding="utf-8"))


def test_regex_finds_the_expected_constants(cpp_constants: dict[str, int]) -> None:
    """Sanity check on the parser itself: every name we intend to cross-check, plus the
    C++-only BITS_PER_BYTE, must actually have been extracted from the header."""
    missing = [name for name in (*_SHARED_CONSTANT_NAMES, "BITS_PER_BYTE") if name not in cpp_constants]
    assert not missing, f"Regex failed to find expected constexpr declarations: {missing}"


@pytest.mark.parametrize("name", _SHARED_CONSTANT_NAMES)
def test_constant_matches_python(cpp_constants: dict[str, int], name: str) -> None:
    cpp_value = cpp_constants[name]
    py_value = getattr(shm_protocol, name)
    assert cpp_value == py_value, (
        f"{name}: C++ shm_layout.h has {cpp_value!r} but Python shm_protocol.py has {py_value!r} -- "
        "the wire protocols have desynced."
    )


def test_bits_per_byte_is_cpp_only_and_equals_8(cpp_constants: dict[str, int]) -> None:
    assert cpp_constants["BITS_PER_BYTE"] == 8
    assert not hasattr(shm_protocol, "BITS_PER_BYTE"), (
        "BITS_PER_BYTE was believed to be C++-only -- if Python now defines it too, "
        "add it to _SHARED_CONSTANT_NAMES instead of this exemption."
    )


@pytest.mark.parametrize(
    ("num_slots", "expected_total_size"),
    [(2, 305), (3, 433), (4, 561)],  # golden values pinned in tests/core/test_shm_protocol.py
)
def test_shmlayout_formula_matches_golden_and_python(
    cpp_constants: dict[str, int], num_slots: int, expected_total_size: int
) -> None:
    """Reproduce the C++ SHMLayout::total_size() formula (header + N*slot + shutdown +
    metadata + timestamp) purely from the parsed constants, and cross-check it against
    both the pinned golden byte count and the live Python SHMLayout(n).total_size."""
    header = cpp_constants["SHM_HEADER_SIZE"]
    slot = cpp_constants["SLOT_SIZE"]
    shutdown = cpp_constants["SHUTDOWN_FLAG_SIZE"]
    metadata = cpp_constants["METADATA_SIZE"]
    timestamp = cpp_constants["TIMESTAMP_SIZE"]

    cpp_formula_total = header + num_slots * slot + shutdown + metadata + timestamp
    assert cpp_formula_total == expected_total_size

    py_total = shm_protocol.SHMLayout(num_slots=num_slots).total_size
    assert py_total == expected_total_size
    assert cpp_formula_total == py_total
