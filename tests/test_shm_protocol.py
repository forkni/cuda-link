"""
Contract tests for shm_protocol — the canonical SHM layout module.

Tests assert protocol invariants (round-trips, ordering guarantees, offset contracts)
using the module's own types. No local constant duplication.
"""

from __future__ import annotations

import time

import pytest

from cuda_link.shm_protocol import (
    METADATA_SIZE,
    SHM_HEADER_SIZE,
    SHUTDOWN_FLAG_SIZE,
    SLOT_SIZE,
    DtypeCodec,
    Metadata,
    SHMLayout,
    SlotState,
    acquire_slot,
    bump_version,
    publish_frame,
    read_magic,
    read_version,
    read_write_idx,
)

# ---------------------------------------------------------------------------
# SHMLayout — offset contract tests
# ---------------------------------------------------------------------------


def _make_buf(layout: SHMLayout) -> memoryview:
    return memoryview(bytearray(layout.total_size))


def test_layout_slot_offsets() -> None:
    layout = SHMLayout(num_slots=3)
    assert layout.slot_offset(0) == SHM_HEADER_SIZE
    assert layout.slot_offset(1) == SHM_HEADER_SIZE + SLOT_SIZE
    assert layout.slot_offset(2) == SHM_HEADER_SIZE + 2 * SLOT_SIZE


def test_layout_shutdown_offset() -> None:
    for n in [2, 3, 4, 5]:
        layout = SHMLayout(num_slots=n)
        assert layout.shutdown_offset == SHM_HEADER_SIZE + n * SLOT_SIZE


def test_layout_metadata_offset() -> None:
    for n in [2, 3, 4]:
        layout = SHMLayout(num_slots=n)
        assert layout.metadata_offset == layout.shutdown_offset + SHUTDOWN_FLAG_SIZE


def test_layout_timestamp_offset() -> None:
    for n in [2, 3, 4]:
        layout = SHMLayout(num_slots=n)
        assert layout.timestamp_offset == layout.metadata_offset + METADATA_SIZE


def test_layout_total_size() -> None:
    # 3 slots: 20 + 384 + 1 + 20 + 8 = 433
    assert SHMLayout(num_slots=3).total_size == 433
    # 2 slots: 20 + 256 + 1 + 20 + 8 = 305
    assert SHMLayout(num_slots=2).total_size == 305
    # 4 slots: 20 + 512 + 1 + 20 + 8 = 561
    assert SHMLayout(num_slots=4).total_size == 561


# ---------------------------------------------------------------------------
# Header read/write contract
# ---------------------------------------------------------------------------


def test_bump_version_monotonic() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)
    for expected in range(1, 6):
        v = bump_version(buf)
        assert v == expected
        assert read_version(buf) == expected


def test_read_magic_initial_zero() -> None:
    layout = SHMLayout(num_slots=2)
    buf = _make_buf(layout)
    assert read_magic(buf) == 0  # all-zero buffer


def test_read_write_idx_initial_zero() -> None:
    layout = SHMLayout(num_slots=2)
    buf = _make_buf(layout)
    assert read_write_idx(buf) == 0


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width,height,num_comps,dtype",
    [
        (1920, 1080, 4, "float32"),
        (512, 512, 4, "float16"),
        (256, 256, 4, "uint8"),
        (1280, 720, 4, "uint16"),
    ],
)
def test_metadata_roundtrip(width: int, height: int, num_comps: int, dtype: str) -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)

    kind, bits, flags = DtypeCodec.encode(dtype)
    data_size = width * height * num_comps * (bits // 8)
    meta_in = Metadata(
        width=width,
        height=height,
        num_comps=num_comps,
        format_kind=kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=data_size,
    )
    meta_in.pack_into(buf, layout)
    meta_out = Metadata.read_from(buf, layout)

    assert meta_out == meta_in
    # Size invariant
    assert meta_out.width * meta_out.height * meta_out.num_comps * (meta_out.bits_per_comp // 8) == data_size


# ---------------------------------------------------------------------------
# DtypeCodec round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float16", "uint8", "uint16"])
def test_dtype_codec_roundtrip(dtype: str) -> None:
    kind, bits, flags = DtypeCodec.encode(dtype)
    decoded = DtypeCodec.decode(kind, bits, flags)
    assert decoded == dtype


def test_dtype_codec_unknown_decode_fallback() -> None:
    # Unknown (kind=0, bits=64, flags=0) should fallback to "float32"
    assert DtypeCodec.decode(0, 64, 0) == "float32"


def test_dtype_codec_unknown_encode_raises() -> None:
    with pytest.raises(KeyError):
        DtypeCodec.encode("bfloat16")


# ---------------------------------------------------------------------------
# publish_frame / acquire_slot round-trip
# ---------------------------------------------------------------------------


def test_publish_then_acquire_new_frame() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)

    # Simulate a frame publication
    ts = time.perf_counter()
    publish_frame(buf, layout, write_idx=1, timestamp=ts)

    result = acquire_slot(buf, layout, last_write_idx=0, last_version=0)
    assert result.state == SlotState.NEW_FRAME
    assert result.slot == 0  # (1 - 1) % 3
    assert result.write_idx == 1
    assert abs(result.timestamp - ts) < 1e-6


def test_acquire_no_frame_when_write_idx_unchanged() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)

    publish_frame(buf, layout, write_idx=2, timestamp=time.perf_counter())
    result = acquire_slot(buf, layout, last_write_idx=2, last_version=0)
    assert result.state == SlotState.NO_FRAME


def test_acquire_no_frame_when_write_idx_zero() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)
    result = acquire_slot(buf, layout, last_write_idx=0, last_version=0)
    assert result.state == SlotState.NO_FRAME


def test_acquire_shutdown_when_flag_set() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)
    buf[layout.shutdown_offset] = 1
    result = acquire_slot(buf, layout, last_write_idx=0, last_version=0)
    assert result.state == SlotState.SHUTDOWN


def test_acquire_version_changed() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)
    # Set version to 2 in the buffer; consumer thinks it's 1
    bump_version(buf)
    bump_version(buf)
    result = acquire_slot(buf, layout, last_write_idx=0, last_version=1)
    assert result.state == SlotState.VERSION_CHANGED
    assert result.new_version == 2


def test_acquire_version_zero_skips_version_check() -> None:
    """last_version=0 means consumer hasn't initialized yet; version check is skipped."""
    layout = SHMLayout(num_slots=2)
    buf = _make_buf(layout)
    bump_version(buf)
    publish_frame(buf, layout, write_idx=1, timestamp=time.perf_counter())
    # last_version=0 should not trigger VERSION_CHANGED
    result = acquire_slot(buf, layout, last_write_idx=0, last_version=0)
    assert result.state == SlotState.NEW_FRAME


def test_slot_wraps_correctly() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)

    expected_slots = [0, 1, 2, 0, 1, 2]
    last_write_idx = 0
    for i, expected_slot in enumerate(expected_slots, start=1):
        publish_frame(buf, layout, write_idx=i, timestamp=time.perf_counter())
        result = acquire_slot(buf, layout, last_write_idx=last_write_idx, last_version=0)
        assert result.state == SlotState.NEW_FRAME
        assert result.slot == expected_slot
        last_write_idx = result.write_idx


def test_publish_clears_shutdown_flag() -> None:
    layout = SHMLayout(num_slots=2)
    buf = _make_buf(layout)
    # Set shutdown, then publish a frame — flag should be cleared
    buf[layout.shutdown_offset] = 1
    publish_frame(buf, layout, write_idx=1, timestamp=time.perf_counter())
    assert buf[layout.shutdown_offset] == 0
