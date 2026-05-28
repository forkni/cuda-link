"""
Contract tests for shm_protocol — the canonical SHM layout module.

Tests assert protocol invariants (round-trips, ordering guarantees, offset contracts)
using the module's own types. No local constant duplication.
"""

from __future__ import annotations

import time

import pytest

from cuda_link.shm_protocol import (
    FLAGS_BFLOAT16,
    FORMAT_KIND_SIGNED,
    FORMAT_KIND_UNSIGNED,
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
    clear_shutdown,
    publish_frame,
    read_magic,
    read_num_slots,
    read_version,
    read_write_idx,
    set_shutdown,
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
    # "float64" is not in the supported set (bfloat16 is now supported)
    with pytest.raises(KeyError):
        DtypeCodec.encode("float64")


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


# ---------------------------------------------------------------------------
# set_shutdown / clear_shutdown — new B1 helpers
# ---------------------------------------------------------------------------


def test_set_shutdown_writes_flag() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)
    assert buf[layout.shutdown_offset] == 0
    set_shutdown(buf, layout)
    assert buf[layout.shutdown_offset] == 1


def test_clear_shutdown_clears_flag() -> None:
    layout = SHMLayout(num_slots=3)
    buf = _make_buf(layout)
    buf[layout.shutdown_offset] = 1
    clear_shutdown(buf, layout)
    assert buf[layout.shutdown_offset] == 0


def test_set_shutdown_triggers_acquire_shutdown() -> None:
    """Consumer sees SHUTDOWN after set_shutdown, not a stale raw write."""
    layout = SHMLayout(num_slots=2)
    buf = _make_buf(layout)
    set_shutdown(buf, layout)
    result = acquire_slot(buf, layout, last_write_idx=0, last_version=0)
    assert result.state == SlotState.SHUTDOWN


def test_slot_offset_now_consumed_by_protocol() -> None:
    """slot_offset is the canonical slot-address accessor; verify it against raw formula."""
    for n in [2, 3, 4]:
        layout = SHMLayout(num_slots=n)
        for slot in range(n):
            expected = SHM_HEADER_SIZE + slot * SLOT_SIZE
            assert layout.slot_offset(slot) == expected


# ---------------------------------------------------------------------------
# DtypeCodec — extended 7-dtype coverage (C1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    ["float32", "float16", "bfloat16", "uint8", "uint16", "int8", "int16"],
)
def test_dtype_codec_roundtrip_all_dtypes(dtype: str) -> None:
    """encode then decode is identity for all 7 registered dtypes."""
    kind, bits, flags = DtypeCodec.encode(dtype)
    assert DtypeCodec.decode(kind, bits, flags) == dtype


def test_dtype_codec_bfloat16_vs_float16_distinct() -> None:
    """bfloat16 and float16 share (kind=Float, bits=16) but differ by FLAGS_BFLOAT16."""
    k16, b16, f16 = DtypeCodec.encode("float16")
    kbf, bbf, fbf = DtypeCodec.encode("bfloat16")
    assert (k16, b16) == (kbf, bbf)  # same kind+bits
    assert f16 == 0
    assert fbf == FLAGS_BFLOAT16
    # Codec distinguishes them
    assert DtypeCodec.decode(k16, b16, f16) == "float16"
    assert DtypeCodec.decode(kbf, bbf, fbf) == "bfloat16"


def test_dtype_codec_int_vs_uint_distinct() -> None:
    """int8 and uint8 have the same bits but different format_kind."""
    ki, bi, fi = DtypeCodec.encode("int8")
    ku, bu, fu = DtypeCodec.encode("uint8")
    assert bi == bu == 8
    assert ki == FORMAT_KIND_SIGNED
    assert ku == FORMAT_KIND_UNSIGNED
    assert DtypeCodec.decode(ki, bi, fi) == "int8"
    assert DtypeCodec.decode(ku, bu, fu) == "uint8"


@pytest.mark.parametrize(
    "dtype,expected_itemsize",
    [
        ("float32", 4),
        ("float16", 2),
        ("bfloat16", 2),
        ("uint8", 1),
        ("uint16", 2),
        ("int8", 1),
        ("int16", 2),
    ],
)
def test_dtype_codec_itemsize(dtype: str, expected_itemsize: int) -> None:
    assert DtypeCodec.itemsize(dtype) == expected_itemsize


def test_dtype_codec_supported_contains_all_seven() -> None:
    supported = DtypeCodec.supported()
    assert set(supported) == {"float32", "float16", "bfloat16", "uint8", "uint16", "int8", "int16"}


def test_dtype_codec_itemsize_unknown_raises() -> None:
    with pytest.raises(KeyError):
        DtypeCodec.itemsize("float64")


# ---------------------------------------------------------------------------
# read_version / read_num_slots — header helpers now consumed by protocol
# ---------------------------------------------------------------------------


def test_read_version_read_num_slots_helpers() -> None:
    """read_version and read_num_slots return correct values from a built buffer."""
    layout = SHMLayout(num_slots=3)
    buf = bytearray(layout.total_size)
    buf = layout.build_buffer(version=7, write_idx=0)
    mv = memoryview(buf)
    assert read_version(mv) == 7
    assert read_num_slots(mv) == 3


# ---------------------------------------------------------------------------
# Metadata.read_from now consumed for all 7 dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    ["float32", "float16", "bfloat16", "uint8", "uint16", "int8", "int16"],
)
def test_metadata_roundtrip_all_dtypes(dtype: str) -> None:
    """Metadata.read_from correctly recovers every registered dtype."""
    layout = SHMLayout(num_slots=2)
    buf = memoryview(bytearray(layout.total_size))
    kind, bits, flags = DtypeCodec.encode(dtype)
    itemsize = DtypeCodec.itemsize(dtype)
    w, h, c = 64, 48, 4
    meta_in = Metadata(
        width=w,
        height=h,
        num_comps=c,
        format_kind=kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=w * h * c * itemsize,
    )
    meta_in.pack_into(buf, layout)
    meta_out = Metadata.read_from(buf, layout)
    assert meta_out == meta_in
    # Verify round-trip through DtypeCodec
    assert DtypeCodec.decode(meta_out.format_kind, meta_out.bits_per_comp, meta_out.flags) == dtype
