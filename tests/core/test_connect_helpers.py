"""The connect helpers the Importer and the .tox Receiver share (importer.py)."""

from __future__ import annotations

import os
import struct

import pytest

from cuda_link.importer import (
    HeaderFault,
    SlotEventOpenError,
    check_header,
    open_slot,
    read_slot_handle_bytes,
)
from cuda_link.shm_protocol import (
    IPC_HANDLE_SIZE,
    NUM_SLOTS_OFFSET,
    PROTOCOL_MAGIC,
    PROTOCOL_MAGIC_V05,
    SHMLayout,
    write_header,
)

_OTHER_PID = os.getpid() + 1


def _segment(num_slots: int = 3, sender_pid: int = _OTHER_PID) -> bytearray:
    buf = bytearray(SHMLayout(num_slots).total_size)
    write_header(buf, num_slots, sender_pid)
    return buf


class _FakeCuda:
    """Records the opens; fails the event open when told to."""

    def __init__(self, event_error: Exception | None = None, mem_error: Exception | None = None) -> None:
        self.event_error = event_error
        self.mem_error = mem_error
        self.mem_opens: list[tuple[bytes, int]] = []
        self.event_opens: list[bytes] = []

    def ipc_open_mem_handle(self, handle: object, flags: int) -> str:
        if self.mem_error is not None:
            raise self.mem_error
        self.mem_opens.append((bytes(handle), flags))  # type: ignore[arg-type]
        return "dev_ptr"

    def ipc_open_event_handle(self, handle: object) -> str:
        if self.event_error is not None:
            raise self.event_error
        self.event_opens.append(bytes(handle))  # type: ignore[arg-type]
        return "event"


# --- check_header -----------------------------------------------------------------------


def test_check_header_accepts_a_fresh_segment() -> None:
    check = check_header(_segment(num_slots=4), os.getpid())

    assert check.fault is None
    assert check.magic == PROTOCOL_MAGIC
    assert check.num_slots == 4


def test_check_header_names_a_foreign_magic() -> None:
    buf = _segment()
    struct.pack_into("<I", buf, 0, PROTOCOL_MAGIC_V05)

    check = check_header(buf, os.getpid())

    assert check.fault is HeaderFault.BAD_MAGIC
    assert check.magic == PROTOCOL_MAGIC_V05


def test_check_header_rejects_a_sender_in_this_process() -> None:
    check = check_header(_segment(sender_pid=os.getpid()), os.getpid())

    assert check.fault is HeaderFault.SAME_PROCESS


def test_check_header_reports_a_buffer_too_short_for_the_magic() -> None:
    check = check_header(bytearray(2), os.getpid())

    assert check.fault is HeaderFault.UNREADABLE
    assert isinstance(check.error, struct.error)


@pytest.mark.parametrize("num_slots", [0, 11], ids=["zero", "over_max"])
def test_check_header_rejects_an_out_of_range_slot_count(num_slots: int) -> None:
    buf = _segment()
    struct.pack_into("<I", buf, NUM_SLOTS_OFFSET, num_slots)

    check = check_header(buf, os.getpid())

    assert check.fault is HeaderFault.BAD_NUM_SLOTS
    assert check.num_slots == num_slots


def test_check_header_tests_the_magic_before_the_same_process_rule() -> None:
    """A v0.5 segment has no sender pid field, so the magic must decide first."""
    buf = _segment(sender_pid=os.getpid())
    struct.pack_into("<I", buf, 0, PROTOCOL_MAGIC_V05)

    assert check_header(buf, os.getpid()).fault is HeaderFault.BAD_MAGIC


# --- slot handles -----------------------------------------------------------------------


def _write_slot(buf: bytearray, layout: SHMLayout, slot: int, mem: int, event: int) -> None:
    buf[layout.mem_handle_offset(slot) : layout.mem_handle_offset(slot) + IPC_HANDLE_SIZE] = bytes([mem]) * 64
    buf[layout.event_handle_offset(slot) : layout.event_handle_offset(slot) + IPC_HANDLE_SIZE] = bytes([event]) * 64


def test_read_slot_handle_bytes_reads_the_right_slot() -> None:
    layout = SHMLayout(3)
    buf = _segment()
    _write_slot(buf, layout, 1, mem=0xAA, event=0xBB)

    raw = read_slot_handle_bytes(buf, layout, 1)
    empty = read_slot_handle_bytes(buf, layout, 2)

    assert raw.mem == b"\xaa" * 64
    assert raw.event == b"\xbb" * 64
    assert raw.has_mem is True
    assert raw.has_event is True
    assert empty.has_mem is False
    assert empty.has_event is False


def test_open_slot_opens_memory_with_lazy_peer_access_then_the_event() -> None:
    layout = SHMLayout(2)
    buf = _segment(num_slots=2)
    _write_slot(buf, layout, 0, mem=0x11, event=0x22)
    cuda = _FakeCuda()

    ipc_handle, dev_ptr, event = open_slot(cuda, read_slot_handle_bytes(buf, layout, 0))

    assert bytes(ipc_handle) == b"\x11" * 64
    assert dev_ptr == "dev_ptr"
    assert event == "event"
    assert cuda.mem_opens == [(b"\x11" * 64, 1)]
    assert cuda.event_opens == [b"\x22" * 64]


def test_open_slot_without_an_event_handle_opens_no_event() -> None:
    layout = SHMLayout(2)
    buf = _segment(num_slots=2)
    _write_slot(buf, layout, 0, mem=0x11, event=0x00)
    cuda = _FakeCuda()

    _, _, event = open_slot(cuda, read_slot_handle_bytes(buf, layout, 0))

    assert event is None
    assert cuda.event_opens == []


def test_open_slot_event_failure_hands_the_open_memory_back() -> None:
    """The caller must be able to close the memory handle that did open."""
    layout = SHMLayout(2)
    buf = _segment(num_slots=2)
    _write_slot(buf, layout, 0, mem=0x11, event=0x22)
    cause = RuntimeError("event open failed")

    with pytest.raises(SlotEventOpenError) as raised:
        open_slot(_FakeCuda(event_error=cause), read_slot_handle_bytes(buf, layout, 0))

    assert raised.value.cause is cause
    assert raised.value.dev_ptr == "dev_ptr"
    assert not isinstance(raised.value, RuntimeError)


def test_open_slot_memory_failure_raises_the_port_error_unchanged() -> None:
    layout = SHMLayout(2)
    buf = _segment(num_slots=2)
    _write_slot(buf, layout, 0, mem=0x11, event=0x22)
    cause = RuntimeError("mem open failed")

    with pytest.raises(RuntimeError) as raised:
        open_slot(_FakeCuda(mem_error=cause), read_slot_handle_bytes(buf, layout, 0))

    assert raised.value is cause
