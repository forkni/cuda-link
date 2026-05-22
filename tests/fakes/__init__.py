"""
Test fixture factory for building fake IPCConnection objects.

Shared across test_importer.py, test_wait_for_slot_busywait.py, and
test_cuda_ipc_importer.py — each of which builds a different Importer
but uses the same IPCConnection scaffold (bytearray SHM + MagicMock cuda).
"""

from __future__ import annotations

import struct
from ctypes import c_void_p
from unittest.mock import MagicMock

from cuda_link.importer import IPCConnection
from cuda_link.shm_protocol import (
    METADATA_SIZE,
    SHM_HEADER_SIZE,
    SHUTDOWN_FLAG_SIZE,
    SLOT_SIZE,
    TIMESTAMP_SIZE,
    SHMLayout,
)


def make_fake_ipc_connection(
    *,
    num_slots: int = 1,
    ipc_version: int = 1,
    write_idx: int = 1,
    with_events: bool = False,
    with_shm_handle: bool = True,
    dev_ptr_style: str = "c_void_p",
) -> tuple[IPCConnection, MagicMock, MagicMock | None]:
    """Build a fake IPCConnection and companion mocks for Importer unit tests.

    Args:
        num_slots: Number of ring-buffer slots.
        ipc_version: SHM protocol version field.
        write_idx: SHM write_idx value (0 = no frames written yet).
        with_events: True → ipc_events=[MagicMock()]*N (GPU event path).
                     False → ipc_events=[None]*N (CPU poll / event-less path).
        with_shm_handle: True → mock_shm.buf = bytearray with packed header.
                         False → shm_handle=None (busywait tests that skip SHM reads).
        dev_ptr_style: "c_void_p" → c_void_p(0x1000*(i+1)) per slot.
                       "mock"     → MagicMock() per slot.
                       "none"     → None per slot.

    Returns:
        (conn, mock_cuda, mock_shm)
        - conn: IPCConnection with all fields populated.
        - mock_cuda: MagicMock with query_event.return_value=True pre-wired.
        - mock_shm: MagicMock with .buf set, or None if with_shm_handle=False.
    """
    shm_size = SHM_HEADER_SIZE + num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
    buf = bytearray(shm_size)
    struct.pack_into("<I", buf, 0, 0x43495044)  # magic "CIPD"
    struct.pack_into("<Q", buf, 4, ipc_version)  # version (uint64)
    struct.pack_into("<I", buf, 12, num_slots)  # num_slots
    struct.pack_into("<I", buf, 16, write_idx)  # write_idx

    mock_cuda = MagicMock()
    mock_cuda.query_event.return_value = True

    if with_shm_handle:
        mock_shm: MagicMock | None = MagicMock()
        mock_shm.buf = buf
    else:
        mock_shm = None

    if dev_ptr_style == "c_void_p":
        dev_ptrs = [c_void_p(0x1000 * (i + 1)) for i in range(num_slots)]
    elif dev_ptr_style == "mock":
        dev_ptrs = [MagicMock() for _ in range(num_slots)]
    elif dev_ptr_style == "none":
        dev_ptrs = [None] * num_slots
    else:
        raise ValueError(f"Unknown dev_ptr_style: {dev_ptr_style!r}")

    ipc_events = [MagicMock() for _ in range(num_slots)] if with_events else [None] * num_slots

    layout = SHMLayout(num_slots)
    conn = IPCConnection(
        cuda=mock_cuda,
        shm_handle=mock_shm,
        ipc_version=ipc_version,
        num_slots=num_slots,
        ipc_handles=[None] * num_slots,
        dev_ptrs=dev_ptrs,
        ipc_events=ipc_events,
        layout=layout,
        shutdown_offset=layout.shutdown_offset,
        timestamp_offset=layout.timestamp_offset,
    )
    return conn, mock_cuda, mock_shm
