"""
Lifecycle integration tests for Importer — no GPU required.

These tests use FakeCudaAdapter and a synthetic SHM buffer to exercise the
full open→get_frame→close lifecycle without a real CUDA device.
"""

from __future__ import annotations

import struct
from ctypes import c_void_p
from unittest.mock import MagicMock, patch

import pytest

from cuda_link._cuda_adapters import FakeCudaAdapter
from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
from cuda_link.importer import Format, Importer, IPCConnection, NumpyBuffers
from cuda_link.shm_protocol import (
    METADATA_SIZE,
    SHM_HEADER_SIZE,
    SHUTDOWN_FLAG_SIZE,
    SLOT_SIZE,
    TIMESTAMP_SIZE,
    SHMLayout,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_connected_importer(
    shape: tuple = (8, 8, 4),
    dtype: str = "uint8",
    num_slots: int = 1,
    write_idx: int = 1,
    ipc_version: int = 1,
    timeout_ms: float = 5000.0,
    spin_us: int = 0,
) -> Importer:
    """Build an Importer with a fully-wired IPCConnection (no real SHM / CUDA)."""
    import numpy as np

    shm_size = SHM_HEADER_SIZE + num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
    buf = bytearray(shm_size)
    struct.pack_into("<I", buf, 0, 0x43495044)  # magic "CIPD"
    struct.pack_into("<Q", buf, 4, ipc_version)  # ipc_version
    struct.pack_into("<I", buf, 12, num_slots)  # num_slots
    struct.pack_into("<I", buf, 16, write_idx)  # write_idx

    fmt = Format.from_overrides(shape, dtype)
    layout = SHMLayout(num_slots)

    mock_cuda = MagicMock()
    mock_cuda.query_event.return_value = True  # events are always ready
    mock_shm = MagicMock()
    mock_shm.buf = buf

    conn = IPCConnection(
        cuda=mock_cuda,
        shm_handle=mock_shm,
        ipc_version=ipc_version,
        num_slots=num_slots,
        ipc_handles=[None] * num_slots,
        dev_ptrs=[c_void_p(0x1000 * (i + 1)) for i in range(num_slots)],
        ipc_events=[None] * num_slots,
        layout=layout,
        shutdown_offset=layout.shutdown_offset,
        timestamp_offset=layout.timestamp_offset,
    )

    # Pre-build NumpyBuffers so get_frame_numpy() skips the lazy allocation
    # path (which would call mock_cuda.malloc_host_alloc → MagicMock, not c_void_p).
    mock_stream = MagicMock()
    nb = NumpyBuffers(
        cuda=mock_cuda,
        fmt=fmt,
        buffer=np.zeros(shape, dtype=np.dtype(dtype)),
        pinned_ptr=None,
        host_registered_arr=None,
        pinned_memory_available=False,
        primary_stream=mock_stream,
        d2h_streams=[mock_stream],
        num_streams=1,
        chunk_plan=[],
    )

    spec = ImportSpec(shm_name="fake", device=0, timeout_ms=timeout_ms, shape=shape, dtype=dtype)
    policy = ImportPolicy(wait_spin_us=spin_us, allow_pageable_fallback=True)
    imp = Importer(spec, policy, FakeCudaAdapter())
    imp._conn = conn
    imp._format = fmt
    imp._numpy = nb
    imp._initialized = True
    return imp


# ---------------------------------------------------------------------------
# ImportOutcome coverage
# ---------------------------------------------------------------------------


def test_outcome_no_frame_when_write_idx_unchanged() -> None:
    """write_idx == last_write_idx → NO_FRAME."""
    imp = _make_connected_importer(write_idx=1)
    imp._last_write_idx = 1  # already seen this idx

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.NO_FRAME
    assert result.frame is None


def test_outcome_shutdown_closes_importer() -> None:
    """SHUTDOWN outcome triggers close() and returns SHUTDOWN."""
    imp = _make_connected_importer(write_idx=1)
    # Set shutdown flag in the SHM buffer
    shutdown_offset = imp._conn.layout.shutdown_offset
    imp._conn.shm_handle.buf[shutdown_offset] = 1

    # write_idx must advance so acquire_slot reads the slot
    imp._last_write_idx = 0

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.SHUTDOWN
    assert not imp._initialized


def test_outcome_new_frame_numpy() -> None:
    """NEW_FRAME: get_frame_numpy returns ImportResult with a numpy array."""
    import numpy as np

    shape = (4, 4, 4)
    imp = _make_connected_importer(shape=shape, dtype="uint8", write_idx=1)
    imp._last_write_idx = 0
    imp._conn.ipc_events[0] = None  # no event → cuda.synchronize() path

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.NEW_FRAME
    assert isinstance(result.frame, np.ndarray)
    assert result.frame.shape == shape


def test_outcome_reconnecting_on_version_bump() -> None:
    """VERSION_CHANGED in SHM → RECONNECTING outcome (single-tick stall)."""
    imp = _make_connected_importer(write_idx=2, ipc_version=1)
    imp._last_write_idx = 0

    # Bump ipc_version in SHM buffer to simulate producer restart
    struct.pack_into("<Q", imp._conn.shm_handle.buf, 4, 2)

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.RECONNECTING
    assert result.frame is None


def test_outcome_timeout_raises_no_exception_returns_result() -> None:
    """TimeoutError inside _wait_for_slot → TIMEOUT outcome (no exception propagated)."""
    imp = _make_connected_importer(write_idx=1, timeout_ms=1.0)
    imp._last_write_idx = 0
    # Give slot an event that we can make timeout
    mock_event = MagicMock()
    imp._conn.ipc_events[0] = mock_event
    imp._conn.cuda.query_event.return_value = False  # never ready

    with patch("cuda_link.importer.time") as mock_time:
        mock_time.perf_counter.side_effect = [0.0] + [10.0] * 20
        mock_time.sleep = MagicMock()
        result = imp.get_frame_numpy()

    assert result.outcome is ImportOutcome.TIMEOUT
    assert result.frame is None


# ---------------------------------------------------------------------------
# Idempotent close
# ---------------------------------------------------------------------------


def test_close_is_idempotent() -> None:
    """close() can be called multiple times without error."""
    imp = _make_connected_importer()
    imp.close()
    imp.close()  # second call must not raise
    assert not imp._initialized


def test_close_clears_initialized_flag() -> None:
    imp = _make_connected_importer()
    assert imp._initialized
    imp.close()
    assert not imp._initialized


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_calls_close() -> None:
    """with Importer as imp: calls close() on exit."""
    imp = _make_connected_importer()
    with imp:
        assert imp._initialized
    assert not imp._initialized


def test_context_manager_calls_close_on_exception() -> None:
    """close() is called even when the body raises."""
    imp = _make_connected_importer()
    with pytest.raises(RuntimeError, match="test error"), imp:
        raise RuntimeError("test error")
    assert not imp._initialized


# ---------------------------------------------------------------------------
# open() factory — default policy / adapter selection
# ---------------------------------------------------------------------------


def test_open_raises_on_missing_shm() -> None:
    """open() raises FileNotFoundError when SHM does not exist."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_xyzzy_12345")
    policy = ImportPolicy.for_testing()
    fake = FakeCudaAdapter()
    with pytest.raises(FileNotFoundError):
        Importer.open(spec, policy=policy, cuda=fake)


def test_open_uses_for_testing_policy() -> None:
    """ImportPolicy.for_testing() is safe as a default for unit tests."""
    pol = ImportPolicy.for_testing()
    assert pol.wait_spin_us == 0
    assert pol.allow_pageable_fallback is True


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------


def test_is_ready_when_connected() -> None:
    imp = _make_connected_importer()
    assert imp.is_ready()


def test_is_ready_false_after_close() -> None:
    imp = _make_connected_importer()
    imp.close()
    assert not imp.is_ready()


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


def test_get_stats_structure() -> None:
    imp = _make_connected_importer(shape=(8, 8, 4), dtype="float32")
    stats = imp.get_stats()

    for key in (
        "initialized",
        "shm_name",
        "shape",
        "dtype",
        "device",
        "num_slots",
        "frame_count",
        "torch_available",
        "numpy_available",
        "dev_ptrs",
        "wait_spin_hits",
        "wait_sleep_hits",
        "avg_spin_us",
        "avg_sleep_us",
    ):
        assert key in stats, f"Missing key: {key}"

    assert stats["initialized"] is True
    assert stats["shape"] == (8, 8, 4)


def test_get_stats_zero_counters_no_division_error() -> None:
    imp = _make_connected_importer()
    stats = imp.get_stats()
    assert stats["avg_spin_us"] == 0.0
    assert stats["avg_sleep_us"] == 0.0
