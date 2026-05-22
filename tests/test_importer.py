"""
Lifecycle integration tests for Importer — no GPU required.

These tests use FakeCudaAdapter and a synthetic SHM buffer to exercise the
full open→get_frame→close lifecycle without a real CUDA device.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest
from fakes import make_fake_ipc_connection

from cuda_link._cuda_adapters import FakeCudaAdapter
from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
from cuda_link.importer import Format, Importer, NumpyBuffers

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

    conn, mock_cuda, _ = make_fake_ipc_connection(
        num_slots=num_slots,
        ipc_version=ipc_version,
        write_idx=write_idx,
    )
    fmt = Format.from_overrides(shape, dtype)

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


# ---------------------------------------------------------------------------
# Reconnect-wait (PR-D)
# ---------------------------------------------------------------------------


def _reconnect_policy(**kwargs) -> ImportPolicy:
    """ImportPolicy with reconnect enabled but minimal waits for tests."""
    return ImportPolicy(
        wait_spin_us=0,
        d2h_num_streams=1,
        d2h_stream_high_priority=False,
        allow_pageable_fallback=True,
        debug=False,
        reconnect_enabled=True,
        reconnect_max_attempts=kwargs.get("reconnect_max_attempts", 20),
        reconnect_backoff_frames=kwargs.get("reconnect_backoff_frames", (1, 2, 4, 8, 16, 32, 64, 120)),
    )


def test_reconnect_open_returns_waiting_importer_on_missing_shm() -> None:
    """With reconnect_enabled=True, open() does not raise — returns Importer in waiting state."""
    from cuda_link.importer import _RetryState

    spec = ImportSpec(shm_name="definitely_does_not_exist_reconnect_test_abc")
    policy = _reconnect_policy()
    fake = FakeCudaAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)

    assert not imp._initialized
    assert imp._retry is not None
    assert isinstance(imp._retry, _RetryState)


def test_reconnect_disabled_preserves_old_raise_behavior() -> None:
    """With reconnect_enabled=False, open() raises FileNotFoundError on missing SHM."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_reconnect_disabled_test")
    policy = ImportPolicy.for_testing()  # reconnect_enabled=False
    fake = FakeCudaAdapter()

    with pytest.raises(FileNotFoundError):
        Importer.open(spec, policy=policy, cuda=fake)


def test_reconnect_get_frame_numpy_returns_reconnecting_when_not_initialized() -> None:
    """get_frame_numpy() returns RECONNECTING while waiting for producer."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_reconnect_numpy_test")
    policy = _reconnect_policy()
    fake = FakeCudaAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)
    assert not imp._initialized

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.RECONNECTING
    assert result.frame is None


def test_reconnect_backoff_caps_at_max_attempts() -> None:
    """After max_attempts, get_frame_numpy() still returns RECONNECTING (no crash)."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_backoff_test_xyzzy")
    policy = _reconnect_policy(reconnect_max_attempts=3, reconnect_backoff_frames=(1, 1, 1))
    fake = FakeCudaAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)

    # Drive enough frames to exhaust max_attempts (each attempt needs 1 frame to elapse)
    # Attempt 1 at frame 1, attempt 2 at frame 2, attempt 3 at frame 3 → then silent
    for _ in range(10):
        result = imp.get_frame_numpy()
        assert result.outcome is ImportOutcome.RECONNECTING

    # No crash — silently retrying beyond max_attempts
    assert imp._retry.connect_attempts > 3


def test_reconnect_request_immediate_skips_backoff() -> None:
    """request_immediate_reconnect() causes the next call to attempt connect immediately."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_immediate_test_xyzzy")
    # Use a large backoff so we can confirm immediate override
    policy = _reconnect_policy(reconnect_backoff_frames=(100, 200, 400))
    fake = FakeCudaAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)
    # Drive one frame to get past the first attempt and into backoff
    imp.get_frame_numpy()  # attempt 1, sets retry_interval_frames=100

    frames_before = imp._retry.frames_since_last_retry

    imp.request_immediate_reconnect()

    # After immediate reconnect, frames_since_last_retry == retry_interval_frames
    assert imp._retry.frames_since_last_retry == imp._retry.retry_interval_frames
    # And it changed from the value before
    assert imp._retry.frames_since_last_retry != frames_before or imp._retry.retry_interval_frames == 1


def test_reconnect_for_testing_policy_has_reconnect_disabled() -> None:
    """ImportPolicy.for_testing() has reconnect_enabled=False so unit tests don't wait."""
    policy = ImportPolicy.for_testing()
    assert policy.reconnect_enabled is False
