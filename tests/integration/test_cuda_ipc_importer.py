"""
Tests for CUDAIPCImporter (consumer side).

These tests require CUDA and either torch or numpy.
"""

from __future__ import annotations

import pytest

from cuda_link.shm_protocol import SHMLayout


@pytest.mark.requires_cuda
def test_init_without_shm(temp_shm_name: str) -> None:
    """Construction is cheap and infallible; connect() does the fallible work."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # v1.5.0: __init__ does NOT auto-connect — construction never raises
    importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(64, 64, 4))
    assert not importer.is_ready()


def test_construct_does_not_raise_on_nonexistent_shm() -> None:
    """CUDAIPCImporter(shm_name=...) must never raise — connect() does."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    imp = CUDAIPCImporter(shm_name="definitely_does_not_exist_xyzzy")
    assert not imp._initialized


@pytest.mark.requires_cuda
def test_connect_with_nonexistent_shm_enters_waiting_state() -> None:
    """connect() no longer raises when SHM is absent — enters reconnect-wait state."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    imp = CUDAIPCImporter(shm_name="definitely_does_not_exist_xyzzy")
    imp.connect()  # must not raise
    assert imp._importer is not None
    assert not imp._importer._initialized
    assert imp.get_frame_numpy() is None  # RECONNECTING outcome → None


def test_connect_idempotent() -> None:
    """Calling connect() a second time on an already-connected importer is a no-op."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Inject a fake ready importer to simulate a successful first connect without
    # attempting same-process cudaIpcOpenMemHandle (which always errors on Windows).
    imp = CUDAIPCImporter(shm_name="test_shm", shape=(8, 8, 4))
    imp._importer = _make_importer_with_mock_state(shape=(8, 8, 4), dtype="float32")

    connected_state = imp._initialized
    imp.connect()  # second connect — _importer already set, must be a no-op
    assert imp._initialized == connected_state


def test_torch_available_check() -> None:
    """Test torch availability detection."""
    from cuda_link.cuda_ipc_importer import TORCH_AVAILABLE

    # Either torch is available or not - both cases are valid
    assert isinstance(TORCH_AVAILABLE, bool)


def test_numpy_available_check() -> None:
    """Test numpy availability detection."""
    from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE

    # Either numpy is available or not - both cases are valid
    assert isinstance(NUMPY_AVAILABLE, bool)


def test_get_frame_without_torch() -> None:
    """Test get_frame() raises when torch not available."""
    from cuda_link.cuda_ipc_importer import TORCH_AVAILABLE, CUDAIPCImporter

    if TORCH_AVAILABLE:
        pytest.skip("torch is available, cannot test error case")

    importer = CUDAIPCImporter(shm_name="test", shape=(64, 64, 4))

    with pytest.raises(RuntimeError, match="torch is required"):
        importer.get_frame()


def test_get_frame_numpy_without_numpy() -> None:
    """Test get_frame_numpy() raises when numpy not available."""
    from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, CUDAIPCImporter

    if NUMPY_AVAILABLE:
        pytest.skip("numpy is available, cannot test error case")

    importer = CUDAIPCImporter(shm_name="test", shape=(64, 64, 4))

    with pytest.raises(RuntimeError, match="numpy is required"):
        importer.get_frame_numpy()


def test_get_stats_before_connect() -> None:
    """get_stats() before connect() returns minimal dict with initialized=False."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    importer = CUDAIPCImporter(shm_name="test_shm", shape=(64, 64, 4))
    stats = importer.get_stats()
    assert isinstance(stats, dict)
    assert stats.get("initialized") is False


def test_cleanup_closes_handles() -> None:
    """cleanup() delegates to Importer.close() and leaves the wrapper not-ready."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Inject a fake ready importer — same-process cudaIpcOpenMemHandle cannot succeed on
    # Windows (error 201), so we bypass connect() entirely and set state directly.
    fake_importer = _make_importer_with_mock_state(shape=(8, 8, 4), dtype="float32", num_slots=3)
    imp = CUDAIPCImporter(shm_name="test_shm", shape=(8, 8, 4), dtype="float32")
    imp._importer = fake_importer

    assert imp.is_ready()
    imp.cleanup()
    assert not imp.is_ready()
    assert not imp._initialized


def test_shutdown_detection() -> None:
    """get_frame_numpy() returns None and wrapper becomes not-ready when producer sets shutdown flag."""
    from fakes import make_connected_importer

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Build a connected importer with write_idx=1 so acquire_slot reads the slot,
    # then set the shutdown flag in the existing SHM buffer.  acquire_slot() checks
    # the flag before returning any frame data, so SHUTDOWN is returned.
    fake_importer = make_connected_importer(shape=(8, 8, 4), dtype="float32", num_slots=3, write_idx=1)
    fake_importer._conn.shm_handle.buf[SHMLayout(3).shutdown_offset] = 1

    imp = CUDAIPCImporter(shm_name="test_shm", shape=(8, 8, 4), dtype="float32")
    imp._importer = fake_importer

    frame = imp.get_frame_numpy()
    assert frame is None
    assert not imp.is_ready()


# ---------------------------------------------------------------------------
# Improvement 1: Stream-ordered wait in get_frame_numpy()
# ---------------------------------------------------------------------------


def _make_importer_with_mock_state(shape: tuple, dtype: str, num_slots: int = 1) -> object:
    """Build an Importer with a fully-wired IPCConnection (no real CUDA IPC handles).

    CUDA IPC handles cannot be opened in the same process that created them; this
    helper uses ``fakes.make_connected_importer`` with mock dev-ptrs so that tests
    which patch ``imp._conn.cuda`` methods (e.g. ``stream_wait_event``) work correctly.
    """
    from fakes import make_connected_importer

    return make_connected_importer(
        shape=shape,
        dtype=dtype,
        num_slots=num_slots,
        dev_ptr_style="mock",
        allow_pageable_fallback=False,
    )


def test_get_frame_numpy_always_uses_cpu_poll() -> None:
    """get_frame_numpy() always uses _wait_for_slot (CPU poll) for the normal D2H path.

    cudaStreamWaitEvent on cross-process IPC events has high kernel-mode IPC latency
    on Windows (~100-300ms). The CPU poll path (query_event loop) is used unconditionally
    because improvement #2 guarantees the event is already signaled when write_idx is read.
    """
    from unittest.mock import patch

    for has_event in (False, True):
        imp = _make_importer_with_mock_state(shape=(8, 8, 4), dtype="float32")
        sentinel_event = object() if has_event else None
        imp._conn.ipc_events[0] = sentinel_event

        poll_calls: list[int] = []
        stream_wait_calls: list[tuple] = []

        with (
            patch.object(imp, "_wait_for_slot", side_effect=lambda s: poll_calls.append(s) or 0.0),  # noqa: B023
            patch.object(imp._conn.cuda, "stream_wait_event", side_effect=lambda *a: stream_wait_calls.append(a)),  # noqa: B023
        ):
            imp.get_frame_numpy()

        assert len(poll_calls) == 1, f"has_event={has_event}: _wait_for_slot must always be called"
        assert poll_calls[0] == 0, "Must wait on slot 0"
        assert len(stream_wait_calls) == 0, f"has_event={has_event}: stream_wait_event must NOT be called in numpy path"


def test_read_slot_calculation() -> None:
    """The read slot formula (write_idx - 1) % num_slots handles wrap-around correctly."""
    num_slots = 3

    test_cases = [
        (0, 0),  # Special case
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 0),  # Wraps
        (5, 1),
    ]

    for write_idx, expected_read_slot in test_cases:
        read_slot = 0 if write_idx == 0 else (write_idx - 1) % num_slots
        assert read_slot == expected_read_slot
