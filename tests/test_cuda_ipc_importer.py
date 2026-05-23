"""
Tests for CUDAIPCImporter (consumer side).

These tests require CUDA and either torch or numpy.
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

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


def test_connect_with_nonexistent_shm_enters_waiting_state() -> None:
    """connect() no longer raises when SHM is absent — enters reconnect-wait state."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    imp = CUDAIPCImporter(shm_name="definitely_does_not_exist_xyzzy")
    imp.connect()  # must not raise
    assert imp._importer is not None
    assert not imp._importer._initialized
    assert imp.get_frame_numpy() is None  # RECONNECTING outcome → None


def test_connect_idempotent(temp_shm_name: str) -> None:
    """Calling connect() a second time on an already-connected importer is a no-op."""
    from multiprocessing.shared_memory import SharedMemory

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Build minimal valid SHM (header only, write_idx=0 → no frame yet)
    _layout = SHMLayout(1)
    shm = SharedMemory(name=temp_shm_name, create=True, size=_layout.total_size)
    try:
        shm.buf[: _layout.total_size] = _layout.build_buffer(version=1, write_idx=0)

        imp = CUDAIPCImporter(shm_name=temp_shm_name, shape=(8, 8, 4))
        try:
            imp.connect()
        except (OSError, RuntimeError):
            pytest.skip("CUDA IPC unavailable in this environment")

        connected_state = imp._initialized
        imp.connect()  # second call — must be a no-op
        assert imp._initialized == connected_state
    finally:
        shm.close()
        shm.unlink()


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


@pytest.mark.requires_cuda
def test_cleanup_closes_handles(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Test cleanup() closes IPC handles and SharedMemory."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Create fake SharedMemory with v0.5.0 layout (3 slots)
    _layout = SHMLayout(3)
    shm = SharedMemory(name=temp_shm_name, create=True, size=_layout.total_size)
    shared_memory_cleanup.append(temp_shm_name)

    try:
        shm.buf[: _layout.total_size] = _layout.build_buffer(version=1, write_idx=0)

        # Allocate real GPU buffers and write IPC handles
        for slot in range(3):
            ptr = cuda_runtime.malloc(1024)
            handle = cuda_runtime.ipc_get_mem_handle(ptr)

            base_offset = 20 + slot * 128
            shm.buf[base_offset : base_offset + 64] = bytes(handle.internal)

        # Create importer and explicitly connect
        importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(8, 8, 4), dtype="float32")
        try:
            importer.connect()
        except (OSError, RuntimeError):
            # Clear any sticky CUDA error (e.g. error 201 from same-process IPC on Windows)
            # so subsequent tests in the same process see a clean error state.
            cuda_runtime.cudart.cudaGetLastError()
            pytest.skip("CUDA IPC connect failed in test environment")

        if importer.is_ready():
            # Cleanup
            importer.cleanup()

            # Verify cleanup state
            assert not importer.is_ready()
            assert not importer._initialized

    finally:
        shm.close()


@pytest.mark.requires_cuda
def test_shutdown_detection(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Test producer shutdown flag detection."""
    from cuda_link.cuda_ipc_importer import TORCH_AVAILABLE, CUDAIPCImporter

    if not TORCH_AVAILABLE:
        pytest.skip("torch required for this test")

    # Create fake SharedMemory with v0.5.0 layout (3 slots)
    _layout = SHMLayout(3)
    shm = SharedMemory(name=temp_shm_name, create=True, size=_layout.total_size)
    shared_memory_cleanup.append(temp_shm_name)

    try:
        shm.buf[: _layout.total_size] = _layout.build_buffer(version=1, write_idx=1)

        # Write real IPC handles
        for slot in range(3):
            ptr = cuda_runtime.malloc(1024)
            handle = cuda_runtime.ipc_get_mem_handle(ptr)

            base_offset = 20 + slot * 128
            shm.buf[base_offset : base_offset + 64] = bytes(handle.internal)

        # Create importer and explicitly connect
        importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(8, 8, 4), dtype="float32")
        try:
            importer.connect()
        except (OSError, RuntimeError):
            cuda_runtime.cudart.cudaGetLastError()
            pytest.skip("CUDA IPC connect failed in test environment")

        if importer.is_ready():
            # Set shutdown flag (immediately after slots in v0.5.0 layout)
            shutdown_offset = 20 + 3 * 128
            shm.buf[shutdown_offset] = 1

            # get_frame() should detect shutdown and return None
            frame = importer.get_frame()
            assert frame is None
            assert not importer.is_ready()  # Should have cleaned up

    finally:
        shm.close()


# ---------------------------------------------------------------------------
# Improvement 1: Stream-ordered wait in get_frame_numpy()
# ---------------------------------------------------------------------------


def _make_importer_with_mock_state(shape: tuple, dtype: str, num_slots: int = 1) -> object:
    """Build an Importer with manually-injected value-object state (no real CUDA IPC handles).

    CUDA IPC handles cannot be opened in the same process that created them, so tests
    that check routing logic inject all state via value objects and a bytearray SHM buffer.
    """
    from unittest.mock import MagicMock

    import numpy as np
    from fakes import make_fake_ipc_connection

    from cuda_link._cuda_adapters import FakeCudaAdapter
    from cuda_link._importer_port import ImportPolicy, ImportSpec
    from cuda_link.importer import Format, Importer, NumpyBuffers

    conn, mock_cuda, _ = make_fake_ipc_connection(
        num_slots=num_slots,
        dev_ptr_style="mock",
    )
    fmt = Format.from_overrides(shape, dtype)

    # Pre-build NumpyBuffers with a real numpy buffer so get_frame_numpy() skips
    # reallocation and memcpy_async receives a valid ctypes pointer.
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

    spec = ImportSpec(shm_name="mock_shm", device=0, timeout_ms=5000.0, shape=shape, dtype=dtype)
    policy = ImportPolicy(wait_spin_us=0)
    imp = Importer(spec, policy, FakeCudaAdapter())
    imp._conn = conn
    imp._format = fmt
    imp._numpy = nb
    imp._initialized = True
    return imp


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
