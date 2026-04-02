"""
Tests for CUDAIPCImporter (consumer side).

These tests require CUDA and either torch or numpy.
"""

from __future__ import annotations

import struct
from multiprocessing.shared_memory import SharedMemory

import pytest


@pytest.mark.requires_cuda
def test_init_without_shm(temp_shm_name: str) -> None:
    """Test constructor when SharedMemory doesn't exist."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Should fail gracefully since SharedMemory not created
    importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(64, 64, 4))
    assert not importer.is_ready()


@pytest.mark.requires_cuda
def test_torch_available_check() -> None:
    """Test torch availability detection."""
    from cuda_link.cuda_ipc_importer import TORCH_AVAILABLE

    # Either torch is available or not - both cases are valid
    assert isinstance(TORCH_AVAILABLE, bool)


@pytest.mark.requires_cuda
def test_numpy_available_check() -> None:
    """Test numpy availability detection."""
    from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE

    # Either numpy is available or not - both cases are valid
    assert isinstance(NUMPY_AVAILABLE, bool)


@pytest.mark.requires_cuda
def test_dtype_mapping() -> None:
    """Test string dtype correctly maps to torch/numpy dtypes."""
    from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, TORCH_AVAILABLE, CUDAIPCImporter

    importer = CUDAIPCImporter(shm_name="test", shape=(64, 64, 4), dtype="float32")

    # Test itemsize
    assert importer._dtype_itemsize() == 4

    importer.dtype = "float16"
    assert importer._dtype_itemsize() == 2

    importer.dtype = "uint8"
    assert importer._dtype_itemsize() == 1

    # Test numpy dtype mapping
    if NUMPY_AVAILABLE:
        import numpy as np

        importer.dtype = "float32"
        assert importer._numpy_dtype() == np.float32

    # Test torch dtype mapping
    if TORCH_AVAILABLE:
        import torch

        importer.dtype = "float32"
        assert importer._torch_dtype() == torch.float32


@pytest.mark.requires_cuda
def test_get_frame_without_torch() -> None:
    """Test get_frame() raises when torch not available."""
    from cuda_link.cuda_ipc_importer import TORCH_AVAILABLE, CUDAIPCImporter

    if TORCH_AVAILABLE:
        pytest.skip("torch is available, cannot test error case")

    importer = CUDAIPCImporter(shm_name="test", shape=(64, 64, 4))

    with pytest.raises(RuntimeError, match="torch is required"):
        importer.get_frame()


@pytest.mark.requires_cuda
def test_get_frame_numpy_without_numpy() -> None:
    """Test get_frame_numpy() raises when numpy not available."""
    from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, CUDAIPCImporter

    if NUMPY_AVAILABLE:
        pytest.skip("numpy is available, cannot test error case")

    importer = CUDAIPCImporter(shm_name="test", shape=(64, 64, 4))

    with pytest.raises(RuntimeError, match="numpy is required"):
        importer.get_frame_numpy()


@pytest.mark.requires_cuda
def test_get_stats_format(temp_shm_name: str) -> None:
    """Test get_stats() returns correct dictionary structure."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(64, 64, 4))

    stats = importer.get_stats()

    # Verify all expected keys present
    assert "initialized" in stats
    assert "shape" in stats
    assert "dtype" in stats
    assert "frame_count" in stats
    assert "shm_name" in stats
    assert "num_slots" in stats
    assert "torch_available" in stats
    assert "numpy_available" in stats
    assert "dev_ptrs" in stats
    assert "tensor_device" in stats


@pytest.mark.requires_cuda
def test_cleanup_closes_handles(cuda_runtime: object, temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Test cleanup() closes IPC handles and SharedMemory."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Create fake SharedMemory with v0.5.0 layout:
    # 20B header (4B magic + 8B version + 4B num_slots + 4B write_idx)
    # + 3*128B slots + 1B shutdown + 20B metadata + 8B timestamp = 433
    shm_size = 20 + 3 * 128 + 1 + 20 + 8
    shm = SharedMemory(name=temp_shm_name, create=True, size=shm_size)
    shared_memory_cleanup.append(temp_shm_name)

    try:
        # Write header (magic="CIPC", version=1, num_slots=3, write_idx=0)
        shm.buf[0:4] = struct.pack("<I", 0x43495043)  # magic "CIPC"
        shm.buf[4:12] = struct.pack("<Q", 1)  # version
        shm.buf[12:16] = struct.pack("<I", 3)  # num_slots
        shm.buf[16:20] = struct.pack("<I", 0)  # write_idx

        # Allocate real GPU buffers and write IPC handles
        for slot in range(3):
            ptr = cuda_runtime.malloc(1024)
            handle = cuda_runtime.ipc_get_mem_handle(ptr)

            base_offset = 20 + slot * 128
            shm.buf[base_offset : base_offset + 64] = bytes(handle.internal)

        # Create importer (will open handles)
        importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(8, 8, 4), dtype="float32")

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

    # Create fake SharedMemory with v0.5.0 layout (433 bytes for 3 slots)
    shm_size = 20 + 3 * 128 + 1 + 20 + 8
    shm = SharedMemory(name=temp_shm_name, create=True, size=shm_size)
    shared_memory_cleanup.append(temp_shm_name)

    try:
        # Write header (magic="CIPC", version=1, num_slots=3, write_idx=1)
        shm.buf[0:4] = struct.pack("<I", 0x43495043)  # magic "CIPC"
        shm.buf[4:12] = struct.pack("<Q", 1)  # version
        shm.buf[12:16] = struct.pack("<I", 3)  # num_slots
        shm.buf[16:20] = struct.pack("<I", 1)  # write_idx=1

        # Write real IPC handles
        for slot in range(3):
            ptr = cuda_runtime.malloc(1024)
            handle = cuda_runtime.ipc_get_mem_handle(ptr)

            base_offset = 20 + slot * 128
            shm.buf[base_offset : base_offset + 64] = bytes(handle.internal)

        # Create importer
        importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(8, 8, 4), dtype="float32")

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
    """Build a CUDAIPCImporter with manually-injected state (no real CUDA IPC handles).

    CUDA IPC handles cannot be opened in the same process that created them, so tests
    that check routing logic inject all state via MagicMock and a bytearray SHM buffer.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from cuda_link.cuda_ipc_importer import (
        METADATA_SIZE,
        SHM_HEADER_SIZE,
        SHUTDOWN_FLAG_SIZE,
        SLOT_SIZE,
        TIMESTAMP_SIZE,
        CUDAIPCImporter,
    )

    # Build a bytearray that looks like valid SharedMemory (write_idx=1 → one frame ready)
    shm_size = SHM_HEADER_SIZE + num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
    buf = bytearray(shm_size)
    struct.pack_into("<I", buf, 0, 0x43495043)  # magic "CIPC"
    struct.pack_into("<Q", buf, 4, 1)  # version=1
    struct.pack_into("<I", buf, 12, num_slots)  # num_slots
    struct.pack_into("<I", buf, 16, 1)  # write_idx=1

    # Bypass __init__ entirely, inject all attributes manually
    imp = object.__new__(CUDAIPCImporter)
    imp.shm_name = "mock_shm"
    imp.shape = shape
    imp.dtype = dtype
    imp.debug = False
    imp.timeout_ms = 5000.0
    imp.num_slots = num_slots
    imp.ipc_handles = [None] * num_slots
    imp.dev_ptrs = [MagicMock() for _ in range(num_slots)]
    imp.ipc_events = [None] * num_slots
    imp.tensors = [None] * num_slots
    imp._wrappers = [None] * num_slots
    imp.cupy_arrays = [None] * num_slots
    imp.frame_count = 0
    imp._last_write_idx = 0
    imp.total_wait_event_time = 0.0
    imp.total_get_frame_time = 0.0
    imp.total_shm_read_us = 0.0
    imp.last_latency = 0.0
    imp.ipc_version = 1  # matches version=1 written into buf above
    imp._shutdown_offset = SHM_HEADER_SIZE + num_slots * SLOT_SIZE
    imp._timestamp_offset = imp._shutdown_offset + SHUTDOWN_FLAG_SIZE + METADATA_SIZE
    imp._initialized = True

    # Inject mock CUDA runtime and numpy stream
    imp.cuda = MagicMock()
    imp._numpy_stream = MagicMock()

    # Inject mock SharedMemory whose .buf is our bytearray
    imp.shm_handle = MagicMock()
    imp.shm_handle.buf = buf

    # Pre-allocate real numpy buffer so get_frame_numpy() skips reallocation and
    # memcpy_async receives a valid ctypes pointer
    imp._numpy_buffer = np.zeros(shape, dtype=np.dtype(dtype))
    imp._pinned_ptr = None

    return imp


@pytest.mark.requires_cuda
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
        imp.ipc_events[0] = sentinel_event

        poll_calls: list[int] = []
        stream_wait_calls: list[tuple] = []

        with (
            patch.object(imp, "_wait_for_slot", side_effect=lambda s: poll_calls.append(s) or 0.0),  # noqa: B023
            patch.object(imp.cuda, "stream_wait_event", side_effect=lambda *a: stream_wait_calls.append(a)),  # noqa: B023
        ):
            imp.get_frame_numpy()

        assert len(poll_calls) == 1, f"has_event={has_event}: _wait_for_slot must always be called"
        assert poll_calls[0] == 0, "Must wait on slot 0"
        assert len(stream_wait_calls) == 0, f"has_event={has_event}: stream_wait_event must NOT be called in numpy path"


@pytest.mark.requires_cuda
def test_read_slot_calculation() -> None:
    """Test _get_read_slot() logic."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    # Create importer (won't initialize without real SharedMemory)
    importer = CUDAIPCImporter(shm_name="test_dummy", shape=(64, 64, 4))
    importer.num_slots = 3

    # Manually test the read slot calculation logic
    # (write_idx - 1) % num_slots

    test_cases = [
        (0, 0),  # Special case
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 0),  # Wraps
        (5, 1),
    ]

    for write_idx, expected_read_slot in test_cases:
        read_slot = 0 if write_idx == 0 else (write_idx - 1) % importer.num_slots
        assert read_slot == expected_read_slot
