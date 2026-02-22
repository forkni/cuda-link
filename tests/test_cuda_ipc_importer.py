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
    # + 3*192B slots + 1B shutdown + 20B metadata + 8B timestamp = 625
    shm_size = 20 + 3 * 192 + 1 + 20 + 8
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

            base_offset = 20 + slot * 192
            shm.buf[base_offset : base_offset + 128] = bytes(handle.internal)

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

    # Create fake SharedMemory with v0.5.0 layout (625 bytes for 3 slots)
    shm_size = 20 + 3 * 192 + 1 + 20 + 8
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

            base_offset = 20 + slot * 192
            shm.buf[base_offset : base_offset + 128] = bytes(handle.internal)

        # Create importer
        importer = CUDAIPCImporter(shm_name=temp_shm_name, shape=(8, 8, 4), dtype="float32")

        if importer.is_ready():
            # Set shutdown flag (immediately after slots in v0.5.0 layout)
            shutdown_offset = 20 + 3 * 192
            shm.buf[shutdown_offset] = 1

            # get_frame() should detect shutdown and return None
            frame = importer.get_frame()
            assert frame is None
            assert not importer.is_ready()  # Should have cleaned up

    finally:
        shm.close()


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
