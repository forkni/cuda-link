"""
Tests for CUDARuntimeAPI CUDA wrapper.

These tests require a CUDA-capable GPU.
"""

import pytest


@pytest.mark.requires_cuda
def test_singleton_pattern(cuda_runtime):
    """Verify get_cuda_runtime() returns same instance."""
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

    runtime1 = get_cuda_runtime()
    runtime2 = get_cuda_runtime()

    assert runtime1 is runtime2, "Singleton pattern violated"
    assert runtime1 is cuda_runtime


@pytest.mark.requires_cuda
def test_malloc_free(cuda_runtime):
    """Test GPU memory allocation and deallocation."""
    size = 1024 * 1024  # 1 MB

    # Allocate
    ptr = cuda_runtime.malloc(size)
    assert ptr is not None
    assert ptr.value != 0

    # Free
    cuda_runtime.free(ptr)


@pytest.mark.requires_cuda
def test_malloc_zero_size(cuda_runtime):
    """Test edge case: allocating 0 bytes."""
    ptr = cuda_runtime.malloc(0)
    # Some CUDA versions return NULL for 0-byte allocation, others return valid pointer
    # Just verify it doesn't crash
    if ptr and ptr.value != 0:
        cuda_runtime.free(ptr)


@pytest.mark.requires_cuda
def test_memcpy_d2d(cuda_runtime):
    """Test device-to-device memory copy."""
    size = 1024  # 1 KB

    # Allocate two buffers
    src_ptr = cuda_runtime.malloc(size)
    dst_ptr = cuda_runtime.malloc(size)

    try:
        # Copy D2D
        cuda_runtime.memcpy(dst=dst_ptr, src=src_ptr, count=size, kind=3)  # cudaMemcpyDeviceToDevice
        cuda_runtime.synchronize()

    finally:
        cuda_runtime.free(src_ptr)
        cuda_runtime.free(dst_ptr)


@pytest.mark.requires_cuda
def test_ipc_get_mem_handle(cuda_runtime):
    """Test IPC handle creation from GPU memory."""
    size = 1024 * 1024  # 1 MB

    # Allocate
    ptr = cuda_runtime.malloc(size)

    try:
        # Get IPC handle
        handle = cuda_runtime.ipc_get_mem_handle(ptr)
        assert handle is not None
        assert len(handle.internal) == 128  # 128-byte handle

    finally:
        cuda_runtime.free(ptr)


@pytest.mark.requires_cuda
def test_ipc_event_create_destroy(cuda_runtime):
    """Test IPC event creation and destruction."""
    # Create event
    event = cuda_runtime.create_ipc_event()
    assert event is not None
    assert event.value != 0

    # Destroy event
    cuda_runtime.destroy_event(event)


@pytest.mark.requires_cuda
def test_event_record_query(cuda_runtime):
    """Test event recording and query."""
    # Create event
    event = cuda_runtime.create_ipc_event()

    try:
        # Record event
        cuda_runtime.record_event(event)

        # Query event (may not be complete yet, but should not crash)
        # Note: query_event returns bool (True if complete, False if pending)
        status = cuda_runtime.query_event(event)
        assert isinstance(status, bool), f"Expected bool, got {type(status)}"

    finally:
        cuda_runtime.destroy_event(event)


@pytest.mark.requires_cuda
def test_synchronize(cuda_runtime):
    """Test CUDA device synchronization."""
    # Should not crash
    cuda_runtime.synchronize()


@pytest.mark.requires_cuda
def test_error_checking():
    """Verify CUDAError.get_name for known error codes."""
    from cuda_link.cuda_ipc_wrapper import CUDAError

    assert CUDAError.get_name(0) == "SUCCESS"
    assert CUDAError.get_name(1) == "INVALID_VALUE"
    assert CUDAError.get_name(2) == "MEMORY_ALLOCATION"
    assert CUDAError.get_name(999) == "UNKNOWN_ERROR_999"


@pytest.mark.requires_cuda
def test_ipc_handle_structure():
    """Test cudaIpcMemHandle_t structure size."""
    from cuda_link.cuda_ipc_wrapper import cudaIpcMemHandle_t

    handle = cudaIpcMemHandle_t()
    # Should have 128-byte internal array
    assert len(handle.internal) == 128


@pytest.mark.requires_cuda
def test_ipc_event_handle_structure():
    """Test cudaIpcEventHandle_t structure size."""
    from cuda_link.cuda_ipc_wrapper import cudaIpcEventHandle_t

    handle = cudaIpcEventHandle_t()
    # Should have 64-byte reserved array
    assert len(handle.reserved) == 64
