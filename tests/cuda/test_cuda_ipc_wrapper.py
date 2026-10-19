"""
Tests for CUDARuntimeAPI CUDA wrapper.

These tests require a CUDA-capable GPU.
"""

import ctypes

import pytest


@pytest.mark.requires_cuda
def test_singleton_pattern(cuda_runtime: object) -> None:
    """Verify get_cuda_runtime() returns same instance."""
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

    runtime1 = get_cuda_runtime()
    runtime2 = get_cuda_runtime()

    assert runtime1 is runtime2, "Singleton pattern violated"
    assert runtime1 is cuda_runtime


@pytest.mark.requires_cuda
def test_malloc_free(cuda_runtime: object) -> None:
    """Test GPU memory allocation and deallocation."""
    size = 1024 * 1024  # 1 MB

    # Allocate
    ptr = cuda_runtime.malloc(size)
    assert ptr is not None
    assert ptr.value != 0

    # Free
    cuda_runtime.free(ptr)


@pytest.mark.requires_cuda
def test_malloc_zero_size(cuda_runtime: object) -> None:
    """Edge case: a 0-byte allocation succeeds and its result is safe to free.

    Some CUDA versions return NULL for a 0-byte allocation and others a valid pointer;
    cudaFree accepts either (cudaFree(NULL) is a documented no-op), so no branching.
    """
    ptr = cuda_runtime.malloc(0)
    assert isinstance(ptr, ctypes.c_void_p)
    cuda_runtime.free(ptr)
    assert cuda_runtime.peek_last_error() == 0


@pytest.mark.requires_cuda
def test_memcpy_d2d(cuda_runtime: object) -> None:
    """A device-to-device copy carries the source bytes to the destination."""
    size = 1024  # 1 KB
    pattern = bytes(i % 251 for i in range(size))
    host_src = ctypes.create_string_buffer(pattern, size)
    host_dst = ctypes.create_string_buffer(size)

    src_ptr = cuda_runtime.malloc(size)
    dst_ptr = cuda_runtime.malloc(size)

    try:
        cuda_runtime.memcpy(dst=src_ptr, src=ctypes.cast(host_src, ctypes.c_void_p), count=size, kind=1)
        cuda_runtime.memcpy(dst=dst_ptr, src=src_ptr, count=size, kind=3)  # cudaMemcpyDeviceToDevice
        cuda_runtime.synchronize()
        cuda_runtime.memcpy(dst=ctypes.cast(host_dst, ctypes.c_void_p), src=dst_ptr, count=size, kind=2)

        assert host_dst.raw == pattern

    finally:
        cuda_runtime.free(src_ptr)
        cuda_runtime.free(dst_ptr)


@pytest.mark.requires_cuda
def test_ipc_get_mem_handle(cuda_runtime: object) -> None:
    """Test IPC handle creation from GPU memory."""
    size = 1024 * 1024  # 1 MB

    # Allocate
    ptr = cuda_runtime.malloc(size)

    try:
        # Get IPC handle
        handle = cuda_runtime.ipc_get_mem_handle(ptr)
        assert handle is not None
        assert len(handle.internal) == 64  # 64-byte handle (CUDA_IPC_HANDLE_SIZE)

    finally:
        cuda_runtime.free(ptr)


@pytest.mark.requires_cuda
def test_ipc_event_create_destroy(cuda_runtime: object) -> None:
    """Test IPC event creation and destruction."""
    # Create event
    event = cuda_runtime.create_ipc_event()
    assert event is not None
    assert event.value != 0

    # Destroy event
    cuda_runtime.destroy_event(event)


@pytest.mark.requires_cuda
def test_event_record_query(cuda_runtime: object) -> None:
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
def test_synchronize(cuda_runtime: object) -> None:
    """synchronize() blocks until all previously recorded device work has completed."""
    event = cuda_runtime.create_ipc_event()
    try:
        cuda_runtime.record_event(event)
        cuda_runtime.synchronize()
        assert cuda_runtime.query_event(event) is True
    finally:
        cuda_runtime.destroy_event(event)


@pytest.mark.requires_cuda
def test_error_checking() -> None:
    """Verify CUDAError.get_name for known error codes."""
    from cuda_link.cuda_runtime_types import CUDAError

    assert CUDAError.get_name(0) == "SUCCESS"
    assert CUDAError.get_name(1) == "INVALID_VALUE"
    assert CUDAError.get_name(2) == "MEMORY_ALLOCATION"
    assert CUDAError.get_name(999) == "UNKNOWN_ERROR_999"


@pytest.mark.requires_cuda
def test_ipc_handle_structure() -> None:
    """Test cudaIpcMemHandle_t structure size."""
    from cuda_link.cuda_runtime_types import cudaIpcMemHandle_t

    handle = cudaIpcMemHandle_t()
    # Should have 64-byte internal array (CUDA_IPC_HANDLE_SIZE)
    assert len(handle.internal) == 64


@pytest.mark.requires_cuda
def test_ipc_event_handle_structure() -> None:
    """Test cudaIpcEventHandle_t structure size."""
    from cuda_link.cuda_runtime_types import cudaIpcEventHandle_t

    handle = cudaIpcEventHandle_t()
    # Should have 64-byte reserved array
    assert len(handle.reserved) == 64
