"""
GPU-free coverage tests for CUDARuntimeAPI's high-level wrapper methods
(check_error, event/stream/memcpy/IPC-handle wrappers, set_device/restore_context,
and friends) — cuda_ipc_wrapper.py lines 655-1387.

Background
----------
Every test builds a CUDARuntimeAPI via tests.fakes.make_bare_runtime_api(), which
bypasses __init__ (object.__new__) and injects a MagicMock cudart, so no real CUDA
DLL or GPU is touched. Each wrapper method here is a thin, mostly line-for-line
pass-through onto self.cudart.<cudaXxx>(...); these tests assert both that the
underlying cudart call happened with the expected arguments AND, where relevant,
that the Python-level return-value shaping (e.g. bool from a poll result, tuple
unpacking, ctypes struct construction) is correct.

Out-of-process ctypes out-parameters (byref(...)) are not actually written to by a
MagicMock call — the local ctypes objects (c_int(), c_void_p(), etc.) keep their
zero/None default. Tests account for this and assert against that default rather
than fabricating unrealistic written-through values.

No @pytest.mark.requires_cuda / requires_native marker needed.
"""

from __future__ import annotations

from ctypes import c_void_p
from unittest.mock import MagicMock

import pytest

# CudaIpcError is imported from the wrapper module's own resolved namespace (not
# cuda_link.cuda_runtime_types directly): cuda_ipc_wrapper.py's top-level try/except
# import prefers a bare `CUDARuntimeTypes` module when it's importable (e.g. because
# td_exporter is on pythonpath in this test environment), which binds a *different*
# class object than cuda_link.cuda_runtime_types.CudaIpcError. pytest.raises() needs
# the exact class actually raised, so we source it from cuda_ipc_wrapper itself.
from cuda_link.cuda_ipc_wrapper import CudaIpcError
from cuda_link.cuda_runtime_types import CUDAError, MemcpyKind, StreamFlags
from tests.fakes import make_bare_runtime_api

# ---------------------------------------------------------------------------
# check_error
# ---------------------------------------------------------------------------


def test_check_error_raises_on_nonzero_result() -> None:
    api = make_bare_runtime_api()
    api.cudart.cudaGetErrorString.return_value = b"invalid device pointer"

    with pytest.raises(CudaIpcError) as exc_info:
        api.check_error(CUDAError.INVALID_DEVICE_POINTER, "test_op")

    msg = str(exc_info.value)
    assert "test_op failed" in msg
    assert "invalid device pointer" in msg
    assert "INVALID_DEVICE_POINTER" in msg
    assert exc_info.value.code == CUDAError.INVALID_DEVICE_POINTER


def test_check_error_noop_on_success() -> None:
    api = make_bare_runtime_api()
    api.check_error(CUDAError.SUCCESS, "test_op")  # must not raise
    api.cudart.cudaGetErrorString.assert_not_called()


def test_check_error_unknown_error_fallback() -> None:
    api = make_bare_runtime_api()
    api.cudart.cudaGetErrorString.return_value = None

    with pytest.raises(CudaIpcError) as exc_info:
        api.check_error(4242, "weird_op")

    assert "unknown error 4242" in str(exc_info.value)


# ---------------------------------------------------------------------------
# peek_last_error / check_sticky_error
# ---------------------------------------------------------------------------


def test_peek_last_error_returns_int() -> None:
    api = make_bare_runtime_api()
    api.cudart.cudaPeekAtLastError.return_value = 17

    assert api.peek_last_error() == 17
    api.cudart.cudaPeekAtLastError.assert_called_once()


def test_check_sticky_error_noop_when_disabled() -> None:
    api = make_bare_runtime_api()
    api._sticky_check_enabled = False

    api.check_sticky_error("ctx")  # must not raise

    api.cudart.cudaPeekAtLastError.assert_not_called()


def test_check_sticky_error_noop_when_enabled_and_clean() -> None:
    api = make_bare_runtime_api()
    api._sticky_check_enabled = True
    api.cudart.cudaPeekAtLastError.return_value = CUDAError.SUCCESS

    api.check_sticky_error("ctx")  # must not raise


def test_check_sticky_error_raises_when_enabled_and_poisoned(caplog) -> None:
    import logging

    api = make_bare_runtime_api()
    api._sticky_check_enabled = True
    api.cudart.cudaPeekAtLastError.return_value = CUDAError.INVALID_VALUE
    api.cudart.cudaGetErrorString.return_value = b"invalid value"

    with (
        caplog.at_level(logging.WARNING, logger="cuda_link.cuda_ipc_wrapper"),
        pytest.raises(CudaIpcError) as exc_info,
    ):
        api.check_sticky_error("my_context")

    assert "my_context" in str(exc_info.value)
    assert "invalid value" in str(exc_info.value)
    assert exc_info.value.code == CUDAError.INVALID_VALUE
    assert any("Sticky CUDA error detected" in rec.message for rec in caplog.records)


def test_check_sticky_error_unknown_error_fallback() -> None:
    api = make_bare_runtime_api()
    api._sticky_check_enabled = True
    api.cudart.cudaPeekAtLastError.return_value = 999
    api.cudart.cudaGetErrorString.return_value = None

    with pytest.raises(CudaIpcError) as exc_info:
        api.check_sticky_error("ctx")

    assert "unknown error 999" in str(exc_info.value)


# ---------------------------------------------------------------------------
# host_register / host_unregister
# ---------------------------------------------------------------------------


def test_host_register_forwards_args() -> None:
    api = make_bare_runtime_api()
    api.host_register(0x1000, 4096, flags=1)
    api.cudart.cudaHostRegister.assert_called_once()
    call_args = api.cudart.cudaHostRegister.call_args[0]
    assert call_args[0].value == 0x1000
    assert call_args[1].value == 4096
    assert call_args[2].value == 1


def test_host_unregister_forwards_ptr() -> None:
    api = make_bare_runtime_api()
    api.host_unregister(0x2000)
    api.cudart.cudaHostUnregister.assert_called_once()
    assert api.cudart.cudaHostUnregister.call_args[0][0].value == 0x2000


# ---------------------------------------------------------------------------
# malloc / free
# ---------------------------------------------------------------------------


def test_malloc_returns_device_pointer() -> None:
    api = make_bare_runtime_api()
    result = api.malloc(1024)
    api.cudart.cudaMalloc.assert_called_once()
    assert isinstance(result, c_void_p)


def test_free_forwards_pointer() -> None:
    api = make_bare_runtime_api()
    ptr = c_void_p(0x3000)
    api.free(ptr)
    api.cudart.cudaFree.assert_called_once_with(ptr)


# ---------------------------------------------------------------------------
# _check_drv
# ---------------------------------------------------------------------------


def test_check_drv_noop_on_success() -> None:
    api = make_bare_runtime_api()
    api._drv = None  # not touched on the success path
    api._check_drv(0, "cuDeviceGet")  # must not raise


def test_check_drv_raises_with_unknown_error_fallback() -> None:
    api = make_bare_runtime_api()
    # _check_drv requires self._drv is not None on the raise path (asserted internally).
    fake_drv = MagicMock()
    fake_drv.cuGetErrorString.return_value = 1  # rc != 0 -> "unknown error" fallback
    api._drv = fake_drv

    with pytest.raises(CudaIpcError) as exc_info:
        api._check_drv(5, "someop")

    assert "someop failed" in str(exc_info.value)
    assert "unknown error 5" in str(exc_info.value)
    assert exc_info.value.code == 5


# ---------------------------------------------------------------------------
# set_device / restore_context
# ---------------------------------------------------------------------------


def test_set_device_success_pushes_retained_ctx_stack() -> None:
    api = make_bare_runtime_api()
    api._retained_ctx_stack = []
    api._drv = MagicMock()
    api._drv.cuDeviceGet.return_value = 0
    api._drv.cuCtxGetCurrent.return_value = 0
    api._drv.cuDevicePrimaryCtxRetain.return_value = 0
    api._drv.cuCtxSetCurrent.return_value = 0

    token = api.set_device(2)

    assert api._retained_ctx_stack == [2]
    assert isinstance(token, int)
    api._drv.cuDevicePrimaryCtxRetain.assert_called_once()


def test_set_device_failure_releases_retain_and_reraises() -> None:
    api = make_bare_runtime_api()
    api._retained_ctx_stack = []
    api._drv = MagicMock()
    api._drv.cuDeviceGet.return_value = 0
    api._drv.cuCtxGetCurrent.return_value = 0
    api._drv.cuDevicePrimaryCtxRetain.return_value = 0
    api._drv.cuCtxSetCurrent.return_value = 999  # fails cuCtxSetCurrent -> triggers except path

    with pytest.raises(CudaIpcError):
        api.set_device(7)

    assert api._retained_ctx_stack == []  # popped back out
    api._drv.cuDevicePrimaryCtxRelease.assert_called_once()


def test_set_device_fallback_when_no_driver_api() -> None:
    api = make_bare_runtime_api()
    api._drv = None

    token = api.set_device(3)

    assert token == 0
    # ctypes simple-value objects (c_int, etc.) don't support value-based __eq__, so
    # assert_called_once_with(c_int(3)) would always fail here; compare .value instead.
    api.cudart.cudaSetDevice.assert_called_once()
    assert api.cudart.cudaSetDevice.call_args[0][0].value == 3


def test_restore_context_releases_retained_device() -> None:
    api = make_bare_runtime_api()
    api._retained_ctx_stack = [5]
    api._drv = MagicMock()
    api._drv.cuCtxSetCurrent.return_value = 0
    api._drv.cuDevicePrimaryCtxRelease.return_value = 0

    api.restore_context(1234)

    assert api._retained_ctx_stack == []
    # ctypes simple-value objects don't support value-based __eq__; compare .value.
    api._drv.cuDevicePrimaryCtxRelease.assert_called_once()
    assert api._drv.cuDevicePrimaryCtxRelease.call_args[0][0].value == 5


def test_restore_context_noop_when_stack_empty() -> None:
    api = make_bare_runtime_api()
    api._retained_ctx_stack = []
    api._drv = MagicMock()
    api._drv.cuCtxSetCurrent.return_value = 0

    api.restore_context(0)  # must not raise

    api._drv.cuDevicePrimaryCtxRelease.assert_not_called()


def test_restore_context_noop_when_no_driver_api() -> None:
    api = make_bare_runtime_api()
    api._drv = None

    api.restore_context(0)  # must not raise


# ---------------------------------------------------------------------------
# malloc_host / free_host / memcpy
# ---------------------------------------------------------------------------


def test_malloc_host_returns_pointer() -> None:
    api = make_bare_runtime_api()
    result = api.malloc_host(2048)
    api.cudart.cudaMallocHost.assert_called_once()
    assert isinstance(result, c_void_p)


def test_free_host_forwards_pointer() -> None:
    api = make_bare_runtime_api()
    ptr = c_void_p(0x4000)
    api.free_host(ptr)
    api.cudart.cudaFreeHost.assert_called_once_with(ptr)


def test_memcpy_forwards_all_args() -> None:
    api = make_bare_runtime_api()
    dst, src = c_void_p(0x1000), c_void_p(0x2000)
    api.memcpy(dst, src, 256, MemcpyKind.DEVICE_TO_DEVICE)
    api.cudart.cudaMemcpy.assert_called_once_with(dst, src, 256, MemcpyKind.DEVICE_TO_DEVICE)


# ---------------------------------------------------------------------------
# ipc_get_mem_handle / ipc_close_mem_handle / synchronize
# ---------------------------------------------------------------------------


def test_ipc_get_mem_handle_returns_handle() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    result = api.ipc_get_mem_handle(c_void_p(0x5000))
    api.cudart.cudaIpcGetMemHandle.assert_called_once()
    assert isinstance(result, _w.cudaIpcMemHandle_t)


def test_ipc_close_mem_handle_forwards_pointer() -> None:
    api = make_bare_runtime_api()
    ptr = c_void_p(0x6000)
    api.ipc_close_mem_handle(ptr)
    api.cudart.cudaIpcCloseMemHandle.assert_called_once_with(ptr)


def test_synchronize_calls_device_synchronize() -> None:
    api = make_bare_runtime_api()
    api.synchronize()
    api.cudart.cudaDeviceSynchronize.assert_called_once()


# ---------------------------------------------------------------------------
# CUDA Event API
# ---------------------------------------------------------------------------


def test_create_ipc_event_uses_interprocess_disable_timing_flags() -> None:
    api = make_bare_runtime_api()
    api.create_ipc_event()
    args = api.cudart.cudaEventCreateWithFlags.call_args[0]
    assert args[1] == 6  # cudaEventInterprocess(4) | cudaEventDisableTiming(2)


def test_record_event_defaults_stream_to_zero() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    event = _w.CUDAEvent_t(42)
    api.record_event(event, stream=None)
    args = api.cudart.cudaEventRecord.call_args[0]
    assert args[0].value == 42
    assert args[1].value == 0


def test_record_event_forwards_explicit_stream() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    event = _w.CUDAEvent_t(1)
    stream = _w.CUDAStream_t(99)
    api.record_event(event, stream=stream)
    args = api.cudart.cudaEventRecord.call_args[0]
    assert args[1].value == 99


def test_record_event_external_uses_external_flag() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    event = _w.CUDAEvent_t(1)
    stream = _w.CUDAStream_t(2)
    api.record_event_external(event, stream)
    args = api.cudart.cudaEventRecordWithFlags.call_args[0]
    assert args[2] == _w.CUDA_EVENT_RECORD_EXTERNAL


@pytest.mark.parametrize(
    ("cuda_result", "expected"),
    [
        (CUDAError.SUCCESS, True),
        (CUDAError.NOT_READY, False),
    ],
)
def test_query_event_poll_results(cuda_result: int, expected: bool) -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    api.cudart.cudaEventQuery.return_value = cuda_result
    assert api.query_event(_w.CUDAEvent_t(1)) is expected


def test_query_event_raises_on_real_error() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    api.cudart.cudaEventQuery.return_value = CUDAError.INVALID_VALUE
    api.cudart.cudaGetErrorString.return_value = b"invalid value"

    with pytest.raises(CudaIpcError):
        api.query_event(_w.CUDAEvent_t(1))


def test_wait_event_calls_event_synchronize() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    event = _w.CUDAEvent_t(7)
    api.wait_event(event)
    api.cudart.cudaEventSynchronize.assert_called_once_with(event)


def test_ipc_get_event_handle_returns_handle() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    result = api.ipc_get_event_handle(_w.CUDAEvent_t(3))
    api.cudart.cudaIpcGetEventHandle.assert_called_once()
    assert isinstance(result, _w.cudaIpcEventHandle_t)


def test_destroy_event_forwards_event() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    event = _w.CUDAEvent_t(9)
    api.destroy_event(event)
    api.cudart.cudaEventDestroy.assert_called_once_with(event)


def test_create_timing_event_uses_flags_zero() -> None:
    api = make_bare_runtime_api()
    api.create_timing_event()
    args = api.cudart.cudaEventCreateWithFlags.call_args[0]
    assert args[1] == 0


def test_create_sync_event_uses_disable_timing_flag() -> None:
    api = make_bare_runtime_api()
    api.create_sync_event()
    args = api.cudart.cudaEventCreateWithFlags.call_args[0]
    assert args[1] == 0x02


def test_event_elapsed_time_returns_float() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    start, end = _w.CUDAEvent_t(1), _w.CUDAEvent_t(2)
    result = api.event_elapsed_time(start, end)
    api.cudart.cudaEventElapsedTime.assert_called_once()
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Device / stream API
# ---------------------------------------------------------------------------


def test_get_device_returns_int() -> None:
    api = make_bare_runtime_api()
    result = api.get_device()
    api.cudart.cudaGetDevice.assert_called_once()
    assert isinstance(result, int)


def test_create_stream_default_flags_is_non_blocking() -> None:
    api = make_bare_runtime_api()
    api.create_stream()
    args = api.cudart.cudaStreamCreateWithFlags.call_args[0]
    assert args[1] == StreamFlags.NON_BLOCKING


def test_create_stream_with_priority_queries_range_when_priority_none() -> None:
    api = make_bare_runtime_api()
    api.cudart.cudaDeviceGetStreamPriorityRange.side_effect = lambda least, greatest: 0

    api.create_stream_with_priority(priority=None)

    api.cudart.cudaDeviceGetStreamPriorityRange.assert_called_once()
    api.cudart.cudaStreamCreateWithPriority.assert_called_once()


def test_create_stream_with_priority_skips_range_query_when_explicit() -> None:
    api = make_bare_runtime_api()

    api.create_stream_with_priority(priority=-5)

    api.cudart.cudaDeviceGetStreamPriorityRange.assert_not_called()
    args = api.cudart.cudaStreamCreateWithPriority.call_args[0]
    assert args[2] == -5


def test_destroy_stream_forwards_stream() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    stream = _w.CUDAStream_t(11)
    api.destroy_stream(stream)
    api.cudart.cudaStreamDestroy.assert_called_once_with(stream)


def test_stream_wait_event_forwards_args() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    stream, event = _w.CUDAStream_t(1), _w.CUDAEvent_t(2)
    api.stream_wait_event(stream, event, flags=3)
    api.cudart.cudaStreamWaitEvent.assert_called_once_with(stream, event, 3)


def test_stream_synchronize_forwards_stream() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    stream = _w.CUDAStream_t(4)
    api.stream_synchronize(stream)
    api.cudart.cudaStreamSynchronize.assert_called_once_with(stream)


def test_memcpy_async_forwards_all_args() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    dst, src, stream = c_void_p(0x10), c_void_p(0x20), _w.CUDAStream_t(1)
    api.memcpy_async(dst, src, 64, MemcpyKind.HOST_TO_DEVICE, stream)
    api.cudart.cudaMemcpyAsync.assert_called_once_with(dst, src, 64, MemcpyKind.HOST_TO_DEVICE, stream)


def test_mem_get_info_returns_tuple() -> None:
    api = make_bare_runtime_api()
    result = api.mem_get_info()
    api.cudart.cudaMemGetInfo.assert_called_once()
    assert result == (0, 0)  # MagicMock never writes through byref; c_size_t() defaults to 0


@pytest.mark.parametrize(
    ("cuda_result", "expected"),
    [
        (CUDAError.SUCCESS, True),
        (CUDAError.NOT_READY, False),
    ],
)
def test_stream_query_poll_results(cuda_result: int, expected: bool) -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    api.cudart.cudaStreamQuery.return_value = cuda_result
    assert api.stream_query(_w.CUDAStream_t(1)) is expected


def test_stream_query_raises_on_real_error() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    api.cudart.cudaStreamQuery.return_value = CUDAError.INVALID_DEVICE
    api.cudart.cudaGetErrorString.return_value = b"invalid device"

    with pytest.raises(CudaIpcError):
        api.stream_query(_w.CUDAStream_t(1))


def test_pointer_get_attributes_returns_struct() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    result = api.pointer_get_attributes(0x7000)
    api.cudart.cudaPointerGetAttributes.assert_called_once()
    assert isinstance(result, _w.cudaPointerAttributes)


def test_device_can_access_peer_returns_bool() -> None:
    api = make_bare_runtime_api()
    result = api.device_can_access_peer(0, 1)
    api.cudart.cudaDeviceCanAccessPeer.assert_called_once()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# malloc_host_alloc / get_device_attribute
# ---------------------------------------------------------------------------


def test_malloc_host_alloc_uses_portable_flag_by_default() -> None:
    from cuda_link.cuda_runtime_types import HOST_ALLOC_PORTABLE

    api = make_bare_runtime_api()
    result = api.malloc_host_alloc(4096)
    args = api.cudart.cudaHostAlloc.call_args[0]
    assert args[2].value == HOST_ALLOC_PORTABLE
    assert isinstance(result, c_void_p)


def test_get_device_attribute_defaults_to_self_device() -> None:
    api = make_bare_runtime_api()
    api.device = 5

    api.get_device_attribute(4)

    args = api.cudart.cudaDeviceGetAttribute.call_args[0]
    assert args[2].value == 5


def test_get_device_attribute_uses_explicit_device() -> None:
    api = make_bare_runtime_api()
    api.device = 5

    api.get_device_attribute(4, device=2)

    args = api.cudart.cudaDeviceGetAttribute.call_args[0]
    assert args[2].value == 2
