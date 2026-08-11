"""
Tests for CUDARuntimeAPI CUDA wrapper.

These tests require a CUDA-capable GPU.

The GPU-free section at the bottom (_check_pointer_attributes_abi / check_ipc_capability)
is the exception: those build a CUDARuntimeAPI via tests.fakes.make_bare_runtime_api(),
which bypasses __init__ (object.__new__) and injects a MagicMock cudart, so no real CUDA
DLL or GPU is touched.
"""

from __future__ import annotations

from ctypes import POINTER, c_int, cast
from types import SimpleNamespace

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
    """Test edge case: allocating 0 bytes."""
    ptr = cuda_runtime.malloc(0)
    # Some CUDA versions return NULL for 0-byte allocation, others return valid pointer
    # Just verify it doesn't crash
    if ptr and ptr.value != 0:
        cuda_runtime.free(ptr)


@pytest.mark.requires_cuda
def test_memcpy_d2d(cuda_runtime: object) -> None:
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
    """Test CUDA device synchronization."""
    # Should not crash
    cuda_runtime.synchronize()


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


# ---------------------------------------------------------------------------
# GPU-free coverage: _check_pointer_attributes_abi / check_ipc_capability
# ---------------------------------------------------------------------------


def _fill_scalar_byref(ptr_arg: object, ctype: type, value: object) -> None:
    """Write `value` through a scalar ctypes byref() out-param captured by a mocked call.

    A MagicMock call never performs the write a real ctypes call would (there is no
    argtypes-driven marshalling) — the raw CArgObject from byref() is simply recorded
    as a call argument and otherwise ignored. Casting that CArgObject back to
    POINTER(ctype) and assigning .contents.value lets a side_effect simulate a real
    driver call actually populating the out-param, so a test can prove a wrapper
    method's return value is read from that out-param rather than a hardcoded
    constant. Only for scalar ctypes (c_void_p/c_int/c_float/...) — struct out-params
    (e.g. cudaIpcMemHandle_t) assign into `.contents.<field>` directly instead.
    """
    cast(ptr_arg, POINTER(ctype)).contents.value = value


def _mock_runtime_version(api: object, version: int) -> None:
    """Make api.get_runtime_version() (cudaRuntimeGetVersion) report `version`.

    Mirrors _fill_scalar_byref's byref-writeback trick, but cudaRuntimeGetVersion
    takes a single out-param (no attr/device args), so the side_effect signature
    differs from cudaDeviceGetAttribute's.
    """
    api.cudart.cudaRuntimeGetVersion.side_effect = lambda version_ptr: (
        _fill_scalar_byref(version_ptr, c_int, version) or 0
    )


def test_check_pointer_attributes_abi_warns_on_unrecognized_future_runtime(caplog: pytest.LogCaptureFixture) -> None:
    """A CUDA 14.x runtime post-dates every cudaPointerAttributes layout this binding

    has verified, so the method must warn — that warning is its entire purpose.
    """
    import logging

    from tests.fakes import make_bare_runtime_api

    api = make_bare_runtime_api()
    _mock_runtime_version(api, 14000)

    with caplog.at_level(logging.WARNING, logger="cuda_link.cuda_ipc_wrapper"):
        api._check_pointer_attributes_abi()

    assert any("newer than any" in rec.message for rec in caplog.records)


def test_check_pointer_attributes_abi_debug_logs_13x_layout(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from tests.fakes import make_bare_runtime_api

    api = make_bare_runtime_api()
    _mock_runtime_version(api, 13000)

    with caplog.at_level(logging.DEBUG, logger="cuda_link.cuda_ipc_wrapper"):
        api._check_pointer_attributes_abi()

    assert any("56 bytes" in rec.message for rec in caplog.records)
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_check_pointer_attributes_abi_debug_logs_legacy_layout(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from tests.fakes import make_bare_runtime_api

    api = make_bare_runtime_api()
    _mock_runtime_version(api, 12080)

    with caplog.at_level(logging.DEBUG, logger="cuda_link.cuda_ipc_wrapper"):
        api._check_pointer_attributes_abi()

    assert any("24 bytes" in rec.message for rec in caplog.records)


def test_check_ipc_capability_returns_none_when_not_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.cuda_ipc_wrapper as _w
    from tests.fakes import make_bare_runtime_api

    # Rebind the module-local `os` name (not the real os module — mutating the real
    # os.name process-wide breaks pathlib.Path.__new__, which pytest itself uses).
    monkeypatch.setattr(_w, "os", SimpleNamespace(name="posix"))
    api = make_bare_runtime_api()
    api.device = 0
    _mock_runtime_version(api, 12080)
    api.cudart.cudaDeviceGetAttribute.side_effect = lambda value_ptr, attr, device: (
        _fill_scalar_byref(value_ptr, c_int, 1) or 0
    )

    assert api.check_ipc_capability() is None


def test_check_ipc_capability_returns_diagnostic_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.cuda_ipc_wrapper as _w
    from tests.fakes import make_bare_runtime_api

    monkeypatch.setattr(_w, "os", SimpleNamespace(name="nt"))
    api = make_bare_runtime_api()
    api.device = 0
    _mock_runtime_version(api, 12080)
    api.cudart.cudaDeviceGetAttribute.side_effect = lambda value_ptr, attr, device: (
        _fill_scalar_byref(value_ptr, c_int, 1) or 0
    )

    msg = api.check_ipc_capability()
    assert msg is not None
    assert "IPC" in msg


def test_check_ipc_capability_raises_when_support_is_zero() -> None:
    from cuda_link.cuda_ipc_wrapper import CudaIpcError
    from tests.fakes import make_bare_runtime_api

    api = make_bare_runtime_api()
    api.device = 0
    _mock_runtime_version(api, 12080)
    api.cudart.cudaDeviceGetAttribute.side_effect = lambda value_ptr, attr, device: (
        _fill_scalar_byref(value_ptr, c_int, 0) or 0
    )

    with pytest.raises(CudaIpcError, match="cudaDevAttrIpcEventSupport=0"):
        api.check_ipc_capability()


def test_check_ipc_capability_skips_query_on_pre_12_runtime() -> None:
    """CUDA 11.x (e.g. TouchDesigner's cudart64_110.dll) predates attribute 125

    (cudaDevAttrIpcEventSupport, added in CUDA 12.0). Querying it on an 11.x
    runtime returns an invalid-attribute error; the probe must skip the query
    entirely rather than let that error abort Exporter.open().
    """
    from tests.fakes import make_bare_runtime_api

    api = make_bare_runtime_api()
    api.device = 0
    _mock_runtime_version(api, 11080)  # CUDA 11.8 — last 11.x release

    msg = api.check_ipc_capability()

    assert msg is not None
    assert "11080" in msg
    api.cudart.cudaDeviceGetAttribute.assert_not_called()


def test_check_ipc_capability_degrades_on_query_failure() -> None:
    """An unexpected query failure (e.g. attribute unsupported on this driver)

    must degrade to a diagnostic string rather than propagate — the probe's
    sole job is diagnostics, and only an explicit support==0 result should be
    able to raise.
    """
    from cuda_link.cuda_ipc_wrapper import CudaIpcError
    from tests.fakes import make_bare_runtime_api

    api = make_bare_runtime_api()
    api.device = 0
    _mock_runtime_version(api, 12080)
    api.cudart.cudaDeviceGetAttribute.side_effect = CudaIpcError("cudaDeviceGetAttribute failed: boom")

    msg = api.check_ipc_capability()

    assert msg is not None
    assert "could not query" in msg.lower()


def test_check_ipc_capability_survives_runtime_version_failure() -> None:
    """get_runtime_version() failing must not abort the probe — it degrades to

    version 0, which then takes the pre-12.0 skip path. This probe is diagnostic
    and must never be able to abort Exporter.open().
    """
    from tests.fakes import make_bare_runtime_api

    api = make_bare_runtime_api()
    api.device = 0
    api.cudart.cudaRuntimeGetVersion.side_effect = RuntimeError("cudart not loaded")

    msg = api.check_ipc_capability()

    assert msg is not None
    assert "runtime version 0" in msg
    api.cudart.cudaDeviceGetAttribute.assert_not_called()


def test_check_ipc_capability_uses_explicit_device(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.cuda_ipc_wrapper as _w
    from tests.fakes import make_bare_runtime_api

    monkeypatch.setattr(_w, "os", SimpleNamespace(name="nt"))
    api = make_bare_runtime_api()
    api.device = 0
    _mock_runtime_version(api, 12080)
    seen: dict[str, int] = {}

    def _attr(value_ptr, attr, device):
        seen["device"] = device.value
        _fill_scalar_byref(value_ptr, c_int, 1)
        return 0

    api.cudart.cudaDeviceGetAttribute.side_effect = _attr

    msg = api.check_ipc_capability(device=3)

    assert seen["device"] == 3
    assert msg is not None and "device 3" in msg
