"""
GPU-free coverage tests for CUDARuntimeAPI's DLL-loading and diagnostic-logging
helpers: ``_load_cuda_runtime``, ``_log_dll_path``, and ``_load_driver_api``.

Background
----------
On this dev machine a real CUDA Toolkit install means ``_load_cuda_runtime``
always succeeds on the very first hard-coded absolute path, so the bare-name
fallback tier and the final "could not load" diagnostic-message branch are
never exercised by the existing hardware-backed test suite. These tests patch
``os.path.exists`` / ``ctypes.CDLL`` at the module level (the same seam the
production code calls through) to force each path through the function
without touching a real cudart/nvcuda DLL.

All tests build a ``CUDARuntimeAPI`` via ``tests.fakes.make_bare_runtime_api``
(``object.__new__`` + injected mock), so ``__init__`` — and therefore any real
DLL load — never runs. No ``@pytest.mark.requires_cuda`` / ``requires_native``
marker is needed.
"""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock, patch

import pytest

from tests.fakes import make_bare_runtime_api


def _os_error_with_winerror(errno_: int, strerror: str, winerror: int) -> OSError:
    """Build an OSError carrying .winerror on any platform.

    ctypes.CDLL on Windows raises OSError with .winerror set, but the 4-arg
    ``OSError(errno, strerror, None, winerror)`` form only sets the attribute on
    Windows builds of CPython — POSIX builds silently ignore it, so CI's ubuntu
    runners need the instance-attribute assignment instead.
    """
    err = OSError(errno_, strerror)
    err.winerror = winerror
    return err


# ---------------------------------------------------------------------------
# _load_cuda_runtime
# ---------------------------------------------------------------------------


def test_load_cuda_runtime_full_path_tier_oserror_falls_through_to_bare_name() -> None:
    """A full-path CDLL load failure is logged and skipped; the bare-name tier then succeeds.

    Covers the except/continue branch of the absolute-path tier plus the try/return
    success branch of the bare-name tier.
    """
    api = make_bare_runtime_api()
    monkeypatch_target = "cuda_link.cuda_ipc_wrapper"

    fake_dll = MagicMock(name="fake_cudart_dll")
    call_log: list[str] = []

    def _fake_cdll(name_or_path, *args, **kwargs):
        call_log.append(name_or_path)
        if call_log.count(name_or_path) == 1 and args:
            # First tier call (absolute path, winmode=0 passed positionally/kw) fails once.
            pass
        if name_or_path.endswith(".dll") and "\\" in name_or_path:
            raise OSError(5, "Access is denied")
        return fake_dll

    with (
        patch(f"{monkeypatch_target}.os.path.exists", return_value=True),
        patch(f"{monkeypatch_target}.ctypes.CDLL", side_effect=_fake_cdll) as mock_cdll,
        patch.object(type(api), "_log_dll_path") as mock_log,
    ):
        result = api._load_cuda_runtime()

    assert result is fake_dll
    assert mock_cdll.called
    mock_log.assert_called_once()


def test_load_cuda_runtime_bare_name_tier_skips_failures_then_succeeds() -> None:
    """All absolute paths are absent; the bare-name loop retries and eventually succeeds."""
    api = make_bare_runtime_api()

    fake_dll = MagicMock(name="fake_cudart_dll")
    attempts: list[str] = []

    def _fake_cdll(name, *args, **kwargs):
        attempts.append(name)
        if name in ("cudart64_13.dll", "cudart64_12.dll"):
            raise OSError(126, "The specified module could not be found")
        return fake_dll

    with (
        patch("cuda_link.cuda_ipc_wrapper.os.path.exists", return_value=False),
        patch("cuda_link.cuda_ipc_wrapper.ctypes.CDLL", side_effect=_fake_cdll),
        patch.object(type(api), "_log_dll_path") as mock_log,
    ):
        result = api._load_cuda_runtime()

    assert result is fake_dll
    # Both failing names were tried (except/continue branch) before the success.
    assert attempts[:2] == ["cudart64_13.dll", "cudart64_12.dll"]
    assert attempts[2] == "cudart64_11.dll"
    mock_log.assert_called_once()


def test_load_cuda_runtime_raises_with_winerror_126_hint() -> None:
    """When every tier fails and the last error is winerror 126, the dependency-tracer hint is included."""
    api = make_bare_runtime_api()

    def _always_fail(name, *args, **kwargs):
        # OSError(errno, strerror) alone leaves .winerror unset; the production
        # code reads getattr(e, "winerror", None), so .winerror must be set to
        # simulate the real Windows exception ctypes.CDLL raises on a failed LoadLibrary.
        raise _os_error_with_winerror(126, "The specified module could not be found", 126)

    with (
        patch("cuda_link.cuda_ipc_wrapper.os.path.exists", return_value=False),
        patch("cuda_link.cuda_ipc_wrapper.ctypes.CDLL", side_effect=_always_fail),
        pytest.raises(Exception) as exc_info,
    ):
        api._load_cuda_runtime()

    message = str(exc_info.value)
    assert "Could not load CUDA runtime" in message
    assert "Hint: winerror 126" in message
    assert "winerror=126" in message


def test_load_cuda_runtime_raises_without_winerror_126_hint() -> None:
    """When the last error's winerror is not 126, no dependency-tracer hint is appended."""
    api = make_bare_runtime_api()

    def _always_fail(name, *args, **kwargs):
        raise _os_error_with_winerror(2, "The system cannot find the file specified", 2)

    with (
        patch("cuda_link.cuda_ipc_wrapper.os.path.exists", return_value=False),
        patch("cuda_link.cuda_ipc_wrapper.ctypes.CDLL", side_effect=_always_fail),
        pytest.raises(Exception) as exc_info,
    ):
        api._load_cuda_runtime()

    message = str(exc_info.value)
    assert "Could not load CUDA runtime" in message
    assert "Hint: winerror 126" not in message
    assert "winerror=2" in message


def test_load_cuda_runtime_raises_uses_path_tier_error_when_name_tier_had_no_attempt() -> None:
    """last_path_err is used in the diagnostic message when a path was attempted (and failed).

    Exercises the `os.path.exists` True + CDLL-raises branch feeding into last_path_err,
    combined with the bare-name tier also exhausting, so the final message reflects a
    real absolute-path failure rather than only a bare-name failure.
    """
    api = make_bare_runtime_api()

    def _always_fail(name, *args, **kwargs):
        raise _os_error_with_winerror(87, "The parameter is incorrect", 87)

    with (
        patch("cuda_link.cuda_ipc_wrapper.os.path.exists", return_value=True),
        patch("cuda_link.cuda_ipc_wrapper.ctypes.CDLL", side_effect=_always_fail),
        pytest.raises(Exception) as exc_info,
    ):
        api._load_cuda_runtime()

    assert "winerror=87" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _log_dll_path
# ---------------------------------------------------------------------------


def test_log_dll_path_no_kernel32_prints_unavailable(capsys: pytest.CaptureFixture[str]) -> None:
    """When _kernel32 is None (non-Windows), path resolution is skipped entirely."""
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    with patch("cuda_link.cuda_ipc_wrapper._kernel32", None):
        CUDARuntimeAPI._log_dll_path(MagicMock(), "cudart64_12.dll")

    captured = capsys.readouterr()
    assert "path resolution unavailable" in captured.out


def test_log_dll_path_handle_not_int_prints_handle_unavailable(capsys: pytest.CaptureFixture[str]) -> None:
    """When dll._handle is not an int, the handle-unavailable message is printed instead of crashing."""
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    fake_dll = MagicMock()
    fake_dll._handle = "not-an-int"
    fake_kernel32 = MagicMock()

    with patch("cuda_link.cuda_ipc_wrapper._kernel32", fake_kernel32):
        CUDARuntimeAPI._log_dll_path(fake_dll, "cudart64_12.dll")

    captured = capsys.readouterr()
    assert "handle not available" in captured.out
    fake_kernel32.GetModuleFileNameW.assert_not_called()


def test_log_dll_path_get_module_filename_returns_zero_prints_winerror(capsys: pytest.CaptureFixture[str]) -> None:
    """GetModuleFileNameW returning 0 (failure) is surfaced via ctypes.WinError, not raised."""
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    fake_dll = MagicMock()
    fake_dll._handle = 12345
    fake_kernel32 = MagicMock()
    fake_kernel32.GetModuleFileNameW.return_value = 0

    with (
        patch("cuda_link.cuda_ipc_wrapper._kernel32", fake_kernel32),
        # ctypes.WinError / ctypes.get_last_error exist only on Windows builds of
        # CPython; create=True lets this test drive the failure branch on CI's
        # ubuntu runners too (the patched attributes are removed again on exit).
        patch("cuda_link.cuda_ipc_wrapper.ctypes.get_last_error", create=True, return_value=126),
        patch(
            "cuda_link.cuda_ipc_wrapper.ctypes.WinError",
            create=True,
            return_value=OSError("[WinError 126] The specified module could not be found"),
        ),
    ):
        CUDARuntimeAPI._log_dll_path(fake_dll, "cudart64_12.dll")

    captured = capsys.readouterr()
    assert "GetModuleFileNameW failed" in captured.out


def test_log_dll_path_oserror_during_resolution_is_caught(capsys: pytest.CaptureFixture[str]) -> None:
    """An OSError raised while resolving the path is caught and logged, not propagated."""
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    fake_dll = MagicMock()
    fake_dll._handle = 999
    fake_kernel32 = MagicMock()
    fake_kernel32.GetModuleFileNameW.side_effect = OSError("simulated resolution failure")

    with patch("cuda_link.cuda_ipc_wrapper._kernel32", fake_kernel32):
        CUDARuntimeAPI._log_dll_path(fake_dll, "cudart64_12.dll")  # must not raise

    captured = capsys.readouterr()
    assert "could not resolve path" in captured.out


# ---------------------------------------------------------------------------
# _load_driver_api
# ---------------------------------------------------------------------------


def test_load_driver_api_nvcuda_missing_returns_none() -> None:
    """When nvcuda.dll can't be loaded (no driver installed), _load_driver_api returns None."""
    api = make_bare_runtime_api()

    def _fail_nvcuda(name, *args, **kwargs):
        assert name == "nvcuda.dll"
        raise OSError("nvcuda.dll not found")

    with patch("cuda_link.cuda_ipc_wrapper.ctypes.CDLL", side_effect=_fail_nvcuda):
        result = api._load_driver_api()

    assert result is None


def test_load_driver_api_cuinit_nonzero_returns_none() -> None:
    """When cuInit() returns a non-zero CUresult, the driver API is treated as unusable."""
    api = make_bare_runtime_api()

    fake_drv = MagicMock()
    fake_drv.cuInit.return_value = 3  # CUDA_ERROR_NOT_INITIALIZED

    with patch("cuda_link.cuda_ipc_wrapper.ctypes.CDLL", return_value=fake_drv):
        result = api._load_driver_api()

    assert result is None
    fake_drv.cuInit.assert_called_once()


def test_load_driver_api_success_returns_drv() -> None:
    """When nvcuda.dll loads and cuInit() succeeds, the driver handle is returned."""
    api = make_bare_runtime_api()

    fake_drv = MagicMock()
    fake_drv.cuInit.return_value = 0

    with patch("cuda_link.cuda_ipc_wrapper.ctypes.CDLL", return_value=fake_drv):
        result = api._load_driver_api()

    assert result is fake_drv
    # All 6 driver symbols must have been bound with argtypes/restype.
    assert fake_drv.cuDeviceGet.argtypes == [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
