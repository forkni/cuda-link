"""
GPU-free coverage tests for CUDARuntimeAPI.__init__'s CUDA_LAUNCH_BLOCKING warning,
the _strict_errcheck closure's raise body, and get_cuda_runtime's device-mismatch guard.

Background
----------
__init__ runs real DLL-loading code, so exercising its CUDA_LAUNCH_BLOCKING branch
without a GPU requires patching _load_cuda_runtime / _load_driver_api on the class
before construction. The errcheck closure is a plain Python callable stored as
`.errcheck` on each strict cudart function — since ctypes' own errcheck-invocation
machinery only fires for real (non-mocked) FFI calls, these tests invoke the
retrieved closure directly with the same `(result, func, args)` signature ctypes
would use. get_cuda_runtime's singleton conflict path is tested by monkeypatching
the module-level `_cuda_runtime` global directly (auto-restored by monkeypatch).

All tests are GPU-free; no @pytest.mark.requires_cuda / requires_native needed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.fakes import make_bare_runtime_api

# ---------------------------------------------------------------------------
# __init__ — CUDA_LAUNCH_BLOCKING warning
# ---------------------------------------------------------------------------


def test_init_warns_when_cuda_launch_blocking_env_set(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """__init__ logs a warning when CUDA_LAUNCH_BLOCKING=1 is set in the environment."""
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    monkeypatch.setattr(CUDARuntimeAPI, "_load_cuda_runtime", lambda self: MagicMock())
    monkeypatch.setattr(CUDARuntimeAPI, "_load_driver_api", lambda self: None)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")

    with caplog.at_level(logging.WARNING, logger="cuda_link.cuda_ipc_wrapper"):
        api = CUDARuntimeAPI(device=0)

    assert api.device == 0
    assert any("CUDA_LAUNCH_BLOCKING=1 is set" in rec.message for rec in caplog.records)


def test_init_no_warning_when_cuda_launch_blocking_unset(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """__init__ does not log the CUDA_LAUNCH_BLOCKING warning when the env var is absent."""
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    monkeypatch.setattr(CUDARuntimeAPI, "_load_cuda_runtime", lambda self: MagicMock())
    monkeypatch.setattr(CUDARuntimeAPI, "_load_driver_api", lambda self: None)
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

    with caplog.at_level(logging.WARNING, logger="cuda_link.cuda_ipc_wrapper"):
        CUDARuntimeAPI(device=0)

    assert not any("CUDA_LAUNCH_BLOCKING=1 is set" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# _install_errcheck — _strict_errcheck closure raise body
# ---------------------------------------------------------------------------


def test_strict_errcheck_raises_cuda_ipc_error_on_nonzero_result() -> None:
    """The errcheck closure raises CudaIpcError with the decoded error string on failure."""
    # Sourced from cuda_ipc_wrapper's own resolved namespace, not cuda_link.cuda_runtime_types
    # directly — see the comment in test_wrapper_cov_methods.py for why identity matters here.
    from cuda_link.cuda_ipc_wrapper import CudaIpcError

    api = make_bare_runtime_api(install_errcheck=True)
    api.cudart.cudaGetErrorString.return_value = b"out of memory"

    errcheck_fn = api.cudart.cudaMalloc.errcheck
    fake_func = SimpleNamespace(__name__="cudaMalloc")

    with pytest.raises(CudaIpcError) as exc_info:
        errcheck_fn(2, fake_func, ())

    assert "cudaMalloc failed" in str(exc_info.value)
    assert "out of memory" in str(exc_info.value)
    assert "code 2" in str(exc_info.value)
    assert exc_info.value.code == 2


def test_strict_errcheck_falls_back_to_unknown_error_when_cstr_none() -> None:
    """When cudaGetErrorString returns None, the closure falls back to an 'unknown error' message."""
    from cuda_link.cuda_ipc_wrapper import CudaIpcError

    api = make_bare_runtime_api(install_errcheck=True)
    api.cudart.cudaGetErrorString.return_value = None

    errcheck_fn = api.cudart.cudaFree.errcheck
    fake_func = SimpleNamespace(__name__="cudaFree")

    with pytest.raises(CudaIpcError) as exc_info:
        errcheck_fn(999, fake_func, ())

    assert "unknown error 999" in str(exc_info.value)


# ---------------------------------------------------------------------------
# get_cuda_runtime — device-mismatch guard
# ---------------------------------------------------------------------------


def test_get_cuda_runtime_raises_on_device_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_cuda_runtime() raises when called with a device index != the existing singleton's."""
    import cuda_link.cuda_ipc_wrapper as wrapper_mod
    from cuda_link.cuda_ipc_wrapper import CudaLinkError

    fake_singleton = SimpleNamespace(device=0)
    monkeypatch.setattr(wrapper_mod, "_cuda_runtime", fake_singleton)

    with pytest.raises(CudaLinkError) as exc_info:
        wrapper_mod.get_cuda_runtime(device=1)

    assert "device 0" in str(exc_info.value)
    assert "device 1" in str(exc_info.value)
    # The global must be left untouched by the failed call.
    assert wrapper_mod._cuda_runtime is fake_singleton


def test_get_cuda_runtime_returns_existing_singleton_on_matching_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_cuda_runtime() returns the existing singleton unchanged when the device matches."""
    import cuda_link.cuda_ipc_wrapper as wrapper_mod

    fake_singleton = SimpleNamespace(device=0)
    monkeypatch.setattr(wrapper_mod, "_cuda_runtime", fake_singleton)

    result = wrapper_mod.get_cuda_runtime(device=0)

    assert result is fake_singleton
