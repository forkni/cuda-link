"""
Native backend loader — adapts the compiled ``_native_waiter`` pybind11 module to
the :class:`WaitBackend` Protocol.

The compiled module (``native/src/cuda_link_native/_cpp/native_waiter.cpp``) is
built on Windows with no CUDA Toolkit or SDK required (see native/README.md). It
is intentionally imported lazily — importing ``cuda_link_native`` and using
FakeWaitBackend never requires the native module, so the pure-Python layer and
its tests run anywhere.
"""

from __future__ import annotations

import glob as _glob
import os as _os
import sys as _sys

from ._backend import WaitBackend, WaitResult, WaitStatus

# Python 3.8+ no longer searches PATH for DLL dependencies of extension modules.
# Explicitly register CUDA bin so cudart64_*.dll is visible for GetModuleHandleW
# resolution when _native_waiter loads (mirrors cuda_link_spout._native).
if _sys.platform == "win32":
    for _cuda_bin in sorted(
        _glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"),
        reverse=True,
    ):
        if _os.path.isdir(_cuda_bin):
            _os.add_dll_directory(_cuda_bin)
            break

_STATUS_FROM_INT = {
    0: WaitStatus.READY_SPIN,
    1: WaitStatus.READY_DOORBELL,
    2: WaitStatus.READY_LATE,
    3: WaitStatus.TIMEOUT,
}


def load_native_backend() -> WaitBackend:
    """Import the compiled native module and return a WaitBackend-conforming adapter.

    Raises:
        RuntimeError: with actionable guidance if the native module is not built,
            or if it cannot resolve a cudart instance already loaded by this process.
    """
    try:
        from . import _native_waiter  # type: ignore[attr-defined]  # compiled extension
    except ImportError as e:  # pragma: no cover - exercised only without the built ext
        raise RuntimeError(
            "cuda-link-native module (_native_waiter) is not available. It must be "
            "built on Windows (no CUDA Toolkit required — see native/README.md for "
            "build instructions). (The pure-Python API and tests work without it "
            "via FakeWaitBackend.)"
        ) from e
    if not _native_waiter.cudart_resolved():
        raise RuntimeError(
            "cuda-link-native module (_native_waiter) is built but could not resolve "
            "a loaded cudart instance (cudart64_13/12/11/110.dll). This process must "
            "have already loaded a CUDA runtime (e.g. via cuda_link.cuda_ipc_wrapper) "
            "before the native wait backend is activated."
        )
    return _NativeWaitBackend(_native_waiter)


class _NativeWaitBackend:
    """Thin adapter over the compiled ``_native_waiter`` module.

    Kept deliberately thin: forwards 1:1 to the native module, which owns the
    cudart resolution, the spin/block state machine, and the Win32 doorbell wait.
    Satisfies :class:`WaitBackend` structurally.
    """

    def __init__(self, mod: object) -> None:
        self._mod = mod

    def wait_slot(
        self,
        event_ptr: int,
        doorbell_handle: int,
        write_idx_addr: int,
        last_write_idx: int,
        spin_us: int,
        timeout_ms: int,
    ) -> WaitResult:
        status_int, waited_us, method = self._mod.wait_slot(
            event_ptr, doorbell_handle, write_idx_addr, last_write_idx, spin_us, timeout_ms
        )
        return WaitResult(status=_STATUS_FROM_INT[status_int], waited_us=waited_us, method=method)
