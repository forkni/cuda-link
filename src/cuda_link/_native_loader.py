# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Native backend loader — adapts the compiled ``_native_waiter`` pybind11 module to
the :class:`WaitBackend` Protocol.

The compiled module (``src/cuda_link/_cpp/native_waiter.cpp``) is built on
Windows with no CUDA Toolkit or SDK required (see the root CMakeLists.txt). It
is intentionally imported lazily — importing ``cuda_link`` and using
FakeWaitBackend never requires the native module, so the pure-Python layer and
its tests run anywhere.
"""

from __future__ import annotations

from typing import Any

from ._wait_backend import WaitBackend, WaitResult, WaitStatus

_STATUS_FROM_INT = {
    0: WaitStatus.READY_SPIN,
    1: WaitStatus.READY_DOORBELL,
    2: WaitStatus.READY_LATE,  # retired 2026-07-04 (torn-frame fix) -- kept for ABI
    3: WaitStatus.TIMEOUT,
    4: WaitStatus.READY_POLL,
    5: WaitStatus.CUDA_ERROR,  # method slot carries the cudaError_t code
}


def load_native_backend(cuda_event_query_fn_ptr: int) -> WaitBackend:
    """Import the compiled native module and return a WaitBackend-conforming adapter.

    Args:
        cuda_event_query_fn_ptr: the raw address of the ``cudaEventQuery`` symbol,
            as resolved by the caller's own CUDA runtime wrapper (e.g.
            ``CUDARuntimeAPI.cudart_event_query_fn_ptr()`` in cuda_link's
            cuda_ipc_wrapper.py). Passed in explicitly rather than resolved
            independently: a bare-name ``GetModuleHandleW`` lookup here would not
            reliably find the *same* cudart instance the rest of the process is
            using when more than one same-named cudart DLL is loaded from
            different directories (e.g. one pulled in transitively by torch, one
            loaded via an explicit full path) — diagnosed 2026-07-04 (PLAN-002 R5)
            after a native module that resolved the wrong instance silently
            misreported a genuinely-pending CUDA event as complete.

    Raises:
        RuntimeError: with actionable guidance if the native module is not built,
            if ``cuda_event_query_fn_ptr`` is 0 (caller has no cudart loaded yet),
            or if the host cannot create high-resolution waitable timers
            (Windows 10 pre-1803) — activating without them would reintroduce
            the ~15.6ms-quantized block-phase waits fixed on 2026-07-06, so the
            caller should fall back to the python wait path instead.
    """
    try:
        from . import _native_waiter  # type: ignore[attr-defined]  # compiled extension
    except ImportError as e:  # pragma: no cover - exercised only without the built ext
        raise RuntimeError(
            "cuda-link's native module (_native_waiter) is not available. It must be "
            "built on Windows (no CUDA Toolkit required — see the root CMakeLists.txt "
            "and README for build instructions). (The pure-Python API and tests work "
            "without it via FakeWaitBackend.)"
        ) from e
    # Validate before set_cuda_event_query(): the pointer is module-global, shared by
    # every live _NativeWaitBackend in the process, so a refused load must not touch it.
    if not cuda_event_query_fn_ptr:
        raise RuntimeError(
            "cuda-link's native module (_native_waiter) is built but "
            "cuda_event_query_fn_ptr was 0 — the caller has no cudart runtime "
            "loaded yet (e.g. via cuda_link.cuda_ipc_wrapper.get_cuda_runtime()) "
            "before the native wait backend was activated."
        )
    if not _native_waiter.hr_nap_supported():
        raise RuntimeError(
            "cuda-link's native module (_native_waiter) is built but this host "
            "cannot create high-resolution waitable timers (requires Windows 10 "
            "1803+). Without them the native block phase would be quantized to "
            "the ~15.6ms default timer tick — the exact defect fixed on "
            "2026-07-06 — so the native backend refuses to activate; the python "
            "wait path (which the Importer falls back to automatically) is "
            "strictly better on this host."
        )
    _native_waiter.set_cuda_event_query(cuda_event_query_fn_ptr)
    return _NativeWaitBackend(_native_waiter)


class _NativeWaitBackend:
    """Thin adapter over the compiled ``_native_waiter`` module.

    Kept deliberately thin: forwards 1:1 to the native module, which owns the
    spin/block state machine and its high-resolution timer naps (cudart
    dispatch is the fixed function pointer registered by load_native_backend()
    above). Satisfies :class:`WaitBackend` structurally.
    """

    def __init__(self, mod: Any) -> None:
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
        return _to_wait_result(status_int, waited_us, method)


def _to_wait_result(status_int: int, waited_us: float, method: int) -> WaitResult:
    """Translate a native result tuple, raising per the WaitBackend contract.

    A hard cudaEventQuery error raises RuntimeError (as the python wait path's
    query_event does) instead of masquerading as a TIMEOUT; an unknown status int
    (a native module newer than this loader) raises rather than KeyError-ing.
    """
    status = _STATUS_FROM_INT.get(status_int)
    if status is None:
        raise RuntimeError(f"_native_waiter.wait_slot returned unknown status {status_int}")
    if status is WaitStatus.CUDA_ERROR:
        raise RuntimeError(f"cudaEventQuery failed: {method}")
    return WaitResult(status=status, waited_us=waited_us, method=method)
