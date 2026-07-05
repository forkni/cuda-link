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

from ._backend import WaitBackend, WaitResult, WaitStatus

_STATUS_FROM_INT = {
    0: WaitStatus.READY_SPIN,
    1: WaitStatus.READY_DOORBELL,
    2: WaitStatus.READY_LATE,  # retired 2026-07-04 (torn-frame fix) -- kept for ABI
    3: WaitStatus.TIMEOUT,
    4: WaitStatus.READY_POLL,
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
            or if ``cuda_event_query_fn_ptr`` is 0 (caller has no cudart loaded yet).
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
    _native_waiter.set_cuda_event_query(cuda_event_query_fn_ptr)
    if not _native_waiter.cudart_resolved():
        raise RuntimeError(
            "cuda-link-native module (_native_waiter) is built but "
            "cuda_event_query_fn_ptr was 0 — the caller has no cudart runtime "
            "loaded yet (e.g. via cuda_link.cuda_ipc_wrapper.get_cuda_runtime()) "
            "before the native wait backend was activated."
        )
    return _NativeWaitBackend(_native_waiter)


class _NativeWaitBackend:
    """Thin adapter over the compiled ``_native_waiter`` module.

    Kept deliberately thin: forwards 1:1 to the native module, which owns the
    spin/block state machine and the Win32 doorbell wait (cudart dispatch is
    the fixed function pointer registered by load_native_backend() above).
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
