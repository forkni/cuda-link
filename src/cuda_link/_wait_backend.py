"""
Wait backend port — the seam between cuda-link's Importer and the native
CUDA-event/doorbell wait module.

Mirrors cuda-link's port-adapter pattern (ADR-0001) and the sibling
cuda-link-spout package's ``SpoutBackend`` seam: a structural ``Protocol``
plus two adapters —

  _NativeWaitBackend  — wraps the compiled ``_native_waiter`` pybind11 module
                        (in _native_loader.py; Windows only)
  FakeWaitBackend     — in-memory, no GPU / no native module; drives all unit tests

``Importer._wait_for_slot`` (src/cuda_link/importer.py) depends only on this
Protocol, so the counter-translation/timeout behaviour is testable on any
machine without CUDA or a compiled extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable


class WaitStatus(Enum):
    """How ``wait_slot`` discovered the frame was ready (or that it timed out)."""

    READY_SPIN = auto()  # caught by the native busy-spin phase
    READY_DOORBELL = auto()  # block phase caught it; doorbell channel present (name kept for ABI/telemetry -- the doorbell itself is no longer waited on, 2026-07-06 timer-nap fix)
    READY_LATE = auto()  # retired 2026-07-04 (torn-frame fix) -- never emitted anymore, kept for ABI
    TIMEOUT = auto()  # deadline exceeded with no frame
    READY_POLL = auto()  # block phase caught it with no doorbell channel present (same timer-nap loop; distinct name only so telemetry never implies a doorbell that wasn't there)


@dataclass(frozen=True)
class WaitResult:
    """What the backend reports after a single ``wait_slot`` call."""

    status: WaitStatus
    waited_us: float
    method: int = 0  # opaque diagnostic code from the native layer; 0 for the fake


@runtime_checkable
class WaitBackend(Protocol):
    """Structural interface ``Importer._wait_for_slot`` requires from the native layer.

    All methods raise RuntimeError on a native/CUDA failure. ``TimeoutError`` is
    never raised here — a timed-out wait is reported as ``WaitResult(status=TIMEOUT,
    ...)`` and the caller (``Importer._wait_for_slot``) is responsible for raising
    ``TimeoutError`` to preserve today's contract.
    """

    def wait_slot(
        self,
        event_ptr: int,
        doorbell_handle: int,
        write_idx_addr: int,
        last_write_idx: int,
        spin_us: int,
        timeout_ms: int,
    ) -> WaitResult:
        """Wait for the per-slot CUDA event to signal readiness.

        *event_ptr* — the per-slot ``cudaEvent_t`` (as an int) to poll via
        ``cudaEventQuery`` — the only valid "this slot is ready" signal; see the
        2026-07-04 torn-frame fix note in ``native_waiter.cpp`` for why nothing
        else (e.g. ``write_idx`` alone) may be treated as ready. *doorbell_handle*
        — the Win32 event HANDLE (as an int) opened for this SHM channel, or 0 if
        unavailable. Since the 2026-07-06 timer-nap fix the block phase never
        waits on it (it re-checks ``cudaEventQuery`` between bounded
        high-resolution timer naps instead); the handle only selects the
        READY_DOORBELL-vs-READY_POLL status attribution. *write_idx_addr* /
        *last_write_idx* — kept for interface stability; no longer consulted by
        the real implementation (retained only so a stale/unrebuilt native module
        still accepts the same call shape). *spin_us* — the busy-spin budget
        before falling into the blocking phase. *timeout_ms* — the overall
        deadline for the call.
        """
        ...


# ---------------------------------------------------------------------------
# Fake backend — in-memory, deterministic; the test substrate.
# ---------------------------------------------------------------------------


class FakeWaitBackend:
    """In-memory WaitBackend for unit tests — records calls, scripts results.

    Satisfies the :class:`WaitBackend` Protocol structurally. No GPU, no native
    module. Tune behaviour via the public attributes before exercising the seam.
    """

    def __init__(self) -> None:
        # call log (assertable in tests)
        self.calls: list[dict] = []
        # scripting knobs
        self.ready_after_calls: int = 0  # 0 = ready immediately (first call)
        self.force_status: WaitStatus = WaitStatus.READY_SPIN
        self.force_waited_us: float = 1.0
        self.force_method: int = 0

    def wait_slot(
        self,
        event_ptr: int,
        doorbell_handle: int,
        write_idx_addr: int,
        last_write_idx: int,
        spin_us: int,
        timeout_ms: int,
    ) -> WaitResult:
        self.calls.append(
            {
                "event_ptr": event_ptr,
                "doorbell_handle": doorbell_handle,
                "write_idx_addr": write_idx_addr,
                "last_write_idx": last_write_idx,
                "spin_us": spin_us,
                "timeout_ms": timeout_ms,
            }
        )
        if len(self.calls) <= self.ready_after_calls:
            return WaitResult(status=WaitStatus.TIMEOUT, waited_us=float(timeout_ms) * 1000.0, method=0)
        return WaitResult(status=self.force_status, waited_us=self.force_waited_us, method=self.force_method)
