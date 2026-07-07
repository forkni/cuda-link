"""
Tests for the R5 native-wait-backend seam in Importer._wait_for_slot.

Deliberately does NOT import cuda_link._native_waiter — core tests must stay
independent of the compiled extension (it may not be built on this host or
platform at all). Instead, a tiny local duck-typed fake exercises the seam
structurally: anything with a matching `.wait_slot(...)` method and a result
carrying `.status.name` / `.waited_us` satisfies what `_wait_for_slot` needs.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum, auto

import pytest
from fakes import make_connected_importer

from cuda_link._importer_port import ImportOutcome, ImportPolicy


class _FakeStatus(Enum):
    READY_SPIN = auto()
    READY_DOORBELL = auto()
    READY_LATE = auto()
    TIMEOUT = auto()


@dataclass(frozen=True)
class _FakeResult:
    status: _FakeStatus
    waited_us: float
    method: int = 0


class _LocalFakeWaitBackend:
    """Local stand-in for cuda_link._wait_backend.FakeWaitBackend — records calls, scripts results."""

    def __init__(self, status: _FakeStatus = _FakeStatus.READY_SPIN, waited_us: float = 1.5) -> None:
        self.calls: list[dict] = []
        self.status = status
        self.waited_us = waited_us

    def wait_slot(self, event_ptr, doorbell_handle, write_idx_addr, last_write_idx, spin_us, timeout_ms):
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
        return _FakeResult(status=self.status, waited_us=self.waited_us)


def _engage_native_seam(imp, backend, *, doorbell_handle: int = 0xDEAD_BEEF, event_value: int = 0x1234_5678):
    """Wire an Importer built via make_connected_importer() to take the native branch.

    make_connected_importer() uses Importer.from_connection(), which never calls
    _open_ipc_slots — so _wait_backend/doorbell_handle must be set directly here,
    mirroring how existing tests inject imp._conn.ipc_events[0] = mock_event.
    """
    imp._wait_backend = backend
    imp._conn.ipc_events[0] = ctypes.c_uint64(event_value)
    imp._conn.doorbell_handle = doorbell_handle


def test_ready_spin_updates_spin_counters_and_returns_waited_us() -> None:
    backend = _LocalFakeWaitBackend(status=_FakeStatus.READY_SPIN, waited_us=3.3)
    imp = make_connected_importer(write_idx=1)
    _engage_native_seam(imp, backend)

    result = imp.get_frame_numpy()

    assert result.outcome is ImportOutcome.NEW_FRAME
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call["event_ptr"] == 0x1234_5678
    assert call["doorbell_handle"] == 0xDEAD_BEEF
    # _try_acquire() bumps self._last_write_idx to the just-acquired slot's
    # write_idx BEFORE _wait_for_slot runs, so the native call sees write_idx=1
    # (this frame's value), not the pre-acquire 0 — correct: the lost-wakeup
    # pre-check needs "value we've now seen" to detect a newer frame mid-wait.
    assert call["last_write_idx"] == 1
    assert imp.wait_spin_hits == 1
    assert imp.wait_sleep_hits == 0
    assert imp.total_wait_spin_us == pytest.approx(3.3)


def test_ready_doorbell_updates_sleep_counters() -> None:
    backend = _LocalFakeWaitBackend(status=_FakeStatus.READY_DOORBELL, waited_us=42.0)
    imp = make_connected_importer(write_idx=1)
    _engage_native_seam(imp, backend)

    result = imp.get_frame_numpy()

    assert result.outcome is ImportOutcome.NEW_FRAME
    assert imp.wait_spin_hits == 0
    assert imp.wait_sleep_hits == 1
    assert imp.total_wait_sleep_us == pytest.approx(42.0)


def test_ready_late_also_counts_as_sleep_bucket() -> None:
    backend = _LocalFakeWaitBackend(status=_FakeStatus.READY_LATE, waited_us=7.0)
    imp = make_connected_importer(write_idx=1)
    _engage_native_seam(imp, backend)

    result = imp.get_frame_numpy()

    assert result.outcome is ImportOutcome.NEW_FRAME
    assert imp.wait_sleep_hits == 1
    assert imp.total_wait_sleep_us == pytest.approx(7.0)


def test_timeout_status_raises_timeout_error_same_as_python_path() -> None:
    backend = _LocalFakeWaitBackend(status=_FakeStatus.TIMEOUT, waited_us=5000.0)
    imp = make_connected_importer(write_idx=1, timeout_ms=1.0)
    _engage_native_seam(imp, backend)

    result = imp.get_frame_numpy()

    # Importer.get_frame_numpy() catches TimeoutError internally and reports it
    # as an outcome, same contract as the pre-R5 python path (see
    # test_outcome_timeout_raises_no_exception_returns_result in test_importer.py).
    assert result.outcome is ImportOutcome.TIMEOUT
    assert result.frame is None


def test_backend_not_engaged_without_doorbell_handle() -> None:
    """Even with a resolved backend, no doorbell handle means the python path runs."""
    backend = _LocalFakeWaitBackend(status=_FakeStatus.READY_SPIN, waited_us=1.0)
    imp = make_connected_importer(write_idx=1)
    imp._wait_backend = backend
    imp._conn.ipc_events[0] = ctypes.c_uint64(0x1)
    imp._conn.doorbell_handle = None  # not opened

    imp._conn.cuda.query_event.return_value = True  # python path: event already ready
    result = imp.get_frame_numpy()

    assert result.outcome is ImportOutcome.NEW_FRAME
    assert backend.calls == []  # never called — python path took over


def test_wait_backend_defaults_to_none_and_python_path_is_unaffected() -> None:
    """Importer.from_connection() never resolves a backend — pre-R5 behavior exactly."""
    imp = make_connected_importer(write_idx=1)
    assert imp._wait_backend is None

    imp._conn.ipc_events[0] = None  # no-event path
    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.NEW_FRAME


def test_for_testing_policy_forces_python_wait_backend() -> None:
    """ImportPolicy.for_testing() must not enable auto-resolution during unit tests."""
    policy = ImportPolicy.for_testing()
    assert policy.wait_backend == "python"


def test_default_policy_wait_backend_is_auto() -> None:
    """Production default stays 'auto' — only for_testing() overrides it."""
    assert ImportPolicy().wait_backend == "auto"


def test_from_env_reads_wait_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDALINK_WAIT_BACKEND", "native")
    assert ImportPolicy.from_env().wait_backend == "native"
