"""
First on-hardware smoke tests for the _native_waiter native module.

These tests are the first exercises of the real cudart dispatch + spin/block
state machine that is otherwise only covered by pure-Python unit tests against
FakeWaitBackend. They require a Windows box with a CUDA runtime already loaded
in-process and the native module compiled via:

    pip install ./native

Run:
    pytest native/tests/test_native_smoke.py -m requires_native -v

All tests skip automatically when the native module is absent so the file is
safe to collect in any environment.
"""

from __future__ import annotations

import pytest
from cuda_link_native._backend import WaitStatus
from cuda_link_native._native import load_native_backend


def _get_runtime():
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime  # noqa: PLC0415

    return get_cuda_runtime()


def _load_backend():
    """load_native_backend() using this process's real, already-resolved cudart.

    The function-pointer address is sourced from cuda_link's own
    CUDARuntimeAPI (cudart_event_query_fn_ptr()) rather than rediscovered
    independently — see the module header comment in native_waiter.cpp for
    why an independent bare-name GetModuleHandle lookup is unsafe.
    """
    rt = _get_runtime()
    return load_native_backend(rt.cudart_event_query_fn_ptr())


def _native_module_present() -> bool:
    """Return True if the compiled _native_waiter extension is usable end-to-end."""
    try:
        _get_runtime()
    except (RuntimeError, OSError):
        return False
    try:
        _load_backend()
        return True
    except (ImportError, RuntimeError):
        return False


_skip_no_native = pytest.mark.skipif(
    not _native_module_present(),
    reason="native _native_waiter not built or no cudart loaded — see native/README.md",
)

pytestmark = [pytest.mark.requires_native, _skip_no_native]


def _make_event():
    """Create a real CUDA event via cuda_link's runtime wrapper; return (runtime, event)."""
    rt = _get_runtime()
    evt = rt.create_ipc_event()
    return rt, evt


def test_wait_slot_returns_ready_spin_for_already_signaled_event() -> None:
    """An event that is already complete should be caught in the spin phase."""
    backend = _load_backend()
    rt, evt = _make_event()
    try:
        rt.record_event(evt)  # record on the default/null stream -> completes immediately
        rt.synchronize()
        result = backend.wait_slot(int(evt.value), 0, 0, 0, spin_us=200, timeout_ms=1000)
        assert result.status is WaitStatus.READY_SPIN
        assert result.waited_us < 1000  # well under the full timeout
    finally:
        rt.destroy_event(evt)


def test_wait_slot_times_out_with_no_event_or_doorbell() -> None:
    """With nothing to signal readiness, wait_slot must hit the deadline.

    Deliberately does NOT try to construct a "genuinely pending real CUDA
    event" fixture. An earlier version of this test recorded `target` on a
    stream told to `cudaStreamWaitEvent` on a never-recorded `blocker`,
    expecting that to block indefinitely — but NVIDIA's own documentation is
    explicit that this is a **no-op**: "If cudaEventRecord() has not been
    called on event, the cudaStreamWaitEvent() call acts as if the record has
    already completed" (CUDA Runtime API, Event Management). That fixture
    measured a real event as ready in only a few ms (WDDM command-batch
    latency for the trivial record, not genuine blocking) — a driver-timing-
    dependent, flaky basis for a deadline test. event_ptr=0 / doorbell_handle=0
    exercises the identical block-phase deadline-checking code path
    deterministically, independent of any real GPU/driver timing.

    This test is also the regression test for the 2026-07-04 diagnosis
    (PLAN-002 R5): an earlier version of the native module resolved cudart
    independently via a bare-name GetModuleHandleW lookup, which — on a
    process with more than one same-named cudart DLL loaded from different
    directories — silently found the wrong instance. That specific defect is
    exercised by test_wait_slot_returns_ready_spin_for_already_signaled_event
    above (a real, valid event dispatched through the fixed function pointer);
    this test covers the orthogonal "nothing is ready" deadline path.
    """
    backend = _load_backend()
    result = backend.wait_slot(0, 0, 0, 0, spin_us=0, timeout_ms=50)
    assert result.status is WaitStatus.TIMEOUT
    assert result.waited_us >= 50 * 1000 * 0.8  # allow scheduling slack
