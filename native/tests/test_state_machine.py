"""
Injected-fake tests for the real ``wait_slot_impl`` C++ state machine.

Unlike ``test_backend.py`` (which drives the independent, pure-Python
``FakeWaitBackend`` double) and ``test_native_smoke.py`` (which requires a real
Windows box with a loaded CUDA runtime), these tests drive the *actual* compiled
state machine in ``native_waiter.cpp`` through its test-only ``wait_slot_test()``
pybind entry point, with Python-injected fake polling functions standing in for
``cudaEventQuery`` / ``write_idx`` / the doorbell / the clock.

Requires only that the ``_native_waiter`` extension is compiled (Windows) — no
GPU or loaded cudart needed, since ``set_cuda_event_query()`` is never called
here and the fakes never touch real CUDA state. This is a lighter bar than
``test_native_smoke.py``'s ``_native_module_present()`` (which also requires a
real cudart), so this file can run on any Windows box with `pip install ./native`
done, GPU or not.

CAVEAT: native-tests CI (.github/workflows/tests.yml, the `native-tests` job)
runs on ubuntu-latest and never builds the Windows-only extension. Like
test_native_smoke.py, this file only actually exercises anything when run
manually on a Windows box with the extension compiled -- it is not enforced by
the automated CI gate.

Run:
    pytest native/tests/test_state_machine.py -m requires_native -v
"""

from __future__ import annotations

import pytest


def _native_waiter_present() -> bool:
    """Return True if the compiled _native_waiter extension is importable.

    Deliberately does NOT require a loaded cudart / real GPU (contrast with
    test_native_smoke.py's _native_module_present()) -- these tests never call
    set_cuda_event_query() or touch real CUDA state.
    """
    try:
        import cuda_link_native._native_waiter  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


_skip_no_native_waiter = pytest.mark.skipif(
    not _native_waiter_present(),
    reason="native _native_waiter not built -- see native/README.md",
)

pytestmark = [pytest.mark.requires_native, _skip_no_native_waiter]


# ---------------------------------------------------------------------------
# Fake clock / polling primitives shared across tests.
# ---------------------------------------------------------------------------


class _FakeClock:
    """Monotonically-advancing fake microsecond clock, advanced explicitly by tests.

    Passed as wait_slot_test's now_us callable. Every call to wait_doorbell()
    (the block-phase's "wake or bounded slice" primitive) also advances the
    clock by the requested slice, mimicking a real WaitForSingleObject timeout,
    so deadline math behaves realistically without any real sleeping.
    """

    def __init__(self, start_us: float = 0.0) -> None:
        self.t = start_us

    def now_us(self) -> float:
        return self.t

    def advance(self, delta_us: float) -> None:
        self.t += delta_us


def _wait_slot_test(**kwargs):
    from cuda_link_native._backend import WaitResult, WaitStatus  # noqa: PLC0415
    from cuda_link_native._native_waiter import wait_slot_test  # noqa: PLC0415

    status_int, waited_us, method = wait_slot_test(**kwargs)
    status = {
        0: WaitStatus.READY_SPIN,
        1: WaitStatus.READY_DOORBELL,
        2: WaitStatus.READY_LATE,
        3: WaitStatus.TIMEOUT,
        4: WaitStatus.READY_POLL,
    }[status_int]
    return WaitResult(status=status, waited_us=waited_us, method=method)


# ---------------------------------------------------------------------------
# Baseline behaviors
# ---------------------------------------------------------------------------


def test_ready_spin_when_event_already_signaled():
    """An event that's ready from the very first spin-phase check returns READY_SPIN."""
    clock = _FakeClock()
    result = _wait_slot_test(
        event_query=lambda ptr: True,
        read_write_idx=lambda addr: 0,
        wait_doorbell=lambda handle, slice_ms: False,
        now_us=clock.now_us,
        event_ptr=0x1000,
        doorbell_handle=0x2000,
        write_idx_addr=0x3000,
        last_write_idx=5,
        spin_us=200,
        timeout_ms=1000,
    )
    assert result.status.name == "READY_SPIN"


def test_timeout_when_nothing_ever_ready():
    """Event never ready, no doorbell wake -- must hit the deadline, not hang."""
    clock = _FakeClock()

    def wait_doorbell(handle, slice_ms):
        clock.advance(slice_ms * 1000.0)  # simulate a real bounded wait consuming time
        return False

    result = _wait_slot_test(
        event_query=lambda ptr: False,
        read_write_idx=lambda addr: 0,
        wait_doorbell=wait_doorbell,
        now_us=clock.now_us,
        event_ptr=0x1000,
        doorbell_handle=0x2000,
        write_idx_addr=0x3000,
        last_write_idx=5,
        spin_us=0,
        timeout_ms=10,  # 10ms deadline
    )
    assert result.status.name == "TIMEOUT"


def test_ready_doorbell_after_genuine_wake():
    """A doorbell wake (wait_doorbell -> True) followed by the event turning ready."""
    clock = _FakeClock()
    calls = {"n": 0}

    def event_query(ptr):
        calls["n"] += 1
        return calls["n"] >= 2  # not ready first check, ready after the wake

    def wait_doorbell(handle, slice_ms):
        clock.advance(slice_ms * 1000.0)
        return True  # genuine wake

    result = _wait_slot_test(
        event_query=event_query,
        read_write_idx=lambda addr: 0,
        wait_doorbell=wait_doorbell,
        now_us=clock.now_us,
        event_ptr=0x1000,
        doorbell_handle=0x2000,
        write_idx_addr=0x3000,
        last_write_idx=5,
        spin_us=0,
        timeout_ms=1000,
    )
    assert result.status.name == "READY_DOORBELL"


def test_ready_poll_when_no_doorbell_available():
    """doorbell_handle=0 -- block phase must use bare Sleep polling and report READY_POLL,
    never claim a doorbell wake that could not have happened."""
    clock = _FakeClock()
    calls = {"n": 0}

    def event_query(ptr):
        calls["n"] += 1
        return calls["n"] >= 3

    result = _wait_slot_test(
        event_query=event_query,
        read_write_idx=lambda addr: 0,
        wait_doorbell=lambda handle, slice_ms: pytest.fail("wait_doorbell must not be called when doorbell_handle=0"),
        now_us=clock.now_us,
        event_ptr=0x1000,
        doorbell_handle=0,  # no doorbell
        write_idx_addr=0x3000,
        last_write_idx=5,
        spin_us=0,
        timeout_ms=1000,
    )
    assert result.status.name == "READY_POLL"


# ---------------------------------------------------------------------------
# The regression test: the actual torn-frame race this fix closes.
# ---------------------------------------------------------------------------


def test_write_idx_advancing_does_not_short_circuit_a_pending_event():
    """The core regression test for the 2026-07-04 torn-frame fix.

    Before the fix, the block phase pre-checked write_idx and returned
    READY_LATE as soon as it advanced past last_write_idx -- even if THIS
    slot's own event was still not ready (simulating a producer that has
    raced ahead under async export while this slot's D2D copy is still in
    flight). That is exactly the scenario constructed here: write_idx reports
    a newer value than last_write_idx on every check, but event_query never
    returns True until well after several such checks.

    Assert: the call does NOT return early on the write_idx mismatch alone.
    It must keep waiting (and, per this test, correctly resolve once the
    event genuinely becomes ready) -- proving cudaEventQuery is the only exit
    condition, not write_idx.
    """
    clock = _FakeClock()
    calls = {"event": 0, "write_idx": 0}

    def read_write_idx(addr):
        calls["write_idx"] += 1
        # Always report a NEWER write_idx than what the caller observed --
        # the exact condition that used to trigger the unsafe READY_LATE
        # early-return.
        return 999

    def event_query(ptr):
        calls["event"] += 1
        # Not ready for the first several checks (simulating an in-flight
        # async D2D copy for THIS slot), then genuinely ready.
        return calls["event"] >= 5

    def wait_doorbell(handle, slice_ms):
        clock.advance(slice_ms * 1000.0)
        return False  # never a genuine wake -- forces repeated re-checks

    result = _wait_slot_test(
        event_query=event_query,
        read_write_idx=read_write_idx,
        wait_doorbell=wait_doorbell,
        now_us=clock.now_us,
        event_ptr=0x1000,
        doorbell_handle=0x2000,
        write_idx_addr=0x3000,
        last_write_idx=5,  # write_idx (999) != last_write_idx (5) from the very first check
        spin_us=0,
        timeout_ms=10_000,
    )
    # Must have genuinely waited for the event, not exited on write_idx alone.
    assert result.status.name in ("READY_DOORBELL", "READY_POLL")
    assert calls["event"] >= 5, "must have re-checked the event until it was genuinely ready"


def test_write_idx_advancing_with_event_never_ready_times_out_not_ready_late():
    """Same setup as above, but the event NEVER becomes ready -- must TIMEOUT,
    never fabricate a READY_LATE (or any other "ready") result from write_idx alone."""
    clock = _FakeClock()

    def read_write_idx(addr):
        return 999  # always "newer" than last_write_idx

    def wait_doorbell(handle, slice_ms):
        clock.advance(slice_ms * 1000.0)
        return False

    result = _wait_slot_test(
        event_query=lambda ptr: False,  # never ready
        read_write_idx=read_write_idx,
        wait_doorbell=wait_doorbell,
        now_us=clock.now_us,
        event_ptr=0x1000,
        doorbell_handle=0x2000,
        write_idx_addr=0x3000,
        last_write_idx=5,
        spin_us=0,
        timeout_ms=10,
    )
    assert result.status.name == "TIMEOUT"
