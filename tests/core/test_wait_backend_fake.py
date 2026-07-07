"""Tests for the WaitBackend seam and FakeWaitBackend."""

from __future__ import annotations

from cuda_link._wait_backend import FakeWaitBackend, WaitBackend, WaitResult, WaitStatus


def test_fake_satisfies_protocol():
    assert isinstance(FakeWaitBackend(), WaitBackend)


def test_wait_slot_ready_immediately_by_default():
    b = FakeWaitBackend()
    r = b.wait_slot(0x1000, 0x2000, 0x3000, 5, 200, 1000)
    assert isinstance(r, WaitResult)
    assert r.status is WaitStatus.READY_SPIN
    assert len(b.calls) == 1
    assert b.calls[0]["event_ptr"] == 0x1000
    assert b.calls[0]["doorbell_handle"] == 0x2000
    assert b.calls[0]["write_idx_addr"] == 0x3000
    assert b.calls[0]["last_write_idx"] == 5
    assert b.calls[0]["spin_us"] == 200
    assert b.calls[0]["timeout_ms"] == 1000


def test_wait_slot_scriptable_status_and_timing():
    b = FakeWaitBackend()
    b.force_status = WaitStatus.READY_DOORBELL
    b.force_waited_us = 4.2
    b.force_method = 2
    r = b.wait_slot(1, 2, 3, 0, 200, 1000)
    assert r.status is WaitStatus.READY_DOORBELL
    assert r.waited_us == 4.2
    assert r.method == 2


def test_wait_slot_ready_after_n_calls():
    b = FakeWaitBackend()
    b.ready_after_calls = 2
    b.force_status = WaitStatus.READY_LATE
    r1 = b.wait_slot(1, 2, 3, 0, 200, 1000)
    r2 = b.wait_slot(1, 2, 3, 0, 200, 1000)
    r3 = b.wait_slot(1, 2, 3, 0, 200, 1000)
    assert r1.status is WaitStatus.TIMEOUT
    assert r2.status is WaitStatus.TIMEOUT
    assert r3.status is WaitStatus.READY_LATE
    assert len(b.calls) == 3


def test_wait_slot_timeout_status():
    b = FakeWaitBackend()
    b.ready_after_calls = 999  # never ready within a handful of calls
    r = b.wait_slot(1, 2, 3, 0, 200, 1000)
    assert r.status is WaitStatus.TIMEOUT
