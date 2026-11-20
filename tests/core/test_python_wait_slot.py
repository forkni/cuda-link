"""The Python spin-then-sleep wait loop (importer._python_wait_slot), without an Importer."""

from __future__ import annotations

import time

import pytest

from cuda_link import importer as imp_mod
from cuda_link._profile import STATS_LOG_EVERY
from cuda_link.importer import _python_wait_slot

_EVT = object()


def _ready_after(calls_before_ready: int):
    seen = []

    def query(evt: object) -> bool:
        seen.append(evt)
        return len(seen) > calls_before_ready

    return query, seen


@pytest.fixture
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(imp_mod.time, "sleep", slept.append)
    return slept


def test_spin_phase_hit_reports_ready_spin_and_no_blocked_time() -> None:
    query, seen = _ready_after(0)
    start = time.perf_counter()

    status, waited_us, blocked_us, slept = _python_wait_slot(query, _EVT, start, 200, start + 1.0)

    assert status == "READY_SPIN"
    assert waited_us >= 0.0
    assert blocked_us == 0.0
    assert slept is False
    assert seen == [_EVT]


def test_sleep_phase_hit_after_one_sleep_reports_ready_poll(no_real_sleep: list[float]) -> None:
    query, _ = _ready_after(1)
    start = time.perf_counter()

    status, waited_us, blocked_us, slept = _python_wait_slot(query, _EVT, start, 0, start + 1.0)

    assert status == "READY_POLL"
    assert slept is True
    assert no_real_sleep == [0.0001]
    assert 0.0 <= blocked_us <= waited_us


def test_sleep_phase_hit_on_the_first_query_did_not_sleep(no_real_sleep: list[float]) -> None:
    query, _ = _ready_after(0)
    start = time.perf_counter()

    status, _, _, slept = _python_wait_slot(query, _EVT, start, 0, start + 1.0)

    assert status == "READY_POLL"
    assert slept is False
    assert no_real_sleep == []


def test_deadline_in_the_spin_phase_is_a_timeout() -> None:
    start = time.perf_counter()

    status, waited_us, _, _ = _python_wait_slot(lambda evt: False, _EVT, start, 10_000_000, start + 0.002)

    assert status == "TIMEOUT"
    assert waited_us >= 2_000.0


def test_deadline_in_the_sleep_phase_is_a_timeout(no_real_sleep: list[float]) -> None:
    start = time.perf_counter()

    status, waited_us, _, _ = _python_wait_slot(lambda evt: False, _EVT, start, 0, start + 0.002)

    assert status == "TIMEOUT"
    assert waited_us >= 2_000.0


def test_stats_log_interval_is_a_prime() -> None:
    """A prime interval does not lock onto the slot count or a round frame rate."""
    assert STATS_LOG_EVERY > 1
    assert all(STATS_LOG_EVERY % d for d in range(2, STATS_LOG_EVERY))
