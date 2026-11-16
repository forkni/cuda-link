"""
Characterization tests: TDReceiver's exponential-backoff retry logic (TDReceiver.py:403-422)
and its RetryState value object (TDReceiver.py:184-212).

import_frame() lazily connects to the sender's SharedMemory. Each failed
initialize_receiver() call walks `backoff_intervals=(1,2,4,8,16,32,64,120)` (L190) via
`min(connect_attempts, len(backoff_intervals) - 1)` -- so the wait grows 1->2->4->8->16->
32->64->120 frames and then clamps at 120 forever. The "Waiting for sender..." log fires
on every attempt while `connect_attempts <= max_connect_attempts` (20); exactly one more
log ("Sender not found. Will keep retrying silently.") fires the instant attempts exceed
20; every attempt after that logs nothing at all (silent retries).

GPU-free harness: initialize_receiver() itself first calls get_cuda_runtime() (real CUDA
driver load) before ever touching SharedMemory, so it cannot be exercised end-to-end
without a GPU. Instead this test monkeypatches the *bound* TDReceiverEngine.initialize_receiver
to always fail, isolating exactly the retry/backoff bookkeeping in import_frame() (the
target under test) from CUDA/SHM connection mechanics (covered elsewhere / behind
requires_cuda). request_immediate_reconnect() (TDReceiver.py:196-198) is used to force
each call to attempt reconnection immediately, regardless of the current
retry_interval_frames wait -- avoiding an actual multi-hundred-frame simulation loop.
"""

from __future__ import annotations

import pytest
from fakes import FakeTDHost
from TDConfig import TDReceiverConfig
from TDReceiver import TDReceiverEngine


def _make_engine(log_calls: list[str]) -> TDReceiverEngine:
    host = FakeTDHost(params={"Active": True})

    def _log_fn(msg: str, force: bool = False) -> None:  # noqa: ARG001 -- force accepted, not asserted on
        log_calls.append(msg)

    return TDReceiverEngine(
        host=host,
        config=TDReceiverConfig(),
        cuda=None,
        log_fn=_log_fn,
        num_slots=1,
        device=0,
        shm_name="nonexistent_shm_for_retry_backoff_test",
        verbose=False,
    )


def _force_attempt(engine: TDReceiverEngine, log_calls: list[str]) -> None:
    """Force the next import_frame() call to attempt reconnection immediately."""
    engine.request_immediate_reconnect()
    log_calls.clear()
    assert engine.import_frame(handle=None) is False


def test_backoff_walks_table_then_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    log_calls: list[str] = []
    engine = _make_engine(log_calls)
    monkeypatch.setattr(engine, "initialize_receiver", lambda: False)

    backoff_intervals = (1, 2, 4, 8, 16, 32, 64, 120)
    # After attempt N, retry_interval_frames == backoff_intervals[min(N, len-1)].
    expected_after_attempt = [backoff_intervals[min(n, len(backoff_intervals) - 1)] for n in range(1, 10)]

    for attempt, expected_interval in enumerate(expected_after_attempt, start=1):
        _force_attempt(engine, log_calls)
        assert engine._retry.connect_attempts == attempt
        assert engine._retry.retry_interval_frames == expected_interval, (
            f"After attempt {attempt}, retry_interval_frames must be {expected_interval} "
            f"(backoff_intervals[min({attempt}, 7)])."
        )

    # Once clamped at the table's last entry (120), further attempts must stay there.
    assert engine._retry.retry_interval_frames == backoff_intervals[-1]


def test_wait_gate_blocks_reattempt_before_interval_elapses(monkeypatch: pytest.MonkeyPatch) -> None:
    """frames_since_last_retry must reach retry_interval_frames before another attempt fires --
    intervening import_frame() calls must not touch initialize_receiver() at all."""
    log_calls: list[str] = []
    engine = _make_engine(log_calls)
    calls = {"n": 0}

    def _fail() -> bool:
        calls["n"] += 1
        return False

    monkeypatch.setattr(engine, "initialize_receiver", _fail)

    _force_attempt(engine, log_calls)  # attempt 1 -> retry_interval_frames becomes 2
    assert calls["n"] == 1
    assert engine._retry.retry_interval_frames == 2

    # Next call: frames_since_last_retry (1) < retry_interval_frames (2) -> must wait,
    # NOT re-invoke initialize_receiver().
    assert engine.import_frame(handle=None) is False
    assert calls["n"] == 1, "Must not attempt reconnection before the backoff interval elapses."

    # Second waiting call reaches the interval -> attempt fires.
    assert engine.import_frame(handle=None) is False
    assert calls["n"] == 2
    assert engine._retry.connect_attempts == 2


def test_logging_stops_silently_past_max_connect_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempts 1..20 log 'Waiting for sender...' every time; attempt 21 logs exactly the
    'Sender not found... retrying silently' message once; attempts 22+ log nothing at all."""
    log_calls: list[str] = []
    engine = _make_engine(log_calls)
    monkeypatch.setattr(engine, "initialize_receiver", lambda: False)

    for attempt in range(1, 21):
        _force_attempt(engine, log_calls)
        assert engine._retry.connect_attempts == attempt
        assert any("Waiting for sender" in m for m in log_calls), f"Attempt {attempt} must log a waiting message."

    # Attempt 21: exceeds max_connect_attempts (20) -> the one-time "silent from now on" log.
    _force_attempt(engine, log_calls)
    assert engine._retry.connect_attempts == 21
    assert not any("Waiting for sender" in m for m in log_calls)
    assert any("keep retrying silently" in m for m in log_calls)

    # Attempt 22: fully silent -- no log call of either kind.
    _force_attempt(engine, log_calls)
    assert engine._retry.connect_attempts == 22
    assert log_calls == [], "Past max_connect_attempts + 1, retries must be completely silent."
