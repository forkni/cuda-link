"""The connect retry and backoff state both receivers use (importer._RetryState)."""

from __future__ import annotations

from cuda_link.importer import BackoffNotice, _RetryState


def test_first_tick_is_due_and_counts_the_attempt() -> None:
    retry = _RetryState()

    assert retry.tick() is True
    assert retry.connect_attempts == 1
    assert retry.frames_since_last_retry == 0


def test_tick_waits_out_the_interval() -> None:
    retry = _RetryState(retry_interval_frames=3)

    ticks = [retry.tick() for _ in range(3)]

    assert ticks == [False, False, True]
    assert retry.connect_attempts == 1


def test_back_off_walks_the_intervals_then_stays_on_the_last() -> None:
    retry = _RetryState(backoff_intervals=(1, 2, 4))
    seen = []

    for _ in range(5):
        retry.request_immediate_reconnect()
        retry.tick()
        retry.back_off()
        seen.append(retry.retry_interval_frames)

    assert seen == [2, 4, 4, 4, 4]


def test_back_off_reports_up_to_the_limit_then_once_more_then_never() -> None:
    retry = _RetryState(max_connect_attempts=2)
    notices = []

    for _ in range(5):
        retry.request_immediate_reconnect()
        retry.tick()
        notices.append(retry.back_off())

    assert notices == [
        BackoffNotice.WAITING,
        BackoffNotice.WAITING,
        BackoffNotice.NOW_SILENT,
        BackoffNotice.SILENT,
        BackoffNotice.SILENT,
    ]


def test_reset_makes_the_next_attempt_immediate() -> None:
    retry = _RetryState(connect_attempts=9, retry_interval_frames=120, frames_since_last_retry=7)

    retry.reset()

    assert (retry.connect_attempts, retry.retry_interval_frames, retry.frames_since_last_retry) == (0, 1, 0)
    assert retry.tick() is True
