"""Unit tests for the `.tox` Receiver Framewaitms cadence guard.

PublishCadence / cadence_add / cadence_wait_useful in TDReceiver are a port of the same
names in cpp_top/src/core/frame_wait.h; these cases match test_frame_wait_cadence in
cpp_top/tests/topcore_tests.cpp. The engine tests check that wait_for_new_frame() skips the
doorbell wait when the guard says it cannot get a frame, and still waits when it can.
"""

from __future__ import annotations

import time

import pytest

from tests.td.test_tdreceiver_frame_wait import _connected_engine, _FakeDoorbell
from tests.td.test_tdreceiver_has_new_frame import shm_cleanup  # noqa: F401 -- pytest fixture

K60 = 1.0 / 60.0
K30 = 1.0 / 30.0


def _feed(c, t0: float, period: float, n: int) -> float:
    from TDReceiver import cadence_add

    t = t0
    for i in range(n):
        t = t0 + period * i
        cadence_add(c, t, 1)
    return t


def _cadence():
    from TDReceiver import PublishCadence

    return PublishCadence()


class TestCadencePure:
    def test_unknown_cadence_waits(self) -> None:
        from TDReceiver import cadence_wait_useful

        assert cadence_wait_useful(_cadence(), 100.0, 3)
        few = _cadence()
        _feed(few, 100.0, K60, 3)
        assert few.samples == 2
        assert cadence_wait_useful(few, 1000.0, 3)

    def test_no_timestamp_leaves_cadence_unknown(self) -> None:
        from TDReceiver import cadence_add

        c = _cadence()
        for _ in range(10):
            cadence_add(c, 0.0, 1)
        assert c.samples == 0
        assert c.last_ts == 0.0

    def test_60fps_sender(self) -> None:
        from TDReceiver import CADENCE_MIN_SAMPLES, cadence_wait_useful

        c = _cadence()
        last = _feed(c, 100.0, K60, 20)
        assert c.samples == CADENCE_MIN_SAMPLES
        assert c.period_s == pytest.approx(K60, rel=1e-6)
        assert cadence_wait_useful(c, last + K60 - 0.0027, 3)  # publish 2.7 ms away
        assert not cadence_wait_useful(c, last + K60 - 0.005, 3)  # 5 ms away: out of budget
        assert cadence_wait_useful(c, last + 1.5 * K60, 3)  # late publish, under 2 periods
        assert not cadence_wait_useful(c, last + 2.5 * K60, 3)  # paused sender
        assert not cadence_wait_useful(c, last + K60 - 0.0027, 0)

    def test_30fps_sender_into_60fps(self) -> None:
        from TDReceiver import cadence_wait_useful

        c = _cadence()
        last = _feed(c, 100.0, K30, 20)
        assert c.period_s == pytest.approx(K30, rel=1e-6)
        assert not cadence_wait_useful(c, last + K60, 3)  # the repeat frame skips
        assert cadence_wait_useful(c, last + K30 - 0.002, 3)  # just before the next publish

    def test_ring_gap_keeps_period_per_publish(self) -> None:
        from TDReceiver import cadence_add

        c = _cadence()
        _feed(c, 100.0, K60, 10)
        cadence_add(c, c.last_ts + 2 * K60, 2)
        assert c.period_s == pytest.approx(K60, rel=1e-6)

    def test_one_pause_is_an_outlier(self) -> None:
        from TDReceiver import cadence_add

        c = _cadence()
        _feed(c, 100.0, K60, 10)
        cadence_add(c, c.last_ts + 0.2, 1)
        assert c.period_s == pytest.approx(K60, rel=1e-6)
        assert c.outliers == 1
        cadence_add(c, c.last_ts + K60, 1)
        assert c.outliers == 0
        cadence_add(c, c.last_ts + 5.0, 1)  # 1 s or more: never a sample
        assert c.period_s == pytest.approx(K60, rel=1e-6)

    def test_rate_change_starts_over(self) -> None:
        from TDReceiver import CADENCE_MAX_OUTLIERS, CADENCE_MIN_SAMPLES, cadence_add

        c = _cadence()
        t = _feed(c, 100.0, K60, 10)
        for _ in range(CADENCE_MAX_OUTLIERS):
            t += 0.1
            cadence_add(c, t, 1)
        assert c.samples == 0
        for _ in range(6):
            t += 0.1
            cadence_add(c, t, 1)
        assert c.samples == CADENCE_MIN_SAMPLES
        assert c.period_s == pytest.approx(0.1, rel=1e-3)


class TestEngineGuard:
    def test_paused_sender_skips_the_wait(self, shm_cleanup, monkeypatch) -> None:  # noqa: F811
        shm = shm_cleanup
        fake = _FakeDoorbell()
        engine = _connected_engine(shm, monkeypatch, fake)
        # 60 fps cadence whose newest frame is 0.1 s old: paused.
        _feed(engine.publish_cadence, time.perf_counter() - 0.1 - 19 * K60, K60, 20)
        assert engine.wait_for_new_frame(3) is False
        assert fake.wait_calls == []
        assert engine.frame_wait_skipped_count == 1
        assert engine.get_stats()["rx_frame_wait_skipped"] == 1

    def test_due_publish_still_waits(self, shm_cleanup, monkeypatch) -> None:  # noqa: F811
        shm = shm_cleanup
        fake = _FakeDoorbell()
        engine = _connected_engine(shm, monkeypatch, fake)
        # Newest frame 15 ms old at 60 fps: the next publish is due within the budget.
        _feed(engine.publish_cadence, time.perf_counter() - 0.015 - 19 * K60, K60, 20)
        engine.wait_for_new_frame(3)
        assert fake.wait_calls, "the guard must not skip a wait that can get a frame"
        assert engine.frame_wait_skipped_count == 0

    def test_close_resets_the_cadence(self, shm_cleanup, monkeypatch) -> None:  # noqa: F811
        shm = shm_cleanup
        engine = _connected_engine(shm, monkeypatch, _FakeDoorbell())
        _feed(engine.publish_cadence, 100.0, K60, 20)
        engine._release_resources()
        assert engine.publish_cadence.samples == 0
