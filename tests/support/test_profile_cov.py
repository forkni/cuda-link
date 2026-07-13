"""Coverage-focused tests for cuda_link._profile.

tests/td/test_report_window.py already covers add_sample/avg_and_reset/reset.
This file fills the remaining gaps: FrameProfile.report(), the
ReportWindow.started property, and ReportWindow.fps().
"""

from __future__ import annotations

import pytest

from cuda_link._profile import FrameProfile, ReportWindow

# ---------------------------------------------------------------------------
# FrameProfile.report()
# ---------------------------------------------------------------------------


def test_report_returns_empty_string_for_zero_frames():
    fp = FrameProfile(("copy", "wait"))
    fp.record("copy", 10.0)
    assert fp.report(0) == ""


def test_report_returns_empty_string_for_negative_frames():
    fp = FrameProfile(("copy",))
    assert fp.report(-1) == ""


def test_report_formats_averages_per_region():
    fp = FrameProfile(("copy", "wait"))
    fp.record("copy", 20.0)
    fp.record("copy", 20.0)
    fp.record("wait", 5.0)
    report = fp.report(2)
    assert report == "copy=20.0 wait=2.5"


def test_avg_matches_report_computation():
    fp = FrameProfile(("copy",))
    fp.record("copy", 30.0)
    assert fp.avg("copy", 3) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# ReportWindow.started
# ---------------------------------------------------------------------------


def test_started_false_before_start_called():
    rw = ReportWindow()
    assert rw.started is False


def test_started_true_after_start_called():
    rw = ReportWindow()
    rw.start(100.0, 5)
    assert rw.started is True


# ---------------------------------------------------------------------------
# ReportWindow.fps()
# ---------------------------------------------------------------------------


def test_fps_first_call_seeds_from_session_baseline():
    rw = ReportWindow()
    rw.start(100.0, 0)
    # First fps() call: last_t is 0.0, so it seeds from start_t/start_frame.
    fps = rw.fps(110.0, 100)
    assert fps == pytest.approx(10.0)  # 100 frames / 10s
    assert rw.last_t == 110.0
    assert rw.last_frame == 100


def test_fps_subsequent_call_advances_window():
    rw = ReportWindow()
    rw.start(0.0, 0)
    rw.fps(10.0, 100)  # seeds + first window: 100 frames / 10s = 10 fps
    fps2 = rw.fps(15.0, 150)  # second window: 50 frames / 5s = 10 fps
    assert fps2 == pytest.approx(10.0)
    assert rw.last_t == 15.0
    assert rw.last_frame == 150


def test_fps_returns_zero_when_dt_not_positive():
    rw = ReportWindow()
    rw.start(50.0, 10)
    # now == last_t (seeded from start_t) -> dt == 0 -> fps 0.0
    fps = rw.fps(50.0, 20)
    assert fps == 0.0


def test_fps_returns_zero_when_dt_negative():
    rw = ReportWindow()
    rw.start(50.0, 10)
    fps = rw.fps(40.0, 20)  # now < last_t -> dt negative
    assert fps == 0.0
