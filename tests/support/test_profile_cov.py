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


# ---------------------------------------------------------------------------
# mutation kill-tests (cosmic-ray survivor triage, src/cuda_link/_profile.py)
# ---------------------------------------------------------------------------


def test_avg_uses_true_division_kills_floordiv_mutant():
    """avg() must use float division, not floor division (line 32)."""
    fp = FrameProfile(("copy",))
    fp.record("copy", 1.0)
    assert fp.avg("copy", 3) == pytest.approx(1.0 / 3.0)


def test_avg_zero_frames_returns_exact_zero_kills_numberreplacer_and_comparison_mutants():
    """avg(region, 0) must short-circuit to exactly 0.0 (line 32).

    Kills four line-32 survivors at once: the ``else 0.0`` literal swapped to
    ``1.0``/``-1.0``, and the ``n > 0`` guard weakened to ``n > -1``/``n >= 0``
    (both of which let n == 0 fall through to a ZeroDivisionError here).
    """
    fp = FrameProfile(("copy",))
    fp.record("copy", 5.0)
    assert fp.avg("copy", 0) == 0.0


def test_avg_negative_n_returns_zero_kills_gt_noteq_mutant():
    """avg() with a negative n must still hit the else-branch (line 32)."""
    fp = FrameProfile(("copy",))
    fp.record("copy", 5.0)
    assert fp.avg("copy", -1) == 0.0


def test_avg_single_frame_computes_average_kills_numberreplacer_mutant():
    """avg(region, 1) must take the division branch, not the else (line 32)."""
    fp = FrameProfile(("copy",))
    fp.record("copy", 7.0)
    assert fp.avg("copy", 1) == pytest.approx(7.0)


def test_report_single_frame_formats_average_kills_numberreplacer_mutant():
    """report(1) must format the average, not short-circuit to '' (line 36)."""
    fp = FrameProfile(("copy",))
    fp.record("copy", 4.0)
    assert fp.report(1) == "copy=4.0"


def test_started_true_for_negative_start_t_kills_noteq_gt_mutant():
    """started must be True for any non-zero start_t, including negative (line 62)."""
    rw = ReportWindow()
    rw.start(-5.0, 3)
    assert rw.started is True


def test_fps_no_reset_when_last_t_positive_kills_eq_gte_mutant():
    """fps() must only re-seed when last_t is exactly 0.0, not merely >= 0.0 (line 71)."""
    rw = ReportWindow()
    rw.start(100.0, 1)
    rw.fps(110.0, 2)  # seeds the window; last_t becomes 110.0 (positive, nonzero)
    fps2 = rw.fps(200.0, 5)  # last_t == 110.0 -> must NOT re-seed
    assert fps2 == pytest.approx(3.0 / 90.0)


def test_fps_no_reset_when_last_t_negative_kills_eq_lte_mutant():
    """fps() must only re-seed when last_t is exactly 0.0, not merely <= 0.0 (line 71)."""
    rw = ReportWindow()
    rw.start(-50.0, 1)
    rw.fps(-40.0, 100)  # seeds the window; last_t becomes -40.0 (negative, nonzero)
    fps2 = rw.fps(-5.0, 105)  # last_t == -40.0 -> must NOT re-seed
    assert fps2 == pytest.approx(5.0 / 35.0)


def test_fps_uses_true_division_kills_floordiv_mutant():
    """fps() must use float division, not floor division (line 75)."""
    rw = ReportWindow()
    rw.start(0.0, 0)
    fps = rw.fps(3.0, 1)  # 1 frame / 3.0 s -> non-integer result
    assert fps == pytest.approx(1.0 / 3.0)


def test_fps_dt_equals_one_computes_fps_kills_numberreplacer_mutant():
    """fps() must take the division branch when dt == 1.0, not only dt > 1 (line 75)."""
    rw = ReportWindow()
    rw.start(0.0, 0)
    fps = rw.fps(1.0, 1)
    assert fps == 1.0


def test_reset_zeroes_frame_counters_kills_numberreplacer_mutants():
    """reset() must clear start_frame and last_frame to exactly 0 (lines 98, 100).

    Kills four survivors: start_frame reset to -1/1 instead of 0, and
    last_frame reset to 1/-1 instead of 0.
    """
    rw = ReportWindow()
    rw.start(5.0, 3)
    rw.fps(6.0, 4)
    rw.add_sample(1.0)
    rw.reset()
    assert rw.start_frame == 0
    assert rw.last_frame == 0
