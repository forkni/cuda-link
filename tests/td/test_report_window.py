"""
Unit tests for ReportWindow.add_sample / avg_and_reset.

These tests lock the shared windowed-average mechanism that both TDSenderEngine
(export-µs) and TDReceiverEngine (copy-µs) now use.  All tests are GPU-free
and import directly from the canonical src/cuda_link/_profile module.
"""

from __future__ import annotations

from cuda_link._profile import ReportWindow

# ---------------------------------------------------------------------------
# add_sample / avg_and_reset basics
# ---------------------------------------------------------------------------


def test_avg_and_reset_returns_mean_and_clears():
    """avg_and_reset() returns the mean of all accumulated samples and zeros the accumulator."""
    w = ReportWindow()
    w.add_sample(0.0001)  # 100 µs
    w.add_sample(0.0002)  # 200 µs
    w.add_sample(0.0003)  # 300 µs

    result = w.avg_and_reset()

    assert abs(result - 0.0002) < 1e-12, f"Expected mean 0.0002 s, got {result}"
    # Accumulator must be cleared
    assert w._acc_sum == 0.0
    assert w._acc_n == 0


def test_avg_and_reset_empty_returns_zero():
    """avg_and_reset() on a fresh or cleared window returns 0.0 without dividing by zero."""
    w = ReportWindow()
    assert w.avg_and_reset() == 0.0


def test_add_sample_accumulates():
    """Multiple add_sample calls accumulate into _acc_sum and _acc_n."""
    w = ReportWindow()
    for _ in range(150):
        w.add_sample(0.0001)  # 100 µs each

    assert w._acc_n == 150
    assert abs(w._acc_sum - 0.015) < 1e-12


def test_reset_clears_accumulator():
    """reset() zeroes _acc_sum and _acc_n as well as the FPS window state."""
    w = ReportWindow()
    w.add_sample(0.005)
    w.add_sample(0.010)
    w.start(1.0, 0)

    w.reset()

    assert w._acc_sum == 0.0
    assert w._acc_n == 0
    assert w.start_t == 0.0
    assert w.last_t == 0.0


# ---------------------------------------------------------------------------
# Two-window independence — the core "windowed, not lifetime" lock
# ---------------------------------------------------------------------------


def test_two_windows_are_independent():
    """The second window's avg_and_reset() reflects only samples added after the first call.

    This is the critical regression lock: if the accumulator were not cleared
    by avg_and_reset(), the second call would return a lifetime average, not a
    windowed one.
    """
    w = ReportWindow()

    # Window 1: 150 samples at 100 µs each → avg = 100 µs
    for _ in range(150):
        w.add_sample(0.0001)
    avg1 = w.avg_and_reset()
    assert abs(avg1 - 0.0001) < 1e-12, f"Window 1 expected 100 µs mean, got {avg1}"

    # Window 2: 150 samples at 200 µs each → avg should be 200 µs (NOT 150 µs lifetime)
    for _ in range(150):
        w.add_sample(0.0002)
    avg2 = w.avg_and_reset()
    assert abs(avg2 - 0.0002) < 1e-12, (
        f"Window 2 expected 200 µs mean (windowed), got {avg2}.\n"
        "If avg2 ≈ 150 µs, the accumulator was not cleared between windows — "
        "avg_and_reset() must zero _acc_sum and _acc_n."
    )


def test_zero_value_samples():
    """add_sample(0.0) is valid and does not corrupt the accumulator."""
    w = ReportWindow()
    for _ in range(10):
        w.add_sample(0.0)
    assert w.avg_and_reset() == 0.0


def test_single_sample():
    """A single sample returns its own value."""
    w = ReportWindow()
    w.add_sample(0.00073)
    assert abs(w.avg_and_reset() - 0.00073) < 1e-15
