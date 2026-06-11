"""
Regression tests: TDSenderEngine._maybe_report_stats windowed periodic log.

The method emits a Debug summary line every CUDALINK_SENDER_REPORT_EVERY published
frames (default 150) when verbose_performance is True, mirroring TDReceiverEngine's
windowed FPS / copy-µs line.  It is deliberately decoupled from CUDALINK_EXPORT_PROFILE
(which is off by default) — FPS and avg-export-µs come from engine-local timers.

All tests are GPU-free: they construct a TDSenderEngine with a stubbed Exporter shim
and call _maybe_report_stats directly so no cuda_memory() / CUDA path is exercised.
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(report_every: int | None = 150, verbose: bool = True, shm_name: str | None = None):
    """Build a TDSenderEngine with minimal stubs and a log-capture list."""
    from TDConfig import TDSenderConfig
    from TDHost import TDHost
    from TDSender import TDSenderEngine

    class _SpyHost(TDHost):
        def param_value(self, name):
            return {"Active": True, "Debug": False}.get(name)

        def set_param_value(self, name, value):
            pass

        def set_param_enabled(self, name, enabled):
            pass

        def show_custom_only(self, value):
            pass

        def is_active(self):
            return True

        def find_top(self, name):
            return None

        def set_warning_status(self, msg):
            pass

        def set_error_status(self, msg):
            pass

        def clear_status(self):
            pass

        def set_info_status(self, msg):
            pass

    shm = shm_name or f"test_report_{uuid.uuid4().hex[:8]}"
    logs: list[str] = []
    engine = TDSenderEngine(
        host=_SpyHost(),
        config=TDSenderConfig(),
        cuda=None,
        log_fn=lambda msg, force=False: logs.append(msg),
        num_slots=3,
        device=0,
        shm_name=shm,
        verbose=verbose,
    )
    if report_every is not None:
        engine._tx_report_every = report_every
    return engine, logs


def _inject_exporter(engine, frame_count: int, shape=(1080, 1920, 4), dtype: str = "uint8"):
    """Attach a minimal stub Exporter so frame_count property + _current_spec work."""
    from TDSender import FrameSpec  # noqa: F401

    stub = types.SimpleNamespace(frame_count=frame_count)
    engine._exporter = stub

    engine._current_spec = FrameSpec(
        shm_name=engine.shm_name,
        height=shape[0],
        width=shape[1],
        channels=shape[2],
        dtype=dtype,
        num_slots=engine.num_slots,
        device=engine.device,
    )


# ---------------------------------------------------------------------------
# Correctness cases
# ---------------------------------------------------------------------------


def test_silent_when_verbose_false():
    """No report emitted when verbose_performance is off regardless of frame count."""
    engine, logs = _make_engine(verbose=False)
    _inject_exporter(engine, frame_count=150)
    engine._tx_window.start(1.0, 0)
    engine._tx_window._acc_sum = 0.015
    engine._tx_window._acc_n = 150
    engine._maybe_report_stats(write_idx=150, now=2.0)
    assert logs == []


def test_silent_when_not_multiple_of_report_every():
    """No report emitted for frame counts that are not multiples of report_every."""
    engine, logs = _make_engine(report_every=150)
    _inject_exporter(engine, frame_count=149)
    engine._tx_window.start(1.0, 0)
    engine._tx_window._acc_sum = 0.01
    engine._tx_window._acc_n = 149
    engine._maybe_report_stats(write_idx=149, now=2.0)
    assert logs == []


def test_report_emitted_at_report_every(monkeypatch):
    """A single line is emitted at frame_count == report_every."""
    engine, logs = _make_engine(report_every=150, verbose=True)
    _inject_exporter(engine, frame_count=150)
    engine._tx_window.start(1.0, 0)  # session started at t=1.0, frame baseline 0
    # 150 samples × 100 µs = 15 ms total → avg_and_reset() = 100 µs/frame
    engine._tx_window._acc_sum = 0.015
    engine._tx_window._acc_n = 150

    engine._maybe_report_stats(write_idx=150, now=4.0)  # 3-second window → 50 FPS

    assert len(logs) == 1
    line = logs[0]
    assert "Frame   150" in line
    assert "50.0 FPS" in line
    assert "shape=(1080, 1920, 4)" in line
    assert "dtype=uint8" in line
    assert "export=100.0 µs avg" in line
    assert "write_idx=150" in line


def test_report_not_emitted_on_frame_zero():
    """frame_count == 0 does not emit even if it is a multiple of report_every=1."""
    engine, logs = _make_engine(report_every=1, verbose=True)
    _inject_exporter(engine, frame_count=0)
    engine._maybe_report_stats(write_idx=0, now=1.0)
    assert logs == []


def test_consecutive_reports_use_windowed_fps_and_avg():
    """The second report computes both FPS and export-avg over its own window, not lifetime.

    This is the core regression lock for windowed averaging: window 1 costs 100 µs/frame,
    window 2 costs 200 µs/frame.  Before the windowed-avg change the second report would
    have shown the lifetime avg (150 µs); it must now show the windowed 200 µs.
    """
    engine, logs = _make_engine(report_every=150, verbose=True)
    # First report: window [t=1.0, t=4.0], 150 frames → 50 FPS, 100 µs/frame
    _inject_exporter(engine, frame_count=150)
    engine._tx_window.start(1.0, 0)
    engine._tx_window._acc_sum = 0.015  # 150 frames × 100 µs = 15 ms
    engine._tx_window._acc_n = 150
    engine._maybe_report_stats(write_idx=150, now=4.0)
    # avg_and_reset() consumed and cleared the accumulator; seed window 2 independently.

    # Second report: window [t=4.0, t=9.0], 150 frames → 30 FPS, 200 µs/frame
    engine._exporter.frame_count = 300
    engine._tx_window._acc_sum = 0.030  # 150 frames × 200 µs = 30 ms (windowed, not cumulative)
    engine._tx_window._acc_n = 150
    engine._maybe_report_stats(write_idx=300, now=9.0)

    assert len(logs) == 2
    # First window: 150 frames / 3.0 s → 50 FPS, 100 µs avg
    assert "50.0 FPS" in logs[0]
    assert "export=100.0 µs avg" in logs[0]
    # Second window: 150 frames / 5.0 s → 30 FPS, 200 µs avg (windowed — NOT lifetime 150 µs)
    assert "30.0 FPS" in logs[1]
    assert "export=200.0 µs avg" in logs[1], (
        f"Second window must show windowed avg (200 µs), not lifetime avg.\nLine: {logs[1]!r}"
    )


def test_report_every_env_override(monkeypatch):
    """CUDALINK_SENDER_REPORT_EVERY env var is respected at engine construction time."""
    monkeypatch.setenv("CUDALINK_SENDER_REPORT_EVERY", "50")
    # Pass report_every=None so _make_engine does NOT override the env-sourced value.
    engine, logs = _make_engine(report_every=None, verbose=True)
    assert engine._tx_report_every == 50


def test_cleanup_resets_window_state():
    """cleanup() resets windowed-report state so the next activation starts fresh."""
    engine, logs = _make_engine()
    _inject_exporter(engine, frame_count=150)
    engine._tx_window.start(1.0, 0)
    engine._tx_window.last_t = 4.0
    engine._tx_window.last_frame = 150
    engine._export_total_s = 0.015

    # cleanup requires _barrier.force_release + _barrier.close — stub them.

    engine._barrier = types.SimpleNamespace(
        force_release=lambda pid, log_fn=None: None,
        close=lambda: None,
    )
    engine.cleanup()

    assert engine._tx_window.start_t == 0.0
    assert engine._tx_window.last_t == 0.0
    assert engine._tx_window.start_frame == 0
    assert engine._tx_window.last_frame == 0
    assert engine._tx_window._acc_sum == 0.0
    assert engine._tx_window._acc_n == 0


def test_reset_report_window_clears_state():
    """_reset_report_window() is callable standalone and zeros all window state.

    This exercises the helper used by the geometry/dtype-change reopen path — so
    tests can verify it exists and works independent of cleanup().
    """
    engine, _ = _make_engine()
    _inject_exporter(engine, frame_count=16500)
    engine._tx_window.start(0.0, 0)
    engine._tx_window.last_t = 100.0
    engine._tx_window.last_frame = 16350
    engine._tx_window._acc_sum = 16.5
    engine._tx_window._acc_n = 16500

    engine._reset_report_window()

    assert engine._tx_window.start_t == 0.0
    assert engine._tx_window.last_t == 0.0
    assert engine._tx_window.start_frame == 0
    assert engine._tx_window.last_frame == 0
    assert engine._tx_window._acc_sum == 0.0
    assert engine._tx_window._acc_n == 0


def test_format_change_reset_prevents_negative_fps():
    """Regression: dtype/geometry change reopens Exporter with frame_count=0.

    Without _reset_report_window(), ReportWindow.fps computes
        (new_frame_count − old_last_frame) / dt  ≈  (150 − 16500) / 2.5  =  -6540 FPS
    and avg_and_reset() would return the stale session-1 samples (16.5 s / 16500 frames = 1 ms),
    not the fresh session-2 measurements.  Both symptoms were observed in Log 3 after a
    uint8→float32 switch during a live TD Sender session (v1.10.3).

    The fix calls _reset_report_window() in the reopen block so the first windowed
    report after the format change shows a positive FPS and a fresh export average.
    """
    engine, logs = _make_engine(report_every=150, verbose=True)

    # Session 1: large lifetime frame count (simulates a long-running session).
    _inject_exporter(engine, frame_count=16500, dtype="uint8")
    engine._tx_window.start(0.0, 0)
    engine._tx_window.last_t = 100.0  # last report was at t=100 s
    engine._tx_window.last_frame = 16350
    engine._tx_window._acc_sum = 16.5  # 1 ms/frame avg over 16 500 frames
    engine._tx_window._acc_n = 16500

    # Simulate geometry/dtype change: Exporter reopened with fresh frame_count=0.
    # The fix calls _reset_report_window() here; replicate that:
    _inject_exporter(engine, frame_count=0, dtype="float32")
    engine._reset_report_window()  # clears FPS window and _acc_sum / _acc_n

    # First published frame of the new session seeds the window (mirrors export_frame:776-779).
    engine._exporter.frame_count = 1
    engine._tx_window.start(101.0, 0)  # start(now, frame_count - 1)

    # Advance to the first report boundary.
    engine._exporter.frame_count = 150
    # 150 samples × 100 µs/frame = 15 ms for the new float32 session.
    engine._tx_window._acc_sum = 0.015
    engine._tx_window._acc_n = 150

    # 2.5 s window → 150 / 2.5 = 60 FPS
    engine._maybe_report_stats(write_idx=150, now=103.5)

    assert len(logs) == 1, f"Expected 1 report line, got: {logs}"
    line = logs[0]
    # FPS token is the second |-delimited field; strip whitespace, grab the numeric part.
    fps_token = line.split("|")[1].strip().split()[0]
    fps_value = float(fps_token)
    assert fps_value > 0, (
        f"FPS must be positive after format-change reset, got {fps_value!r}.\n"
        f"Full line: {line!r}\n"
        "Root cause if failing: _reset_report_window() not called on Exporter reopen."
    )
    # Export average should reflect only the new session (100 µs), not the old total.
    assert "export=100.0 µs avg" in line, f"Export average should be fresh after reset (100.0 µs), got line: {line!r}"
