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
    engine._tx_start = 1.0
    engine._export_total_s = 0.015  # 15 ms total
    engine._maybe_report_stats(write_idx=150, now=2.0)
    assert logs == []


def test_silent_when_not_multiple_of_report_every():
    """No report emitted for frame counts that are not multiples of report_every."""
    engine, logs = _make_engine(report_every=150)
    _inject_exporter(engine, frame_count=149)
    engine._tx_start = 1.0
    engine._export_total_s = 0.01
    engine._maybe_report_stats(write_idx=149, now=2.0)
    assert logs == []


def test_report_emitted_at_report_every(monkeypatch):
    """A single line is emitted at frame_count == report_every."""
    engine, logs = _make_engine(report_every=150, verbose=True)
    _inject_exporter(engine, frame_count=150)
    engine._tx_start = 1.0  # session started at t=1.0
    engine._tx_start_frame = 0
    engine._export_total_s = 0.015  # 15 ms total → avg 100 µs/frame

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


def test_consecutive_reports_use_windowed_fps():
    """The second report computes FPS over its own window, not lifetime."""
    engine, logs = _make_engine(report_every=150, verbose=True)
    # First report: window [t=1.0, t=4.0], 150 frames → 50 FPS
    _inject_exporter(engine, frame_count=150)
    engine._tx_start = 1.0
    engine._tx_start_frame = 0
    engine._export_total_s = 0.015
    engine._maybe_report_stats(write_idx=150, now=4.0)

    # Second report: window [t=4.0, t=9.0], 150 frames → 30 FPS
    engine._exporter.frame_count = 300
    engine._export_total_s = 0.030
    engine._maybe_report_stats(write_idx=300, now=9.0)

    assert len(logs) == 2
    # First window: 150 frames / 3.0 s → 50 FPS
    assert "50.0 FPS" in logs[0]
    # Second window: 150 frames / 5.0 s → 30 FPS
    assert "30.0 FPS" in logs[1]
    # avg µs should be lifetime average: 30 ms / 300 frames = 100 µs
    assert "export=100.0 µs avg" in logs[1]


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
    engine._tx_start = 1.0
    engine._tx_last_report_t = 4.0
    engine._export_total_s = 0.015
    engine._tx_start_frame = 0
    engine._tx_last_report_frame = 150

    # cleanup requires _barrier.force_release + _barrier.close — stub them.

    engine._barrier = types.SimpleNamespace(
        force_release=lambda pid, log_fn=None: None,
        close=lambda: None,
    )
    engine.cleanup()

    assert engine._tx_start == 0.0
    assert engine._tx_last_report_t == 0.0
    assert engine._export_total_s == 0.0
    assert engine._tx_start_frame == 0
    assert engine._tx_last_report_frame == 0
