"""Coverage tests for Importer._resolve_stream(), Importer._log_frame_stats(),
and the _HighResTimer Windows timer-resolution branches.

All GPU-free: uses fakes.make_connected_importer() (FakeCUDAAdapter-backed).
_HighResTimer's winmm calls are monkeypatched with a MagicMock stand-in rather
than relying on the real module-level ``_winmm`` global, which is None on this
interpreter (CPython 3.11+ doesn't need timeBeginPeriod — see the module-level
guard at the top of importer.py) and would otherwise make the body of
__enter__/__exit__ unreachable in any real run on this machine.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fakes import make_connected_importer

import cuda_link.importer as _imp_mod

# ---------------------------------------------------------------------------
# _resolve_stream()
# ---------------------------------------------------------------------------


def test_resolve_stream_passthrough_int() -> None:
    imp = make_connected_importer()

    assert imp._resolve_stream(42) == 42


def test_resolve_stream_torch_stream_uses_cuda_stream_attr() -> None:
    imp = make_connected_importer()
    fake_stream = SimpleNamespace(cuda_stream=0xABCD)

    assert imp._resolve_stream(fake_stream) == 0xABCD


def test_resolve_stream_cupy_like_uses_ptr_attr() -> None:
    imp = make_connected_importer()
    fake_stream = SimpleNamespace(ptr=0x1234)  # no cuda_stream attr

    assert imp._resolve_stream(fake_stream) == 0x1234


def test_resolve_stream_raises_type_error_for_unsupported_type() -> None:
    imp = make_connected_importer()

    with pytest.raises(TypeError, match="Unsupported stream type"):
        imp._resolve_stream(object())


# ---------------------------------------------------------------------------
# _log_frame_stats()
# ---------------------------------------------------------------------------


def test_log_frame_stats_logs_at_97th_frame_with_gpu_events(caplog: pytest.LogCaptureFixture) -> None:
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    imp.frame_count = 97
    imp.wait_spin_hits = 50
    imp.wait_sleep_hits = 47
    caplog.set_level(logging.DEBUG, logger="cuda_link.importer")

    imp._log_frame_stats("torch", 0, imp._conn)  # must not raise

    assert any("GPU-Events" in r.message for r in caplog.records)


def test_log_frame_stats_logs_at_97th_frame_with_cpu_sync(caplog: pytest.LogCaptureFixture) -> None:
    imp = make_connected_importer(with_events=False, dev_ptr_style="mock")
    imp.frame_count = 194  # 97 * 2
    imp.wait_spin_hits = 0
    imp.wait_sleep_hits = 0
    caplog.set_level(logging.DEBUG, logger="cuda_link.importer")

    imp._log_frame_stats("numpy", 0, imp._conn)  # must not raise; wait_spin_hits=0 exercises the 0.0 fallback

    assert any("CPU-Sync" in r.message for r in caplog.records)


def test_log_frame_stats_noop_when_not_a_multiple_of_97(caplog: pytest.LogCaptureFixture) -> None:
    imp = make_connected_importer()
    imp.frame_count = 5
    caplog.set_level(logging.DEBUG, logger="cuda_link.importer")

    imp._log_frame_stats("torch", 0, imp._conn)

    assert not any("Frame 5 " in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _HighResTimer
# ---------------------------------------------------------------------------


def test_high_res_timer_logs_debug_and_restores_period_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a Windows host with winmm loaded (this interpreter's real
    module-level ``_winmm`` is None -- see module docstring above) so both the
    TIMERR_NOCANDO debug-log branch and the __exit__ teardown call execute."""
    fake_winmm = MagicMock()
    fake_winmm.timeBeginPeriod.return_value = 97  # TIMERR_NOCANDO
    monkeypatch.setattr(_imp_mod, "_winmm", fake_winmm)

    with _imp_mod._HighResTimer():
        pass

    fake_winmm.timeBeginPeriod.assert_called_once_with(1)
    fake_winmm.timeEndPeriod.assert_called_once_with(1)


def test_high_res_timer_skips_debug_log_when_timebeginperiod_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """timeBeginPeriod() returning 0 (success) must skip the TIMERR_NOCANDO
    debug-log branch while still tearing the period back down on exit."""
    fake_winmm = MagicMock()
    fake_winmm.timeBeginPeriod.return_value = 0
    monkeypatch.setattr(_imp_mod, "_winmm", fake_winmm)

    with _imp_mod._HighResTimer():
        pass

    fake_winmm.timeBeginPeriod.assert_called_once_with(1)
    fake_winmm.timeEndPeriod.assert_called_once_with(1)
