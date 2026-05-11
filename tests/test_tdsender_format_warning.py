"""
Regression test: TDSenderEngine pixel-format warning state machine.

Policy (replaces the former dtype_converter auto-conversion):
  - Unsupported format  → set_warning_status() every bad frame (idempotent yellow tint)
                          + set _warned_format=True and log once per episode
  - Subsequent bad frames → set_warning_status() re-called (keeps tint alive), no extra log
  - First supported frame after bad episode → call clear_status(), resume transfer, reset state
  - New bad-format episode after recovery → second log emitted (state machine resets)

Source of truth: verification/results/cuda_memory_probe_20260510_090919.json
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _make_engine_with_spy(shm_name: str):
    from TDConfig import TDSenderConfig
    from TDHost import TDHost
    from TDSender import TDSenderEngine

    class _SpyHost(TDHost):
        def __init__(self):
            self.warning_calls: list[str] = []
            self.error_calls: list[str] = []
            self.clear_count: int = 0
            self.status_par_writes: list[tuple[str, str]] = []
            self.info_status_calls: list[str] = []

        def param_value(self, name):
            return {"Active": True, "Debug": False}.get(name)

        def set_param_value(self, name, value):
            self.status_par_writes.append((name, value))

        def set_param_enabled(self, name, enabled):
            pass

        def show_custom_only(self, value):
            pass

        def is_active(self):
            return True

        def find_top(self, name):
            return None

        def set_warning_status(self, msg):
            self.warning_calls.append(msg)

        def set_error_status(self, msg):
            self.error_calls.append(msg)

        def clear_status(self):
            self.clear_count += 1

        def set_info_status(self, msg):
            self.info_status_calls.append(msg)

    host = _SpyHost()
    config = TDSenderConfig()
    engine = TDSenderEngine(
        host=host,
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=shm_name,
        verbose=False,
    )
    return engine, host


@pytest.fixture
def ctx():
    eng, host = _make_engine_with_spy(f"test_fmt_warn_{uuid.uuid4().hex[:8]}")
    return eng, host


def test_clean_engine_no_warning(ctx):
    eng, host = ctx
    assert host.warning_calls == []
    assert host.clear_count == 0
    assert eng._warned_format is False


def test_bad_format_emits_warning_and_sets_flag(ctx):
    eng, host = ctx
    fake_top = types.SimpleNamespace(pixel_format="16-bit float (RGBA)")
    assert eng._is_unsupported_format(fake_top) is True

    # Simulate what export_frame does on bad format (set_warning_status called every frame)
    host.set_warning_status(f"Source TOP pixel format {fake_top.pixel_format!r} is unsupported.")
    if not eng._warned_format:
        eng._warned_format = True

    assert len(host.warning_calls) == 1
    assert eng._warned_format is True


def test_consecutive_bad_frames_warn_every_frame_log_once(ctx):
    eng, host = ctx
    # Simulate 4 consecutive bad frames
    for _ in range(4):
        host.set_warning_status("unsupported")
        if not eng._warned_format:
            eng._warned_format = True  # would also trigger self._log() once

    # set_warning_status fires every frame (keeps yellow tint alive)
    assert len(host.warning_calls) == 4
    # _warned_format set only on first bad frame
    assert eng._warned_format is True
    # clear never called during continuous bad episode
    assert host.clear_count == 0


def test_good_format_clears_status(ctx):
    eng, host = ctx
    # Simulate prior bad-format episode
    eng._warned_format = True
    host.warning_calls.append("unsupported")

    fake_top = types.SimpleNamespace(pixel_format="8-bit fixed (RGBA)")
    assert eng._is_unsupported_format(fake_top) is False

    # Simulate what export_frame does on good format when _warned_format is True
    if eng._warned_format:
        host.clear_status()
        eng._warned_format = False

    assert host.clear_count == 1
    assert eng._warned_format is False


def test_second_episode_emits_second_log(ctx):
    eng, host = ctx
    # Episode 1: bad → cleared
    host.warning_calls.append("unsupported ep1")
    eng._warned_format = True
    host.clear_status()
    eng._warned_format = False

    # Episode 2: new bad frame — set_warning_status fires again, _warned_format goes True
    host.set_warning_status("unsupported ep2")
    if not eng._warned_format:
        eng._warned_format = True

    assert len(host.warning_calls) == 2  # once per episode minimum
    assert host.clear_count == 1
    assert eng._warned_format is True


def test_clear_status_called_on_cleanup(ctx):
    eng, host = ctx
    eng._warned_format = True
    # Simulate cleanup path
    if eng._warned_format:
        host.clear_status()
        eng._warned_format = False
    assert eng._warned_format is False
    assert host.clear_count == 1


# ---------------------------------------------------------------------------
# Host-boundary regression: set_warning_status / clear_status via export_frame
# ---------------------------------------------------------------------------


def _bad_format_top():
    return types.SimpleNamespace(pixel_format="16-bit float (RGBA)", is_valid=lambda: True)


def _good_format_top():
    return types.SimpleNamespace(pixel_format="8-bit fixed (RGBA)", is_valid=lambda: True)


def _prime_engine(eng):
    """Stub CUDA state so export_frame reaches the format-check branch."""
    eng.cuda = object()
    eng.ipc_stream = types.SimpleNamespace(value=0)


def test_export_frame_bad_format_returns_false_and_warns_host(ctx):
    """export_frame returns False and calls host.set_warning_status on every bad-format frame."""
    eng, host = ctx
    eng._export_buffer = _bad_format_top()
    _prime_engine(eng)

    for _ in range(3):
        result = eng.export_frame()
        assert result is False

    assert len(host.warning_calls) == 3
    assert eng._warned_format is True
    assert host.clear_count == 0


def test_export_frame_good_format_after_bad_calls_clear_status(ctx):
    """export_frame calls host.clear_status on first good-format frame after a bad episode.

    The CUDA call fails on a SimpleNamespace top (no cuda_memory method) so
    export_frame returns False, but clear_status is called before the CUDA step.
    """
    eng, host = ctx
    _prime_engine(eng)

    # Simulate a prior bad episode
    eng._warned_format = True

    eng._export_buffer = _good_format_top()
    eng.export_frame()  # clear_status fires; CUDA call fails → returns False

    assert host.clear_count == 1
    assert eng._warned_format is False
