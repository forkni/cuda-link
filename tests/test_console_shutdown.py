"""
Tests for cuda_link._console — ConsoleShutdownState, install_console_ctrl_handler,
and run_with_watchdog.

All tests are pure unit tests — no GPU, no real Windows console required.
On non-win32 platforms the install function is a no-op, and those paths are
tested via the inert-state assertions.  Win32-specific routing tests capture
the installed WINFUNCTYPE routine and invoke it directly in-process.
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from cuda_link._console import (
    _CTRL_BREAK_EVENT,
    _CTRL_C_EVENT,
    _CTRL_CLOSE_EVENT,
    ConsoleShutdownState,
    install_console_ctrl_handler,
    run_with_watchdog,
)

# ---------------------------------------------------------------------------
# ConsoleShutdownState defaults
# ---------------------------------------------------------------------------


def test_state_defaults() -> None:
    """ConsoleShutdownState starts with all fields at safe defaults."""
    state = ConsoleShutdownState()
    assert state.shutdown_via is None
    assert state.stop_requested is False
    assert state._handler_ref is None


# ---------------------------------------------------------------------------
# install_console_ctrl_handler — non-win32 inert path
# ---------------------------------------------------------------------------


def test_install_non_win32_returns_inert_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-win32 platforms install_console_ctrl_handler returns an inert state."""
    monkeypatch.setattr(sys, "platform", "linux")
    state = install_console_ctrl_handler("[test]", lambda: None)
    assert isinstance(state, ConsoleShutdownState)
    assert state.shutdown_via is None
    assert state.stop_requested is False
    assert state._handler_ref is None


def test_install_non_win32_defer_close_also_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    """defer_close=True is a no-op on non-win32 too."""
    monkeypatch.setattr(sys, "platform", "linux")
    state = install_console_ctrl_handler("[test]", lambda: None, defer_close=True)
    assert state.stop_requested is False


# ---------------------------------------------------------------------------
# run_with_watchdog
# ---------------------------------------------------------------------------


def test_watchdog_completes_within_timeout() -> None:
    """run_with_watchdog returns normally when fn finishes quickly."""
    calls: list[str] = []

    def _fn() -> None:
        calls.append("done")

    run_with_watchdog(_fn, timeout_s=2.0, label="test fn", prefix="[t]")
    assert calls == ["done"]


def test_watchdog_prints_timeout_message(capsys: pytest.CaptureFixture) -> None:
    """run_with_watchdog prints a timeout message when fn exceeds the timeout."""

    def _slow() -> None:
        time.sleep(5.0)  # far beyond the watchdog

    run_with_watchdog(_slow, timeout_s=0.05, label="slow fn", prefix="[t]")
    out = capsys.readouterr().out
    assert "slow fn" in out
    assert "timed out" in out


def test_watchdog_fn_exceptions_do_not_propagate_to_caller() -> None:
    """Exceptions inside fn stay in the daemon thread; run_with_watchdog does not re-raise."""
    error_caught: list[str] = []

    def _raises_but_catches() -> None:
        # Callers are responsible for swallowing exceptions (documented contract).
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            error_caught.append(str(exc))

    # Should return normally; fn absorbed its own exception
    run_with_watchdog(_raises_but_catches, timeout_s=2.0, label="raising fn", prefix="[t]")
    assert error_caught == ["boom"]


# ---------------------------------------------------------------------------
# Windows ctrl-handler routing — tested on win32 only
# ---------------------------------------------------------------------------

# These tests capture the installed WINFUNCTYPE routine and invoke it directly
# in-process.  They skip on non-win32 because WINFUNCTYPE is only available there.

pytestmark_win32 = pytest.mark.skipif(sys.platform != "win32", reason="win32 only")


@pytest.mark.skipif(sys.platform != "win32", reason="win32 only")
def test_ctrl_c_sets_shutdown_via_and_returns_false() -> None:
    """CTRL_C_EVENT sets shutdown_via='ctrl_c' and returns False (chains to Python default)."""
    import ctypes

    mock_k32 = MagicMock()
    mock_k32.SetConsoleCtrlHandler.return_value = True
    captured_routine: list = []

    def _capture_install(handler_arg, enable):
        if handler_arg is not None:
            captured_routine.append(handler_arg)
        return True

    mock_k32.SetConsoleCtrlHandler.side_effect = _capture_install

    with patch("cuda_link._console.ctypes") as mock_ctypes:
        mock_ctypes.windll.kernel32 = mock_k32
        # Re-use the real WINFUNCTYPE so the captured function is callable
        mock_ctypes.WINFUNCTYPE = ctypes.WINFUNCTYPE
        mock_ctypes.c_void_p = ctypes.c_void_p
        mock_ctypes.wintypes = None  # not needed for routing test

        state = install_console_ctrl_handler("[t]", lambda: None)

    assert captured_routine, "handler should have been registered"
    # The captured routine is a WINFUNCTYPE wrapper — call it directly
    result = captured_routine[-1](_CTRL_C_EVENT)
    assert state.shutdown_via == "ctrl_c"
    assert not result  # chains to Python default (WINFUNCTYPE returns 0, not False)


@pytest.mark.skipif(sys.platform != "win32", reason="win32 only")
def test_ctrl_break_sets_shutdown_via_and_returns_false() -> None:
    """CTRL_BREAK_EVENT sets shutdown_via='ctrl_break' and returns False."""
    import ctypes

    mock_k32 = MagicMock()
    captured_routine: list = []

    def _capture(handler_arg, enable):
        if handler_arg is not None:
            captured_routine.append(handler_arg)
        return True

    mock_k32.SetConsoleCtrlHandler.side_effect = _capture

    with patch("cuda_link._console.ctypes") as mock_ctypes:
        mock_ctypes.windll.kernel32 = mock_k32
        mock_ctypes.WINFUNCTYPE = ctypes.WINFUNCTYPE
        mock_ctypes.c_void_p = ctypes.c_void_p
        mock_ctypes.wintypes = None

        state = install_console_ctrl_handler("[t]", lambda: None)

    result = captured_routine[-1](_CTRL_BREAK_EVENT)
    assert state.shutdown_via == "ctrl_break"
    assert not result  # chains to Python default (WINFUNCTYPE returns 0, not False)


@pytest.mark.skipif(sys.platform != "win32", reason="win32 only")
def test_ctrl_close_inline_calls_cleanup_returns_true() -> None:
    """CTRL_CLOSE_EVENT with defer_close=False calls on_cleanup and returns True."""
    import ctypes

    cleanup_calls: list[str] = []

    mock_k32 = MagicMock()
    captured_routine: list = []

    def _capture(handler_arg, enable):
        if handler_arg is not None:
            captured_routine.append(handler_arg)
        return True

    mock_k32.SetConsoleCtrlHandler.side_effect = _capture

    with patch("cuda_link._console.ctypes") as mock_ctypes:
        mock_ctypes.windll.kernel32 = mock_k32
        mock_ctypes.WINFUNCTYPE = ctypes.WINFUNCTYPE
        mock_ctypes.c_void_p = ctypes.c_void_p
        mock_ctypes.wintypes = None

        state = install_console_ctrl_handler("[t]", lambda: cleanup_calls.append("cleaned"), defer_close=False)

    result = captured_routine[-1](_CTRL_CLOSE_EVENT)
    assert result  # handled — WINFUNCTYPE returns 1, truthy
    assert state.shutdown_via == "ctrl_close"
    assert state.stop_requested is False
    assert cleanup_calls == ["cleaned"]


@pytest.mark.skipif(sys.platform != "win32", reason="win32 only")
def test_ctrl_close_deferred_sets_stop_requested_not_cleanup() -> None:
    """CTRL_CLOSE_EVENT with defer_close=True sets stop_requested without calling cleanup."""
    import ctypes

    cleanup_calls: list[str] = []

    mock_k32 = MagicMock()
    captured_routine: list = []

    def _capture(handler_arg, enable):
        if handler_arg is not None:
            captured_routine.append(handler_arg)
        return True

    mock_k32.SetConsoleCtrlHandler.side_effect = _capture

    with patch("cuda_link._console.ctypes") as mock_ctypes:
        mock_ctypes.windll.kernel32 = mock_k32
        mock_ctypes.WINFUNCTYPE = ctypes.WINFUNCTYPE
        mock_ctypes.c_void_p = ctypes.c_void_p
        mock_ctypes.wintypes = None

        state = install_console_ctrl_handler("[t]", lambda: cleanup_calls.append("cleaned"), defer_close=True)

    result = captured_routine[-1](_CTRL_CLOSE_EVENT)
    assert result  # handled — WINFUNCTYPE returns 1, truthy
    assert state.shutdown_via == "ctrl_close"
    assert state.stop_requested is True
    assert cleanup_calls == [], "cleanup must NOT be called when defer_close=True"
