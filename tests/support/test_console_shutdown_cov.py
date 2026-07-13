"""Coverage-focused tests for cuda_link._console.

test_console_shutdown.py exercises CTRL_C / CTRL_BREAK / CTRL_CLOSE (both
defer_close variants) plus the non-win32 inert path and run_with_watchdog.
This file fills the remaining two gaps: the _ctrl_handler fallback `return
False` for a ctrl_type matching none of the known events, and the warning
printed when the final SetConsoleCtrlHandler registration call itself fails.
Follows the same win32-only ctypes-mocking pattern as the existing file.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from cuda_link._console import install_console_ctrl_handler


@pytest.mark.skipif(sys.platform != "win32", reason="win32 only")
def test_ctrl_handler_unrecognized_event_returns_false() -> None:
    """A ctrl_type matching none of C/BREAK/CLOSE/LOGOFF/SHUTDOWN falls through to False."""
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

    assert captured_routine, "handler should have been registered"
    result = captured_routine[-1](99)  # unrecognized ctrl_type
    assert not result
    assert state.shutdown_via is None
    assert state.stop_requested is False


@pytest.mark.skipif(sys.platform != "win32", reason="win32 only")
def test_install_prints_warning_when_registration_fails(capsys: pytest.CaptureFixture) -> None:
    """A falsy return from the final SetConsoleCtrlHandler(handler_ref, True) call prints a warning."""
    import ctypes

    mock_k32 = MagicMock()
    mock_k32.SetConsoleCtrlHandler.return_value = False  # both calls "fail"

    with patch("cuda_link._console.ctypes") as mock_ctypes:
        mock_ctypes.windll.kernel32 = mock_k32
        mock_ctypes.WINFUNCTYPE = ctypes.WINFUNCTYPE
        mock_ctypes.c_void_p = ctypes.c_void_p
        mock_ctypes.wintypes = None

        install_console_ctrl_handler("[t]", lambda: None)

    out = capsys.readouterr().out
    assert "SetConsoleCtrlHandler failed" in out
