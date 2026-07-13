"""Coverage-focused tests for cuda_link._doorbell.

test_doorbell.py covers doorbell_event_name(), the Importer-level
wait_for_doorbell() primitive, and the real Win32 happy-path round trip. This
file fills the remaining gaps: the non-Windows (_k32 is None) branches of
every function (via a reload with os.name patched to "posix"), and the
Win32 failure-injection branches of CreateEventW/OpenEventW/SetEvent that the
happy-path round trip never exercises (mocked kernel32 calls, still real
ctypes plumbing on Windows).
"""

from __future__ import annotations

import importlib
import logging
import os
from unittest.mock import MagicMock

import pytest

import cuda_link._doorbell as db

# ---------------------------------------------------------------------------
# Non-Windows (_k32 is None) branches — every public function's early-return.
#
# _doorbell.py decides `_k32` once at import time based on os.name. We patch
# os.name to a non-"nt" value and reload the module to exercise the `else:
# _k32 = None` branch and every function's `if ... _k32 is None: ...` guard,
# then restore the real os.name and reload again so later tests (including
# this file's own Windows-path tests) see the real Win32 _k32.
# ---------------------------------------------------------------------------


def test_non_windows_branches_are_all_inert():
    original_name = os.name
    try:
        os.name = "posix"
        importlib.reload(db)
        assert db._k32 is None

        assert db.create_doorbell("x") is None
        assert db.open_doorbell("x") is None
        assert db.signal(None) is None  # no-op, must not raise
        assert db.wait(None, 10) is False
        assert db.close(None) is None  # no-op, must not raise
    finally:
        os.name = original_name
        importlib.reload(db)


# ---------------------------------------------------------------------------
# Win32 failure-injection branches (Windows-only — mocks kernel32 calls).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_create_doorbell_returns_none_on_createeventw_failure(monkeypatch):
    monkeypatch.setattr(db._k32, "CreateEventW", lambda *a: 0)
    monkeypatch.setattr(db.ctypes, "get_last_error", lambda: 5)  # ERROR_ACCESS_DENIED
    assert db.create_doorbell("test_create_failure_name") is None


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_open_doorbell_returns_none_when_not_found(monkeypatch, caplog: pytest.LogCaptureFixture):
    """err == 2 (ERROR_FILE_NOT_FOUND) — producer not ready yet, logged at debug (not warning)."""
    monkeypatch.setattr(db._k32, "OpenEventW", lambda *a: 0)
    monkeypatch.setattr(db.ctypes, "get_last_error", lambda: 2)
    with caplog.at_level(logging.DEBUG, logger="cuda_link._doorbell"):
        assert db.open_doorbell("nonexistent_event_name") is None
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not found" in r.message for r in debug_records)
    assert warning_records == []


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_open_doorbell_returns_none_on_other_failure(monkeypatch, caplog: pytest.LogCaptureFixture):
    """err != 2 — some other OpenEventW failure, logged at warning (not debug)."""
    monkeypatch.setattr(db._k32, "OpenEventW", lambda *a: 0)
    monkeypatch.setattr(db.ctypes, "get_last_error", lambda: 5)  # ERROR_ACCESS_DENIED
    with caplog.at_level(logging.DEBUG, logger="cuda_link._doorbell"):
        assert db.open_doorbell("some_event_name") is None
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("OpenEventW" in r.message for r in warning_records)
    assert debug_records == []


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_signal_is_noop_when_handle_none(monkeypatch):
    """handle=None must short-circuit before ever touching _k32.SetEvent."""
    mock_set_event = MagicMock()
    monkeypatch.setattr(db._k32, "SetEvent", mock_set_event)
    db.signal(None)  # must not raise, must not touch _k32.SetEvent
    mock_set_event.assert_not_called()


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_signal_logs_warning_on_setevent_failure(monkeypatch):
    monkeypatch.setattr(db._k32, "SetEvent", lambda *a: 0)
    monkeypatch.setattr(db.ctypes, "get_last_error", lambda: 6)  # ERROR_INVALID_HANDLE
    db.signal(object())  # must not raise even though SetEvent "fails"


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_wait_returns_false_when_handle_none():
    assert db.wait(None, 100) is False
