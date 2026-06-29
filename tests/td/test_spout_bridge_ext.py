"""
Characterization tests for SpoutBridgeExt — lock observable behaviour before changes.

All tests use FakeTDHost (from conftest / td_exporter/_td_fakes.py) and mock
subprocess.Popen so no real process is spawned.  GPU / Spout / TD not required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from conftest import FakeTDHost
from SpoutBridgeExt import SpoutBridgeExt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_host(extra: dict | None = None) -> FakeTDHost:
    """Return a FakeTDHost with sensible SpoutBridge defaults."""
    params: dict = {
        "Active": False,
        "Direction": "out",
        "Spoutname": "test_spout",
        "Ipcname": "test_ipc",
        "Pythonexe": "python",  # skip py-launcher probe in _resolve_python
        "Debug": False,
        "Status": "",
    }
    if extra:
        params.update(extra)
    return FakeTDHost(params=params)


def _fake_popen(pid: int = 1234) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = None  # appears alive
    return proc


# ---------------------------------------------------------------------------
# _build_argv — verbose flag propagation
# ---------------------------------------------------------------------------


def test_build_argv_no_verbose_by_default():
    """--verbose must NOT appear in argv when Debug param is False."""
    host = _make_host({"Debug": False})
    ext = SpoutBridgeExt(host)
    argv = ext._build_argv()
    assert "--verbose" not in argv


def test_build_argv_verbose_when_debug_true():
    """--verbose IS appended to argv when Debug param is True."""
    host = _make_host({"Debug": True})
    ext = SpoutBridgeExt(host)
    argv = ext._build_argv()
    assert "--verbose" in argv


def test_build_argv_contains_required_flags():
    """Core flags (--dir, --ipc, --spout) are always present."""
    host = _make_host()
    ext = SpoutBridgeExt(host)
    argv = ext._build_argv()
    assert "--dir" in argv
    assert "--ipc" in argv
    assert "--spout" in argv
    assert "cuda_link_spout.bridge" in argv


def test_build_argv_debug_reads_host_param_not_self_debug():
    """_build_argv reads the host Debug param, not self._debug.

    Simulates the case where set_debug() was never called (project load / Re-Init):
    self._debug is False (default) but the Debug *param* is True.
    """
    host = _make_host({"Debug": True})
    ext = SpoutBridgeExt(host)
    # Deliberately leave self._debug as the default False (no set_debug() called)
    assert ext._debug is False
    argv = ext._build_argv()
    # Must still emit --verbose because it reads from the host param
    assert "--verbose" in argv


# ---------------------------------------------------------------------------
# stop() — no-op when not started
# ---------------------------------------------------------------------------


def test_stop_noop_when_not_started():
    """stop() must not raise when no sidecar is running."""
    host = _make_host()
    ext = SpoutBridgeExt(host)
    ext.stop()  # must not raise


def test_stop_is_noop_when_not_started():
    """stop() does not write Status when no process was ever started."""
    host = _make_host()
    ext = SpoutBridgeExt(host)
    ext.stop()
    # Status remains at its initial value (empty string / whatever was set in the host)
    assert host.param_value("Status") == ""


# ---------------------------------------------------------------------------
# start() — spawns process + syncs _debug from Debug param
# ---------------------------------------------------------------------------


@patch("subprocess.Popen")
def test_start_spawns_process(mock_popen):
    """start() calls Popen with the correct interpreter."""
    mock_popen.return_value = _fake_popen()
    host = _make_host()
    ext = SpoutBridgeExt(host)
    ext.start()
    mock_popen.assert_called_once()
    argv_used = mock_popen.call_args[0][0]
    assert argv_used[0] == "python"


@patch("subprocess.Popen")
def test_start_initializes_debug_from_debug_param(mock_popen):
    """start() syncs self._debug from the Debug param before spawning.

    This is the Bug-1 fix: without this, _debug stays False after project load /
    Re-Init if no set_debug() change-event was fired since extension construction.
    """
    mock_popen.return_value = _fake_popen()
    host = _make_host({"Debug": True})
    ext = SpoutBridgeExt(host)
    # Simulate no set_debug() call — _debug is still the constructor default
    assert ext._debug is False
    ext.start()
    # After start(), self._debug must reflect the Debug param
    assert ext._debug is True


@patch("subprocess.Popen")
def test_start_verbose_in_argv_when_debug_true(mock_popen):
    """The spawned argv contains --verbose when Debug=True."""
    mock_popen.return_value = _fake_popen()
    host = _make_host({"Debug": True})
    ext = SpoutBridgeExt(host)
    ext.start()
    argv_used = mock_popen.call_args[0][0]
    assert "--verbose" in argv_used


@patch("subprocess.Popen")
def test_start_no_verbose_in_argv_when_debug_false(mock_popen):
    """The spawned argv does NOT contain --verbose when Debug=False."""
    mock_popen.return_value = _fake_popen()
    host = _make_host({"Debug": False})
    ext = SpoutBridgeExt(host)
    ext.start()
    argv_used = mock_popen.call_args[0][0]
    assert "--verbose" not in argv_used


@patch("subprocess.Popen")
def test_start_noop_when_already_running(mock_popen):
    """start() is a no-op (no second Popen call) if a process is already running."""
    mock_popen.return_value = _fake_popen()
    host = _make_host()
    ext = SpoutBridgeExt(host)
    ext.start()
    ext.start()  # second call — process still alive (poll() → None)
    assert mock_popen.call_count == 1


# ---------------------------------------------------------------------------
# restart() — stop-then-start
# ---------------------------------------------------------------------------


@patch("subprocess.Popen")
def test_restart_from_stopped_spawns_once(mock_popen):
    """restart() when nothing is running: stop() is a no-op, start() spawns once."""
    mock_popen.return_value = _fake_popen()
    host = _make_host()
    ext = SpoutBridgeExt(host)
    ext.restart()
    assert mock_popen.call_count == 1


@patch("subprocess.Popen")
def test_restart_from_running_stops_then_starts(mock_popen):
    """restart() kills the running sidecar and spawns a new one."""
    first_proc = _fake_popen(pid=111)
    second_proc = _fake_popen(pid=222)
    mock_popen.side_effect = [first_proc, second_proc]

    host = _make_host()
    ext = SpoutBridgeExt(host)
    ext.start()  # first spawn
    assert mock_popen.call_count == 1

    # Make the first process appear alive so stop() sends CTRL_BREAK
    first_proc.poll.return_value = None
    first_proc.wait.return_value = 0  # graceful exit

    ext.restart()  # should kill first, spawn second
    assert mock_popen.call_count == 2
    # On Windows stop() sends CTRL_BREAK_EVENT; on POSIX it calls terminate().
    if sys.platform == "win32":
        assert first_proc.send_signal.called
    else:
        assert first_proc.terminate.called
