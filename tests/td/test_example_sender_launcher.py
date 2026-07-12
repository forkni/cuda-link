"""
Regression test: example_sender_launcher._find_python_exe() / onStart() must not fall back
to spawning a bare, unverified "python" when no interpreter can be resolved (#4 in the
td_exporter audit).

Background: the sibling receiver launcher (example_receiver_launcher.py) blindly returns the
literal string "python" as its final fallback -- if nothing is actually on PATH, Popen raises
deep inside the subprocess with an opaque error. The sender launcher was hardened instead: its
final fallback checks shutil.which("python") and returns None if it doesn't resolve, and
onStart() prints a clear, actionable error and returns without spawning anything (and without
touching the TD `project` global, which does not exist outside a TouchDesigner process).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))


def _reset_module():
    """(Re)import example_sender_launcher fresh so module-level globals (_process,
    _SENDER_PYTHON_EXE) don't bleed state between tests."""
    import example_sender_launcher as mod

    importlib.reload(mod)
    return mod


def test_find_python_exe_returns_none_when_nothing_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """_find_python_exe() must return None (not the bare string 'python') when no
    interpreter can be resolved via env var, 'py -3', or PATH."""
    mod = _reset_module()

    monkeypatch.delenv("CUDALINK_SENDER_PYTHON_EXE", raising=False)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    result = mod._find_python_exe()
    assert result is None, "must return None, not silently fall back to an unverified bare 'python'"


def test_onstart_does_not_spawn_when_no_interpreter_resolved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """onStart() must not call subprocess.Popen -- and must not touch the TD `project`
    global -- when _SENDER_PYTHON_EXE is None; it should print a clear error and return."""
    mod = _reset_module()

    monkeypatch.setattr(mod, "_SENDER_PYTHON_EXE", None)
    mock_popen = MagicMock()
    monkeypatch.setattr(mod.subprocess, "Popen", mock_popen)

    mod.onStart()  # must return before referencing `project.folder` (undefined outside TD)

    mock_popen.assert_not_called()
    assert mod._process is None

    captured = capsys.readouterr()
    assert "ERROR" in captured.out
    assert "CUDALINK_SENDER_PYTHON_EXE" in captured.out
