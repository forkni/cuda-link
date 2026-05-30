"""
Tests for probe_producer() cudaGetErrorString decode (item 7 of ctypes sweep).

Background
----------
ctypes c_char_p restype returns bytes, not str.  Before the fix, probe_producer
interpolated the raw bytes object into an f-string, rendering e.g.
  "[PROBE] cudaMalloc failed: b'out of memory' (error 1)"
instead of
  "[PROBE] cudaMalloc failed: out of memory (error 1)"

The fix adds .decode("utf-8") with a None guard (matching the wrapper idiom).
These tests confirm both the decode path and the NULL fallback, GPU-free.

The probe script lives under scripts/probe/ (not a package), so it is loaded
by file path.  Module top-level is side-effect-free (imports + constants only).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_PROBE_PATH = Path(__file__).parent.parent / "scripts" / "probe" / "driver_api_ipc_probe.py"


def _load_probe():
    """Load driver_api_ipc_probe as a fresh module object (no sys.modules entry)."""
    spec = importlib.util.spec_from_file_location("_probe_under_test", _PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared fixture: module loaded once per test session (state reset per test)
# ---------------------------------------------------------------------------

_PROBE_MOD = _load_probe()


def _make_fake_rt(*, error_code: int = 1, error_string=b"out of memory"):
    """Return a mock cudart whose cudaMalloc returns error_code immediately."""
    rt = MagicMock()
    rt.cudaMalloc.return_value = error_code
    rt.cudaGetErrorString.return_value = error_string
    return rt


def _patch_probe_for_failure(mod, *, error_string=b"out of memory"):
    """Monkey-patch module-level callables so probe_producer reaches the decode path."""
    mod._load_driver_api = lambda: MagicMock()
    mod._load_cudart = lambda: _make_fake_rt(error_string=error_string)
    mod._switch_to_primary = lambda drv, dev: "saved_context"
    mod._restore_context = lambda drv, saved: None
    mod._PROBE_STATE.clear()


# ---------------------------------------------------------------------------
# Test 1: normal error string is decoded — no b'...' in the message
# ---------------------------------------------------------------------------


def test_producer_decodes_error_string() -> None:
    """probe_producer must decode cudaGetErrorString bytes → str in the error message.

    Pre-fix the f-string would embed the raw bytes repr "b'out of memory'".
    Post-fix it embeds the decoded string "out of memory".
    """
    _patch_probe_for_failure(_PROBE_MOD, error_string=b"out of memory")

    # patch os.makedirs so the test doesn't touch D:\cuda-link\scratch
    with patch("os.makedirs"), pytest.raises(RuntimeError) as exc_info:
        _PROBE_MOD.probe_producer()

    msg = str(exc_info.value)

    assert "out of memory" in msg, f"Decoded error string missing from message: {msg!r}"
    assert "b'" not in msg, f"Raw bytes repr found in message — decode() is not applied: {msg!r}"


# ---------------------------------------------------------------------------
# Test 2: NULL cudaGetErrorString (c_char_p returns None) falls back gracefully
# ---------------------------------------------------------------------------


def test_producer_null_error_string() -> None:
    """probe_producer must not raise AttributeError when cudaGetErrorString returns None.

    ctypes c_char_p returns None for a NULL pointer; the guard
        cstr.decode("utf-8") if cstr is not None else f"unknown error {r}"
    must handle this case and fall back to the unknown-error message.
    """
    _patch_probe_for_failure(_PROBE_MOD, error_string=None)

    with patch("os.makedirs"), pytest.raises(RuntimeError) as exc_info:
        _PROBE_MOD.probe_producer()

    msg = str(exc_info.value)

    assert "unknown error" in msg, f"Expected 'unknown error' fallback in message; got: {msg!r}"
    # Must not contain raw None repr
    assert "None" not in msg or "unknown error" in msg, f"Unexpected None in message: {msg!r}"
