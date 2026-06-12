"""
Regression test: TDSenderEngine._resolve_export_sync tri-state collapse.

CUDA 719 root cause (confirmed 2026-06-10):
  In 1.10.0, the TD Sender defaulted to async export (None → async).  The TD source
  buffer (cm.ptr) is TD's transient cook-scoped TOP texture — reclaimed the instant the
  cook returns.  Under a loaded consumer the queued async D2D read executes after TD
  recycled the source → reads freed memory → sticky CUDA 719.

Fix (1.10.1): TD Sender now blocks by default (None → sync).  Explicit
  CUDALINK_EXPORT_SYNC=0 opts back into async for callers with a stable source buffer.

This test locks the policy-collapse helper (_resolve_export_sync) — the only correct
GPU-free seam.  The actual cross-process race cannot be reproduced without a real GPU,
IPC ring buffer, and a loaded consumer; that finding is documented in the plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(value):
    from TDSender import TDSenderEngine

    return TDSenderEngine._resolve_export_sync(value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveExportSync:
    """_resolve_export_sync tri-state collapse: None → True, True → True, False → False."""

    def test_none_resolves_to_true(self):
        """Unset env (None) must default to blocking to protect the transient TD source."""
        assert _resolve(None) is True

    def test_true_resolves_to_true(self):
        """Explicit CUDALINK_EXPORT_SYNC=1 keeps blocking."""
        assert _resolve(True) is True

    def test_false_resolves_to_false(self):
        """Explicit CUDALINK_EXPORT_SYNC=0 opts into async (stable-source callers)."""
        assert _resolve(False) is False

    def test_return_type_is_strict_bool(self):
        """All three branches return exactly bool, not a truthy/falsy value."""
        for v in (None, True, False):
            result = _resolve(v)
            assert type(result) is bool, f"Expected bool, got {type(result)} for input {v!r}"

    def test_is_static_method(self):
        """_resolve_export_sync must be callable without an engine instance."""
        from TDSender import TDSenderEngine

        # Calling via the class (not an instance) confirms it is @staticmethod
        result = TDSenderEngine._resolve_export_sync(None)
        assert result is True

    def test_none_is_blocking_not_async_old_behaviour(self):
        """Explicit regression guard: None must NOT resolve to False (the 1.10.0 bug)."""
        assert _resolve(None) is not False, (
            "None resolved to False — this is the 1.10.0 regression that caused CUDA 719. "
            "TDSender.py must collapse None → True (blocking default)."
        )
