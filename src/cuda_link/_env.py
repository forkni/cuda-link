"""Typed helpers for CUDALINK_* environment variable reads.

Each helper reads os.environ at call time so monkeypatch.setenv works reliably
in tests without import-order constraints.
"""

from __future__ import annotations

import os


def env_bool(name: str, *, default: bool) -> bool:
    """Return the env var as bool.  "1" → True, "0"/"" → False, absent → default."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val == "1"


def env_int(name: str, *, default: int) -> int:
    """Return the env var as int.  Falls back to default when absent or non-numeric."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def env_str(name: str, *, default: str) -> str:
    """Return the env var as str.  Falls back to default when absent."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val
