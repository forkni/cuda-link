"""NVTX annotation shim for the td_exporter COMP namespace.

Mirror of src/cuda_link/_nvtx.py for use by TDSender and TDReceiver.
Identical semantics; different module name since td_exporter uses flat imports.

Enabled via environment variables (read once at import, zero-cost when off):
  CUDALINK_NVTX=1           — top-level phase ranges on the GPU timeline
  CUDALINK_NVTX_VERBOSE=1   — sub-operation ranges (implies CUDALINK_NVTX=1)

Requires the `nvtx` PyPI package when enabled:  pip install nvtx
"""

from __future__ import annotations

import os

_VERBOSE = os.environ.get("CUDALINK_NVTX_VERBOSE", "0") == "1"
_ENABLED = _VERBOSE or os.environ.get("CUDALINK_NVTX", "0") == "1"

if _ENABLED:
    try:
        import nvtx as _lib

        _AVAILABLE = True
    except ImportError:
        _lib = None
        _AVAILABLE = False
else:
    _lib = None
    _AVAILABLE = False


class _Noop:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


_NOOP = _Noop()


def annotate(message, color="white"):
    """Context manager for a named NVTX range. No-op if NVTX is disabled."""
    if _AVAILABLE:
        return _lib.annotate(message, color=color)
    return _NOOP


def verbose_range(message, color="white"):
    """Context manager for a sub-operation range. Only active when CUDALINK_NVTX_VERBOSE=1."""
    if _AVAILABLE and _VERBOSE:
        return _lib.annotate(message, color=color)
    return _NOOP


def push_range(message, color="white"):
    """Push a named NVTX range onto the thread-local stack."""
    if _AVAILABLE:
        _lib.push_range(message, color=color)


def pop_range():
    """Pop the innermost NVTX range from the thread-local stack."""
    if _AVAILABLE:
        _lib.pop_range()


def is_enabled():
    return _AVAILABLE


def is_verbose():
    return _AVAILABLE and _VERBOSE
