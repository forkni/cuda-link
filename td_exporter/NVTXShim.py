"""NVTX annotation shim for cuda-link profiling.

Enabled via environment variables (read at first use, zero-cost when off):
  CUDALINK_NVTX=1           — top-level phase ranges on the GPU timeline
  CUDALINK_NVTX_VERBOSE=1   — sub-operation ranges (implies CUDALINK_NVTX=1)

Requires the `nvtx` PyPI package when enabled:  pip install nvtx

Usage:
  from cuda_link import _nvtx

  _nvtx.push_range("cudalink.exporter.export_frame", "green")
  try:
      ...gpu work...
  finally:
      _nvtx.pop_range()

  # or as a context manager for sub-ranges:
  with _nvtx.annotate("cudalink.exporter.memcpy", "green"):
      cuda.memcpy_async(...)
"""

from __future__ import annotations

from Env import env_bool


class _Noop:
    __slots__ = ()

    def __enter__(self) -> _Noop:
        return self

    def __exit__(self, *_: object) -> None:
        pass


_NOOP = _Noop()

_lib: object = None
_AVAILABLE: bool = False
_VERBOSE: bool = False
_initialized: bool = False


def _ensure_init() -> None:
    global _lib, _AVAILABLE, _VERBOSE, _initialized
    if _initialized:
        return
    _initialized = True
    _VERBOSE = env_bool("CUDALINK_NVTX_VERBOSE", default=False)
    _enabled = _VERBOSE or env_bool("CUDALINK_NVTX", default=False)
    if _enabled:
        try:
            import nvtx as _lib_import  # type: ignore[import]

            _lib = _lib_import
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False


def annotate(message: str, color: str = "white") -> _Noop:
    """Context manager for a named NVTX range. No-op if NVTX is disabled."""
    _ensure_init()
    if _AVAILABLE:
        return _lib.annotate(message, color=color)  # type: ignore[union-attr]
    return _NOOP


def verbose_range(message: str, color: str = "white") -> _Noop:
    """Context manager for a sub-operation range. Only active when CUDALINK_NVTX_VERBOSE=1."""
    _ensure_init()
    if _AVAILABLE and _VERBOSE:
        return _lib.annotate(message, color=color)  # type: ignore[union-attr]
    return _NOOP


def push_range(message: str, color: str = "white") -> None:
    """Push a named NVTX range onto the thread-local stack."""
    _ensure_init()
    if _AVAILABLE:
        _lib.push_range(message, color=color)  # type: ignore[union-attr]


def pop_range() -> None:
    """Pop the innermost NVTX range from the thread-local stack."""
    _ensure_init()
    if _AVAILABLE:
        _lib.pop_range()  # type: ignore[union-attr]


def is_enabled() -> bool:
    _ensure_init()
    return _AVAILABLE


def is_verbose() -> bool:
    _ensure_init()
    return _AVAILABLE and _VERBOSE
