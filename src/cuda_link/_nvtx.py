"""NVTX annotation shim for cuda-link profiling.

Enabled via environment variables (read at module import time, zero-cost when off):
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

from ._env import env_bool


class _Noop:
    __slots__ = ()

    def __enter__(self) -> _Noop:
        return self

    def __exit__(self, *_: object) -> None:
        pass


_NOOP = _Noop()


class _NvtxState:
    __slots__ = ("lib", "available", "verbose")

    def __init__(self, lib: object, available: bool, verbose: bool) -> None:
        self.lib = lib
        self.available = available
        self.verbose = verbose


def _detect_nvtx() -> _NvtxState:
    verbose = env_bool("CUDALINK_NVTX_VERBOSE", default=False)
    enabled = verbose or env_bool("CUDALINK_NVTX", default=False)
    if enabled:
        try:
            import nvtx as _lib  # type: ignore[import]

            return _NvtxState(lib=_lib, available=True, verbose=verbose)
        except ImportError:
            pass
    return _NvtxState(lib=None, available=False, verbose=verbose)


_NVTX_STATE = _detect_nvtx()


def annotate(message: str, color: str = "white") -> _Noop:
    """Context manager for a named NVTX range. No-op if NVTX is disabled."""
    if _NVTX_STATE.available:
        return _NVTX_STATE.lib.annotate(message, color=color)  # type: ignore[union-attr]
    return _NOOP


def verbose_range(message: str, color: str = "white") -> _Noop:
    """Context manager for a sub-operation range. Only active when CUDALINK_NVTX_VERBOSE=1."""
    if _NVTX_STATE.available and _NVTX_STATE.verbose:
        return _NVTX_STATE.lib.annotate(message, color=color)  # type: ignore[union-attr]
    return _NOOP


def push_range(message: str, color: str = "white") -> None:
    """Push a named NVTX range onto the thread-local stack."""
    if _NVTX_STATE.available:
        _NVTX_STATE.lib.push_range(message, color=color)  # type: ignore[union-attr]


def pop_range() -> None:
    """Pop the innermost NVTX range from the thread-local stack."""
    if _NVTX_STATE.available:
        _NVTX_STATE.lib.pop_range()  # type: ignore[union-attr]


def is_enabled() -> bool:
    return _NVTX_STATE.available


def is_verbose() -> bool:
    return _NVTX_STATE.available and _NVTX_STATE.verbose


def slot_names(prefix: str, n: int = 10) -> tuple[str, ...]:
    """Return a pre-computed tuple of ``n`` slot annotation labels for ``prefix``."""
    return tuple(f"{prefix}{i}" for i in range(n))
