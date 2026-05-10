"""
TDHost adapter — isolates all TouchDesigner runtime access behind a Protocol seam.

Every call that touches ownerComp, a TOP, or a Script TOP goes through this module.
Engine code imports nothing from the TD runtime; it calls TDHost / TOPHandle methods only.

textDAT name: TDHost  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# CUDAMemoryRef — TD-agnostic result of top.cudaMemory()
# ---------------------------------------------------------------------------


@dataclass
class CUDAMemoryRef:
    """Wraps the raw CUDAMemory object returned by TOP.cudaMemory().

    All fields are plain Python types — no TD types leak out.
    """

    ptr: int  # GPU pointer as plain int
    width: int
    height: int
    channels: int  # shape.numComps
    size: int
    data_type: Any = field(default=None)  # shape.dataType (TD-specific; forwarded opaquely)


# ---------------------------------------------------------------------------
# TOPHandle protocol
# ---------------------------------------------------------------------------


class TOPHandle:
    """Protocol-compatible base for wrapping a single TouchDesigner TOP operator.

    All concrete methods raise NotImplementedError; subclass RealTOPHandle provides
    the TD-connected implementation and FakeTOPHandle provides the test double.
    """

    def cuda_memory(self, stream: Any = None) -> CUDAMemoryRef:
        """Call top.cudaMemory(stream=stream) and return a CUDAMemoryRef."""
        raise NotImplementedError

    @property
    def pixel_format(self) -> str:
        """top.pixelFormat as a string."""
        raise NotImplementedError

    @property
    def inputs(self) -> list[TOPHandle]:
        """Wrapped TOPHandle for each upstream input operator."""
        raise NotImplementedError

    def set_format(self, fmt: str) -> None:
        """Write top.par.format = fmt."""
        raise NotImplementedError

    def copy_cuda_memory(self, ptr: int, size: int, shape: Any, *, stream: int) -> None:
        """Call script_top.copyCUDAMemory(ptr, size, shape, stream=stream)."""
        raise NotImplementedError

    def copy_numpy_array(self, arr: Any) -> None:
        """Call script_top.copyNumpyArray(arr)."""
        raise NotImplementedError

    def set_resolution(self, width: int, height: int) -> None:
        """Set Script TOP to custom resolution: outputresolution=9, resolutionw, resolutionh."""
        raise NotImplementedError

    def is_valid(self) -> bool:
        """Return True if the underlying TD operator is still present in the network."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TDHost protocol
# ---------------------------------------------------------------------------


class TDHost:
    """Protocol-compatible base for wrapping ownerComp.

    All parameter reads/writes and operator lookups go through this class.
    Subclass RealTDHost is the TD-connected implementation;
    FakeTDHost (in tests) is the in-process test double.
    """

    def param_value(self, name: str) -> Any:
        """Read ownerComp.par.<name>.eval()."""
        raise NotImplementedError

    def set_param_value(self, name: str, value: Any) -> None:
        """Write ownerComp.par.<name> = value."""
        raise NotImplementedError

    def set_param_enabled(self, name: str, enabled: bool) -> None:
        """Write ownerComp.par.<name>.enable = enabled."""
        raise NotImplementedError

    def show_custom_only(self, value: bool) -> None:
        """Write ownerComp.showCustomOnly = value."""
        raise NotImplementedError

    def is_active(self) -> bool:
        """Read ownerComp.par.Active.eval() via cached reference (hot-path safe)."""
        raise NotImplementedError

    def find_top(self, name: str) -> TOPHandle | None:
        """Return ownerComp.op(name) wrapped as a TOPHandle, or None."""
        raise NotImplementedError

    def set_warning_status(self, msg: str) -> None:
        """Tint ownerComp yellow to signal a recoverable warning (e.g. bad pixel format)."""
        raise NotImplementedError

    def set_error_status(self, msg: str) -> None:
        """Tint ownerComp red and emit a persistent script-error badge for fatal failures."""
        raise NotImplementedError

    def clear_status(self) -> None:
        """Restore ownerComp to its original color and clear any script-error badges."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Production adapters
# ---------------------------------------------------------------------------


class RealTOPHandle(TOPHandle):
    """Wraps a real TD TOP operator."""

    def __init__(self, top: Any) -> None:
        self._top = top

    def cuda_memory(self, stream: Any = None) -> CUDAMemoryRef:
        cm = self._top.cudaMemory(stream=stream) if stream is not None else self._top.cudaMemory()
        shape = cm.shape
        return CUDAMemoryRef(
            ptr=int(cm.ptr),
            width=int(shape.width),
            height=int(shape.height),
            channels=int(shape.numComps),
            size=int(cm.size),
            data_type=getattr(shape, "dataType", None),
        )

    @property
    def pixel_format(self) -> str:
        return str(getattr(self._top, "pixelFormat", ""))

    @property
    def inputs(self) -> list[TOPHandle]:
        try:
            return [RealTOPHandle(t) for t in self._top.inputs]
        except (AttributeError, TypeError):
            return []

    def set_format(self, fmt: str) -> None:
        with contextlib.suppress(AttributeError):
            self._top.par.format = fmt

    def copy_cuda_memory(self, ptr: int, size: int, shape: Any, *, stream: int) -> None:
        self._top.copyCUDAMemory(ptr, size, shape, stream=stream)

    def copy_numpy_array(self, arr: Any) -> None:
        self._top.copyNumpyArray(arr)

    def set_resolution(self, width: int, height: int) -> None:
        with contextlib.suppress(AttributeError):
            self._top.par.outputresolution = 9  # Custom Resolution mode
            self._top.par.resolutionw = width
            self._top.par.resolutionh = height

    def is_valid(self) -> bool:
        try:
            return bool(getattr(self._top, "valid", True))
        except (AttributeError, RuntimeError):
            return False


_WARNING_COLOR: tuple[float, float, float] = (0.9137, 1.0, 0.0)
_ERROR_COLOR: tuple[float, float, float] = (0.7, 0.0, 0.0)
_DEFAULT_NODE_COLOR: tuple[float, float, float] = (0.55, 0.55, 0.55)
_MANAGED_COLORS = (_WARNING_COLOR, _ERROR_COLOR)


class RealTDHost(TDHost):
    """Wraps a real TD ownerComp.

    Caches the Active parameter reference so is_active() avoids a 3-deep
    attribute chain on every frame.
    """

    def __init__(self, owner_comp: Any) -> None:
        self._comp = owner_comp
        try:
            self._active_par = owner_comp.par.Active
        except AttributeError:
            self._active_par = None
        # Deliberately NOT cached here: the COMP may be tinted from a prior session
        # (.tox saved while yellow/red) which would poison the cache.  Captured lazily
        # on the first set_warning_status / set_error_status call instead.
        self._default_color: tuple[float, float, float] | None = None

    def param_value(self, name: str) -> Any:
        try:
            return getattr(self._comp.par, name).eval()
        except AttributeError:
            return None

    def set_param_value(self, name: str, value: Any) -> None:
        with contextlib.suppress(AttributeError):
            setattr(self._comp.par, name, value)

    def set_param_enabled(self, name: str, enabled: bool) -> None:
        with contextlib.suppress(AttributeError):
            getattr(self._comp.par, name).enable = enabled

    def show_custom_only(self, value: bool) -> None:
        with contextlib.suppress(AttributeError):
            self._comp.showCustomOnly = value

    def is_active(self) -> bool:
        if self._active_par is None:
            return True  # no Active par → always active (backward compat)
        try:
            return bool(self._active_par.eval())
        except AttributeError:
            return True

    def find_top(self, name: str) -> RealTOPHandle | None:
        try:
            top = self._comp.op(name)
            return RealTOPHandle(top) if top is not None else None
        except (AttributeError, RuntimeError):
            return None

    def _capture_default_color(self) -> None:
        if self._default_color is not None:
            return
        with contextlib.suppress(AttributeError, RuntimeError):
            c = self._comp.color
            current = (float(c[0]), float(c[1]), float(c[2]))
            if current not in _MANAGED_COLORS:
                self._default_color = current
                return
        # Fallback: current color is managed (stale tint from prior session) or
        # unreadable — use TD's default node grey so clear_status always restores
        # to a neutral colour rather than staying stuck at warning/error tint.
        if self._default_color is None:
            self._default_color = _DEFAULT_NODE_COLOR

    def set_warning_status(self, msg: str) -> None:
        self._capture_default_color()
        with contextlib.suppress(AttributeError, RuntimeError):
            self._comp.color = _WARNING_COLOR
            self._comp.store("cuda_link_status_msg", f"WARNING: {msg}")

    def set_error_status(self, msg: str) -> None:
        self._capture_default_color()
        with contextlib.suppress(AttributeError, RuntimeError):
            self._comp.color = _ERROR_COLOR
            self._comp.addScriptError(msg)
            self._comp.store("cuda_link_status_msg", f"ERROR: {msg}")

    def clear_status(self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._default_color is not None:
                self._comp.color = self._default_color
            self._comp.clearScriptErrors(error="*")
            self._comp.unstore("cuda_link_status_msg")
