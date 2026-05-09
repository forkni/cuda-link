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
