"""
TDHost adapter — isolates all TouchDesigner runtime access behind a Protocol seam.

Every call that touches ownerComp, a TOP, or a Script TOP goes through this module.
Engine code imports nothing from the TD runtime; it calls TDHost / TOPHandle methods only.

textDAT name: TDHost  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class TOPHandle(Protocol):
    """Protocol for wrapping a single TouchDesigner TOP operator.

    Satisfied structurally — no inheritance required. Concrete adapters:
    RealTOPHandle (TD-connected) and FakeTOPHandle (in-process test double).
    """

    def cuda_memory(self, stream: Any = None) -> CUDAMemoryRef:
        """Call top.cudaMemory(stream=stream) and return a CUDAMemoryRef."""
        ...

    @property
    def pixel_format(self) -> str:
        """top.pixelFormat as a string."""
        ...

    def set_format(self, fmt: str) -> None:
        """Write top.par.format = fmt."""
        ...

    def copy_cuda_memory(self, ptr: int, size: int, shape: Any, *, stream: int) -> None:
        """Call script_top.copyCUDAMemory(ptr, size, shape, stream=stream)."""
        ...

    def copy_numpy_array(self, arr: Any) -> None:
        """Call script_top.copyNumpyArray(arr)."""
        ...

    def set_resolution(self, width: int, height: int) -> None:
        """Set Script TOP to custom resolution: outputresolution=9, resolutionw, resolutionh."""
        ...

    def is_valid(self) -> bool:
        """Return True if the underlying TD operator is still present in the network."""
        ...


# ---------------------------------------------------------------------------
# TDHost protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TDHost(Protocol):
    """Protocol for wrapping ownerComp.

    Satisfied structurally — no inheritance required. Concrete adapters:
    RealTDHost (TD-connected) and FakeTDHost (in-process test double, see _td_fakes.py).

    All TD runtime access (op(), parent(), me.par.*) goes through this Protocol.
    Engine code holds a TDHost reference and never imports TD globals directly.
    """

    def param_value(self, name: str) -> Any:
        """Read ownerComp.par.<name>.eval()."""
        ...

    def set_param_value(self, name: str, value: Any) -> None:
        """Write ownerComp.par.<name> = value."""
        ...

    def set_param_enabled(self, name: str, enabled: bool) -> None:
        """Write ownerComp.par.<name>.enable = enabled."""
        ...

    def show_custom_only(self, value: bool) -> None:
        """Write ownerComp.showCustomOnly = value."""
        ...

    def is_active(self) -> bool:
        """Read ownerComp.par.Active.eval() via cached reference (hot-path safe)."""
        ...

    def find_top(self, name: str) -> TOPHandle | None:
        """Return ownerComp.op(name) wrapped as a TOPHandle, or None."""
        ...

    def wrap_top(self, top: Any) -> TOPHandle:
        """Wrap a raw TD TOP operator as a TOPHandle.

        Use this factory instead of constructing RealTOPHandle directly —
        it keeps TD-runtime instantiation behind the seam so engine code
        and tests never import RealTOPHandle.
        """
        ...

    def set_warning_status(self, msg: str) -> None:
        """Tint ownerComp yellow to signal a recoverable warning (e.g. bad pixel format)."""
        ...

    def set_error_status(self, msg: str) -> None:
        """Tint ownerComp red and emit a persistent script-error badge for fatal failures."""
        ...

    def clear_status(self) -> None:
        """Restore ownerComp to its original color and clear any script-error badges."""
        ...

    def set_info_status(self, msg: str) -> None:
        """Write an informational status message to the Status par (no tint/cook side effects)."""
        ...


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

    def set_format(self, fmt: str) -> None:
        with contextlib.suppress(AttributeError, RuntimeError, Exception):
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
        # _default_color is captured lazily on the first set_warning_status /
        # set_error_status call so a tinted .tox save doesn't poison the cache.
        # _reset_stale_tint() clears any visible managed-colour tint immediately
        # so the COMP boots grey regardless of how it was saved.
        self._default_color: tuple[float, float, float] | None = None
        self._warning_emitter: Any = None  # lazily resolved; False = looked up, not found
        self._status_msg: str | None = None  # current stored status; drives cook-on-transition
        self._status_par_value: str | None = None  # last value written to Status par; drives dedup
        self._reset_stale_tint()

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

    def wrap_top(self, top: Any) -> RealTOPHandle:
        """Wrap a raw TD TOP operator as a RealTOPHandle."""
        return RealTOPHandle(top)

    def _cook_warning_emitter(self) -> None:
        if self._warning_emitter is None:
            with contextlib.suppress(AttributeError, RuntimeError):
                self._warning_emitter = self._comp.op("warning_emitter") or False
        if self._warning_emitter:
            with contextlib.suppress(AttributeError, RuntimeError):
                self._warning_emitter.cook(force=True)

    def _write_status_par(self, value: str) -> None:
        if self._status_par_value == value:
            return
        self._status_par_value = value
        self.set_param_value("Status", value)

    def _reset_stale_tint(self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            c = self._comp.color
            current = (float(c[0]), float(c[1]), float(c[2]))
            if current in _MANAGED_COLORS:
                self._comp.color = _DEFAULT_NODE_COLOR
                self._comp.clearScriptErrors(error="*")
                self._comp.unstore("cuda_link_status_msg")

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
        full_msg = f"WARNING: {msg}"
        needs_cook = self._status_msg != full_msg
        self._status_msg = full_msg
        with contextlib.suppress(AttributeError, RuntimeError):
            self._comp.color = _WARNING_COLOR
            self._comp.store("cuda_link_status_msg", full_msg)
        self._write_status_par(full_msg)
        if needs_cook:
            self._cook_warning_emitter()

    def set_error_status(self, msg: str) -> None:
        self._capture_default_color()
        full_msg = f"ERROR: {msg}"
        needs_cook = self._status_msg != full_msg
        self._status_msg = full_msg
        with contextlib.suppress(AttributeError, RuntimeError):
            self._comp.color = _ERROR_COLOR
            self._comp.addScriptError(msg)
            self._comp.store("cuda_link_status_msg", full_msg)
        self._write_status_par(full_msg)
        if needs_cook:
            self._cook_warning_emitter()

    def clear_status(self) -> None:
        needs_cook = self._status_msg is not None
        self._status_msg = None
        with contextlib.suppress(AttributeError, RuntimeError):
            if self._default_color is not None:
                self._comp.color = self._default_color
            self._comp.clearScriptErrors(error="*")
            self._comp.unstore("cuda_link_status_msg")
        self._write_status_par("Idle")
        if needs_cook:
            self._cook_warning_emitter()

    def set_info_status(self, msg: str) -> None:
        self._write_status_par(msg)
