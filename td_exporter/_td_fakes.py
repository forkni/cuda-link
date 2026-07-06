"""
_td_fakes.py — In-process test doubles for the TDHost / TOPHandle seam.

FakeTDHost and FakeTOPHandle satisfy TDHost and TOPHandle structurally
(via typing.Protocol) without a live TouchDesigner environment.

Usage in tests:
    from _td_fakes import FakeTDHost, FakeTOPHandle

    host = FakeTDHost(params={"Ipcmemname": "test_ipc", "Active": True})
    ext = CUDAIPCExtension(ownerComp=None, host=host)

textDAT name: _td_fakes
"""

from __future__ import annotations

from typing import Any

from TDHost import TDHost, TOPHandle  # noqa: E402


class FakeCUDAMemoryShape:
    """Minimal CUDAMemoryShape-shaped object returned by FakeTDHost.make_cuda_shape.

    Holds the four fields that TDReceiverEngine writes; tests can read them to
    assert the correct dims and dtype were supplied.
    """

    def __init__(self, width: int, height: int, num_comps: int, data_type: Any) -> None:
        self.width = width
        self.height = height
        self.numComps = num_comps
        self.dataType = data_type


class FakeCUDAMemoryRef:
    """Minimal CUDAMemoryRef-shaped object for tests that don't need the real type."""

    def __init__(
        self,
        ptr: int = 0xDEADBEEF,
        width: int = 64,
        height: int = 64,
        channels: int = 4,
        size: int = 0,
        data_type: Any = None,
    ) -> None:
        self.ptr = ptr
        self.width = width
        self.height = height
        self.channels = channels
        self.size = size or width * height * channels * 4
        self.data_type = data_type


class FakeTOPHandle(TOPHandle):
    """In-process test double for TOPHandle.

    Stores calls so tests can assert on what was invoked.
    """

    def __init__(
        self,
        pixel_format: str = "rgba32float",
        pixel_format_name: str = "rgba32float",
        width: int = 64,
        height: int = 64,
        channels: int = 4,
        gpu_ptr: int = 0xDEADBEEF,
        cuda_size: int | None = None,
        cuda_channels: int | None = None,
        cuda_data_type: Any = None,
    ) -> None:
        self._pixel_format = pixel_format
        self._pixel_format_name = pixel_format_name
        self._width = width
        self._height = height
        self._channels = channels
        self._gpu_ptr = gpu_ptr
        # Opt-in overrides for cuda_memory()'s reported size/channels/data_type — None (the
        # default) preserves the original fixed float32-sized behavior byte-for-byte. Tests
        # that need to simulate a TD source whose reported CUDA allocation size disagrees
        # with width*height*channels*4 (grow/shrink races, RGBA-padded mono, etc.) set these.
        self._cuda_size = cuda_size
        self._cuda_channels = cuda_channels
        self._cuda_data_type = cuda_data_type
        self.format_set: list[str] = []
        self.copy_cuda_calls: list[tuple] = []
        self.copy_numpy_calls: list[Any] = []
        self.resolution_set: list[tuple[int, int]] = []
        self.cook_calls: list[bool] = []

    def cuda_memory(self, stream: Any = None, pixel_format: str | None = None) -> FakeCUDAMemoryRef:
        # pixel_format is intentionally ignored in the fake — channel expansion is a TD runtime
        # behaviour that the in-process test double cannot replicate.
        channels = self._cuda_channels if self._cuda_channels is not None else self._channels
        size = self._cuda_size if self._cuda_size is not None else self._width * self._height * self._channels * 4
        return FakeCUDAMemoryRef(
            ptr=self._gpu_ptr,
            width=self._width,
            height=self._height,
            channels=channels,
            size=size,
            data_type=self._cuda_data_type,
        )

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    @property
    def pixel_format_name(self) -> str:
        return self._pixel_format_name

    def cook(self, force: bool = False) -> None:
        self.cook_calls.append(force)

    def set_format(self, fmt: str) -> None:
        self.format_set.append(fmt)

    def copy_cuda_memory(self, ptr: int, size: int, shape: Any, *, stream: int) -> None:
        self.copy_cuda_calls.append((ptr, size, shape, stream))

    def copy_numpy_array(self, arr: Any) -> None:
        self.copy_numpy_calls.append(arr)

    def set_resolution(self, width: int, height: int) -> None:
        self.resolution_set.append((width, height))

    def is_valid(self) -> bool:
        return True


class FakeTDHost(TDHost):
    """In-process test double for TDHost.

    Backed by a plain dict of parameter values; records all write calls.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        tops: dict[str, FakeTOPHandle] | None = None,
    ) -> None:
        self._params: dict[str, Any] = dict(params or {})
        self._tops: dict[str, FakeTOPHandle] = dict(tops or {})
        self.param_writes: list[tuple[str, Any]] = []
        self.enable_writes: list[tuple[str, bool]] = []
        self.custom_only_calls: list[bool] = []
        self.wrapped_tops: list[Any] = []
        self.cuda_shapes: list[FakeCUDAMemoryShape] = []
        # Records every status/tint side-effect call as (kind, msg); kind is one of
        # "warning" / "error" / "info" / "clear" (msg is None for clear). Additive —
        # existing callers that ignore the return value (always None) are unaffected.
        self.status_calls: list[tuple[str, str | None]] = []

    def param_value(self, name: str) -> Any:
        return self._params.get(name)

    def set_param_value(self, name: str, value: Any) -> None:
        self._params[name] = value
        self.param_writes.append((name, value))

    def set_param_enabled(self, name: str, enabled: bool) -> None:
        self.enable_writes.append((name, enabled))

    def show_custom_only(self, value: bool) -> None:
        self.custom_only_calls.append(value)

    def is_active(self) -> bool:
        return bool(self._params.get("Active", True))

    def find_top(self, name: str) -> FakeTOPHandle | None:
        return self._tops.get(name)

    def wrap_top(self, top: Any) -> FakeTOPHandle:
        """Return a FakeTOPHandle wrapping *top* (or a new default one for tests)."""
        self.wrapped_tops.append(top)
        if isinstance(top, FakeTOPHandle):
            return top
        return FakeTOPHandle()

    def make_cuda_shape(self, width: int, height: int, num_comps: int, data_type: Any) -> FakeCUDAMemoryShape:
        """Return a FakeCUDAMemoryShape and record the call for test assertions."""
        shape = FakeCUDAMemoryShape(width, height, num_comps, data_type)
        self.cuda_shapes.append(shape)
        return shape

    def set_warning_status(self, msg: str) -> None:
        self.status_calls.append(("warning", msg))

    def set_error_status(self, msg: str) -> None:
        self.status_calls.append(("error", msg))

    def clear_status(self) -> None:
        self.status_calls.append(("clear", None))

    def set_info_status(self, msg: str) -> None:
        self.status_calls.append(("info", msg))
