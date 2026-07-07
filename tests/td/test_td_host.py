"""
Unit tests for TDHost / TOPHandle adapters.

RealTDHost and RealTOPHandle wrap real TD objects; these tests drive them
against lightweight in-process stubs shaped like TD's runtime types.
FakeTDHost / FakeTOPHandle live in conftest.py and are used across the suite.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Lightweight TD stubs (shaped like TD runtime objects)
# ---------------------------------------------------------------------------


class _TDParValue:
    def __init__(self, value: Any) -> None:
        self._v = value
        self.enable = True

    def eval(self) -> Any:
        return self._v


class _TDPar:
    def __init__(self, **kwargs: Any) -> None:
        self._attrs: dict[str, _TDParValue] = {}
        for k, v in kwargs.items():
            self._attrs[k] = _TDParValue(v)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._attrs:
            return self._attrs[name]
        raise AttributeError(f"no parameter '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        elif name in self._attrs:
            self._attrs[name]._v = value
        else:
            self._attrs[name] = _TDParValue(value)


class _TDShape:
    def __init__(self, w: int, h: int, c: int) -> None:
        self.width = w
        self.height = h
        self.numComps = c
        self.dataType = None


class _TDCUDAMemory:
    def __init__(self, ptr: int, w: int, h: int, c: int, size: int) -> None:
        self.ptr = ptr
        self.shape = _TDShape(w, h, c)
        self.size = size


class _TDTOP:
    """Stub shaped like a TD TOP operator."""

    def __init__(
        self,
        name: str = "stub_top",
        width: int = 64,
        height: int = 64,
        channels: int = 4,
        pixel_format: str = "rgba32float",
        gpu_ptr: int = 0xDEADBEEF,
        inputs: list | None = None,
    ) -> None:
        self.name = name
        self._width = width
        self._height = height
        self._channels = channels
        self.pixelFormat = pixel_format
        self._gpu_ptr = gpu_ptr
        self.par = _TDPar(format="useinput", outputresolution=0, resolutionw=0, resolutionh=0)
        self._inputs = inputs or []
        self._copy_calls: list[tuple] = []
        self._numpy_calls: list[Any] = []

    def cudaMemory(self, stream: Any = None) -> _TDCUDAMemory:
        size = self._width * self._height * self._channels * 4
        return _TDCUDAMemory(self._gpu_ptr, self._width, self._height, self._channels, size)

    @property
    def inputs(self) -> list[_TDTOP]:
        return self._inputs

    def copyCUDAMemory(self, ptr: int, size: int, shape: Any, *, stream: int) -> None:
        self._copy_calls.append((ptr, size, shape, stream))

    def copyNumpyArray(self, arr: Any) -> None:
        self._numpy_calls.append(arr)


class _TDComp:
    """Stub shaped like a TD COMP operator (ownerComp)."""

    def __init__(self, **par_kwargs: Any) -> None:
        self.par = _TDPar(**par_kwargs)
        self._ops: dict[str, _TDTOP] = {}
        self.showCustomOnly = False

    def op(self, name: str) -> _TDTOP | None:
        return self._ops.get(name)


# ---------------------------------------------------------------------------
# RealTOPHandle tests
# ---------------------------------------------------------------------------


def test_real_top_handle_cuda_memory() -> None:
    """cuda_memory() wraps top.cudaMemory() into a CUDAMemoryRef."""
    from TDHost import CUDAMemoryRef, RealTOPHandle

    top = _TDTOP(width=128, height=64, channels=4, gpu_ptr=0xABCD1234)
    handle = RealTOPHandle(top)

    ref = handle.cuda_memory()

    assert isinstance(ref, CUDAMemoryRef)
    assert ref.ptr == 0xABCD1234
    assert ref.width == 128
    assert ref.height == 64
    assert ref.channels == 4
    assert ref.size == 128 * 64 * 4 * 4


def test_real_top_handle_cuda_memory_with_stream() -> None:
    """cuda_memory(stream=...) passes stream kwarg to top.cudaMemory()."""
    from TDHost import RealTOPHandle

    received: list[Any] = []

    class _StreamCaptureTOP(_TDTOP):
        def cudaMemory(self, stream: Any = None) -> _TDCUDAMemory:
            received.append(stream)
            return super().cudaMemory(stream=stream)

    top = _StreamCaptureTOP()
    handle = RealTOPHandle(top)
    sentinel = object()
    handle.cuda_memory(stream=sentinel)

    assert received == [sentinel]


def test_real_top_handle_pixel_format() -> None:
    """pixel_format returns top.pixelFormat as a string."""
    from TDHost import RealTOPHandle

    top = _TDTOP(pixel_format="rgba16float")
    handle = RealTOPHandle(top)

    assert handle.pixel_format == "rgba16float"


def test_real_top_handle_pixel_format_missing() -> None:
    """pixel_format returns '' when top has no pixelFormat attribute."""
    from TDHost import RealTOPHandle

    top = type("NoPF", (), {"par": _TDPar(), "inputs": []})()  # no pixelFormat attr
    handle = RealTOPHandle(top)
    assert handle.pixel_format == ""


def test_real_top_handle_set_format() -> None:
    """set_format() writes top.par.format."""
    from TDHost import RealTOPHandle

    top = _TDTOP()
    handle = RealTOPHandle(top)
    handle.set_format("rgba32float")

    assert top.par._attrs["format"]._v == "rgba32float"


def test_real_top_handle_copy_cuda_memory() -> None:
    """copy_cuda_memory() delegates to top.copyCUDAMemory()."""
    from TDHost import RealTOPHandle

    top = _TDTOP()
    handle = RealTOPHandle(top)
    fake_shape = object()
    handle.copy_cuda_memory(0xABCD, 1024, fake_shape, stream=42)

    assert len(top._copy_calls) == 1
    ptr, size, shape, stream = top._copy_calls[0]
    assert ptr == 0xABCD
    assert size == 1024
    assert shape is fake_shape
    assert stream == 42


def test_real_top_handle_copy_numpy_array() -> None:
    """copy_numpy_array() delegates to top.copyNumpyArray()."""
    from TDHost import RealTOPHandle

    top = _TDTOP()
    handle = RealTOPHandle(top)
    arr = object()
    handle.copy_numpy_array(arr)

    assert top._numpy_calls == [arr]


def test_real_top_handle_set_resolution() -> None:
    """set_resolution() sets outputresolution=9, resolutionw, resolutionh on Script TOP."""
    from TDHost import RealTOPHandle

    top = _TDTOP()
    handle = RealTOPHandle(top)
    handle.set_resolution(1920, 1080)

    assert top.par._attrs["outputresolution"]._v == 9
    assert top.par._attrs["resolutionw"]._v == 1920
    assert top.par._attrs["resolutionh"]._v == 1080


# ---------------------------------------------------------------------------
# RealTDHost tests
# ---------------------------------------------------------------------------


def test_real_td_host_param_value_reads_par() -> None:
    """param_value() returns ownerComp.par.<name>.eval()."""
    from TDHost import RealTDHost

    comp = _TDComp(Mode="Sender", Numslots=3)
    host = RealTDHost(comp)

    assert host.param_value("Mode") == "Sender"
    assert host.param_value("Numslots") == 3


def test_real_td_host_param_value_missing_returns_none() -> None:
    """param_value() returns None when the parameter does not exist."""
    from TDHost import RealTDHost

    comp = _TDComp()
    host = RealTDHost(comp)

    assert host.param_value("NonExistentPar") is None


def test_real_td_host_set_param_value() -> None:
    """set_param_value() writes ownerComp.par.<name> = value."""
    from TDHost import RealTDHost

    comp = _TDComp(Numslots=3)
    host = RealTDHost(comp)
    host.set_param_value("Numslots", 5)

    assert comp.par._attrs["Numslots"]._v == 5


def test_real_td_host_set_param_enabled() -> None:
    """set_param_enabled() writes ownerComp.par.<name>.enable."""
    from TDHost import RealTDHost

    comp = _TDComp(Numslots=3)
    host = RealTDHost(comp)
    host.set_param_enabled("Numslots", False)

    assert comp.par._attrs["Numslots"].enable is False


def test_real_td_host_show_custom_only() -> None:
    """show_custom_only() writes ownerComp.showCustomOnly."""
    from TDHost import RealTDHost

    comp = _TDComp()
    host = RealTDHost(comp)
    host.show_custom_only(True)

    assert comp.showCustomOnly is True


def test_real_td_host_is_active_true() -> None:
    """is_active() returns True when Active parameter is True."""
    from TDHost import RealTDHost

    comp = _TDComp(Active=True)
    host = RealTDHost(comp)

    assert host.is_active() is True


def test_real_td_host_is_active_false() -> None:
    """is_active() returns False when Active parameter is False."""
    from TDHost import RealTDHost

    comp = _TDComp(Active=False)
    host = RealTDHost(comp)

    assert host.is_active() is False


def test_real_td_host_is_active_no_par_defaults_true() -> None:
    """is_active() returns True when ownerComp has no Active parameter."""
    from TDHost import RealTDHost

    comp = _TDComp()  # no Active parameter
    host = RealTDHost(comp)

    assert host.is_active() is True


def test_real_td_host_find_top_returns_handle() -> None:
    """find_top() wraps ownerComp.op(name) as a RealTOPHandle."""
    from TDHost import RealTDHost, RealTOPHandle

    comp = _TDComp()
    top = _TDTOP(name="ExportBuffer")
    comp._ops["ExportBuffer"] = top
    host = RealTDHost(comp)

    handle = host.find_top("ExportBuffer")

    assert isinstance(handle, RealTOPHandle)
    assert handle.pixel_format == top.pixelFormat


def test_real_td_host_find_top_missing_returns_none() -> None:
    """find_top() returns None when op does not exist in the network."""
    from TDHost import RealTDHost

    comp = _TDComp()
    host = RealTDHost(comp)

    assert host.find_top("NonExistentTOP") is None


def test_real_top_handle_is_valid_present() -> None:
    """is_valid() returns True when underlying TOP has valid=True."""
    from TDHost import RealTOPHandle

    top = type("ValidTOP", (), {"valid": True, "par": _TDPar(), "inputs": []})()
    handle = RealTOPHandle(top)
    assert handle.is_valid() is True


def test_real_top_handle_is_valid_missing_attr_defaults_true() -> None:
    """is_valid() returns True when top has no valid attribute."""
    from TDHost import RealTOPHandle

    top = type("NoValidTOP", (), {"par": _TDPar(), "inputs": []})()
    handle = RealTOPHandle(top)
    assert handle.is_valid() is True


def test_real_top_handle_is_valid_false() -> None:
    """is_valid() returns False when TOP.valid is False (e.g. deleted from network)."""
    from TDHost import RealTOPHandle

    top = type("DeletedTOP", (), {"valid": False, "par": _TDPar(), "inputs": []})()
    handle = RealTOPHandle(top)
    assert handle.is_valid() is False


# ---------------------------------------------------------------------------
# RealTDHost.make_cuda_shape tests
# ---------------------------------------------------------------------------


def test_real_tdhost_make_cuda_shape_sets_all_fields(monkeypatch: Any) -> None:
    """make_cuda_shape() constructs a TD CUDAMemoryShape and sets all four fields."""
    import numpy as np
    from TDHost import RealTDHost

    # Inject a minimal CUDAMemoryShape stub into the TDHost module namespace.
    # RealTDHost.make_cuda_shape references it as a free name (TD interpreter global).
    class _FakeShape:
        pass

    import TDHost as TDHostMod  # noqa: N813

    monkeypatch.setattr(TDHostMod, "CUDAMemoryShape", _FakeShape, raising=False)

    comp = _TDComp()
    host = RealTDHost(comp)

    shape = host.make_cuda_shape(1920, 1080, 4, np.float32)

    assert isinstance(shape, _FakeShape)
    assert shape.width == 1920
    assert shape.height == 1080
    assert shape.numComps == 4
    assert shape.dataType is np.float32


def test_real_tdhost_make_cuda_shape_uint8(monkeypatch: Any) -> None:
    """make_cuda_shape() correctly stores numpy.uint8 scalar type (not dtype instance)."""
    import numpy as np
    from TDHost import RealTDHost

    class _FakeShape:
        pass

    import TDHost as TDHostMod  # noqa: N813

    monkeypatch.setattr(TDHostMod, "CUDAMemoryShape", _FakeShape, raising=False)

    comp = _TDComp()
    host = RealTDHost(comp)

    shape = host.make_cuda_shape(64, 64, 1, np.uint8)

    assert shape.dataType is np.uint8
    assert shape.numComps == 1


# ---------------------------------------------------------------------------
# FakeTDHost.make_cuda_shape tests
# ---------------------------------------------------------------------------


def test_fake_tdhost_make_cuda_shape_records_call() -> None:
    """FakeTDHost.make_cuda_shape returns FakeCUDAMemoryShape and records the call."""
    import numpy as np
    from fakes import FakeCUDAMemoryShape, FakeTDHost

    host = FakeTDHost()
    shape = host.make_cuda_shape(320, 240, 3, np.uint8)

    assert isinstance(shape, FakeCUDAMemoryShape)
    assert shape.width == 320
    assert shape.height == 240
    assert shape.numComps == 3
    assert shape.dataType is np.uint8
    assert len(host.cuda_shapes) == 1
    assert host.cuda_shapes[0] is shape


def test_fake_tdhost_make_cuda_shape_multiple_calls() -> None:
    """Each make_cuda_shape call appends a distinct object to cuda_shapes."""
    import numpy as np
    from fakes import FakeTDHost

    host = FakeTDHost()
    s1 = host.make_cuda_shape(64, 64, 4, np.float32)
    s2 = host.make_cuda_shape(128, 128, 4, np.float32)

    assert len(host.cuda_shapes) == 2
    assert host.cuda_shapes[0] is s1
    assert host.cuda_shapes[1] is s2
    assert s1 is not s2
