"""
Native backend loader — adapts the compiled ``_spout_bridge`` pybind11 module to
the :class:`SpoutBackend` Protocol.

The compiled module (``spout/src/cuda_link_spout/_cpp/spout_bridge.cpp``) is built
on Windows with CUDA + D3D11 + Spout2 (see spout/README.md). It is intentionally
imported lazily — importing ``cuda_link_spout`` and using FakeSpoutBackend never
requires the native module, so the pure-Python layer and its tests run anywhere.
"""

from __future__ import annotations

import glob as _glob
import os as _os
import sys as _sys

from ._backend import NativeReceiveResult, SpoutBackend

# Python 3.8+ no longer searches PATH for DLL dependencies of extension modules.
# Explicitly register CUDA bin so cudart64_*.dll is visible when _spout_bridge loads.
if _sys.platform == "win32":
    for _cuda_bin in sorted(
        _glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"),
        reverse=True,
    ):
        if _os.path.isdir(_cuda_bin):
            _os.add_dll_directory(_cuda_bin)
            break


def load_native_backend(device: int = 0) -> SpoutBackend:
    """Import the compiled native module and return a SpoutBackend-conforming adapter.

    Raises:
        RuntimeError: with actionable guidance if the native module is not built/installed.
    """
    try:
        from . import _spout_bridge  # type: ignore[attr-defined]  # compiled extension
    except ImportError as e:  # pragma: no cover - exercised only without the built ext
        raise RuntimeError(
            "cuda-link-spout native module (_spout_bridge) is not available. It must be "
            "built on Windows with CUDA + D3D11 + the Spout2 SDK. See spout/README.md "
            "for build instructions. (The pure-Python API and tests work without it "
            "via FakeSpoutBackend.)"
        ) from e
    return _NativeSpoutBackend(_spout_bridge, device)


class _NativeSpoutBackend:
    """Thin adapter over the compiled ``_spout_bridge`` module.

    Kept deliberately thin: every method forwards 1:1 to the native module, which
    owns the D3D11 device, the CUDA↔D3D11 interop registration, and the de-swizzle
    copy. Satisfies :class:`SpoutBackend` structurally.
    """

    def __init__(self, mod: object, device: int) -> None:
        self._mod = mod
        self._device = device

    def create_sender(self, name, width, height, dxgi_format, device):
        return self._mod.create_sender(name, width, height, dxgi_format, device)

    def send(self, handle, src_ptr, src_pitch, width, height, bytes_per_pixel, stream):
        self._mod.send(handle, src_ptr, src_pitch, width, height, bytes_per_pixel, stream)

    def close_sender(self, handle):
        self._mod.close_sender(handle)

    def create_receiver(self, name, device):
        return self._mod.create_receiver(name, device)

    def receive(self, handle, dst_ptr, dst_pitch, max_bytes):
        r = self._mod.receive(handle, dst_ptr, dst_pitch, max_bytes)
        # The native module returns a tuple matching NativeReceiveResult's fields.
        return NativeReceiveResult(*r)

    def close_receiver(self, handle):
        self._mod.close_receiver(handle)

    def adapter_luid(self, device):
        return self._mod.adapter_luid(device)
