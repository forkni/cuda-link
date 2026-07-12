"""
CUDA Runtime Types — ctypes structs, type aliases, and error codes for CUDA IPC.

Shared between the pip package (cuda_link) and TouchDesigner textDATs.
Compatible with both Python package and TD COMP namespace imports.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_size_t, c_uint64, c_void_p
from enum import IntEnum

# CUDA handle types - use unsigned 64-bit to prevent overflow on Windows x64
# See: https://github.com/pytorch/pytorch/pull/162920
CUDAEvent_t = c_uint64  # cudaEvent_t opaque pointer
CUDAStream_t = c_uint64  # cudaStream_t opaque pointer
CUDAGraph_t = c_uint64  # cudaGraph_t opaque pointer (CUDA 10.0+)
CUDAGraphExec_t = c_uint64  # cudaGraphExec_t opaque pointer (CUDA 10.0+)
CUDAGraphNode_t = c_uint64  # cudaGraphNode_t opaque pointer (CUDA 10.0+)

# Minimum cudart version required for all CUDA Graphs APIs used by this module.
# cudaGraphInstantiateWithFlags, cudaGraphExecEventRecordNodeSetEvent, and
# cudaGraphExecEventWaitNodeSetEvent are all CUDA 11.4+ (version integer 11040).
CUDART_GRAPHS_MIN_VERSION = 11040

# --- CUDA Graph parameter structs ---


class cudaPos(ctypes.Structure):
    """cudaPos: {x, y, z} offsets into an array or pitched memory."""

    _fields_ = [("x", c_size_t), ("y", c_size_t), ("z", c_size_t)]


class cudaPitchedPtr(ctypes.Structure):
    """cudaPitchedPtr: pointer + pitch metadata for 2D/3D copies."""

    _fields_ = [
        ("ptr", c_void_p),
        ("pitch", c_size_t),
        ("xsize", c_size_t),
        ("ysize", c_size_t),
    ]


class cudaExtent(ctypes.Structure):
    """cudaExtent: width/height/depth dimensions in bytes for 3D copies."""

    _fields_ = [("width", c_size_t), ("height", c_size_t), ("depth", c_size_t)]


class cudaMemcpy3DParms(ctypes.Structure):
    """cudaMemcpy3DParms: full parameter struct for cudaMemcpy3D and graph node updates."""

    _fields_ = [
        ("srcArray", c_void_p),  # cudaArray_t — NULL for linear memory
        ("srcPos", cudaPos),
        ("srcPtr", cudaPitchedPtr),
        ("dstArray", c_void_p),  # cudaArray_t — NULL for linear memory
        ("dstPos", cudaPos),
        ("dstPtr", cudaPitchedPtr),
        ("extent", cudaExtent),
        ("kind", c_int),  # cudaMemcpyKind
    ]


# CUDA IPC Handle structure (64 bytes, CUDA_IPC_HANDLE_SIZE per NVIDIA spec)
class cudaIpcMemHandle_t(ctypes.Structure):
    """CUDA IPC memory handle structure.

    This opaque handle can be transferred between processes via
    SharedMemory or other IPC mechanisms to enable GPU memory sharing.
    """

    _fields_ = [("internal", ctypes.c_byte * 64)]


# CUDA IPC Event Handle structure (64 bytes per NVIDIA spec)
class cudaIpcEventHandle_t(ctypes.Structure):
    """CUDA IPC event handle structure.

    Used for lightweight cross-process synchronization.
    """

    _fields_ = [("reserved", ctypes.c_byte * 64)]


# CUDA pointer attributes — memory type and owning device for a GPU pointer
class cudaPointerAttributes(ctypes.Structure):
    """Result of cudaPointerGetAttributes.

    Useful for validating that a caller-supplied GPU pointer belongs to the
    expected device before issuing D2D operations (C2 affinity check).

    .type values: 0=unregistered, 1=host, 2=device, 3=managed
    .device: GPU index that owns the allocation
    """

    _fields_ = [
        ("type", c_int),  # cudaMemoryType enum (2 = cudaMemoryTypeDevice)
        ("device", c_int),  # GPU device index owning this allocation
        ("devicePointer", c_void_p),
        ("hostPointer", c_void_p),
    ]


def _abi_guard(actual: int, expected: int, name: str) -> None:
    """Raise RuntimeError on ctypes struct size drift.

    Deliberately NOT an `assert` — asserts are stripped under `python -O`, which would
    let a struct layout mismatch pass silently instead of failing loudly at import time.
    """
    if actual != expected:
        raise RuntimeError(f"{name} ABI mismatch: expected {expected} bytes, got {actual}")


_abi_guard(ctypes.sizeof(cudaIpcMemHandle_t), 64, "cudaIpcMemHandle_t")
_abi_guard(ctypes.sizeof(cudaIpcEventHandle_t), 64, "cudaIpcEventHandle_t")
_abi_guard(ctypes.sizeof(cudaPointerAttributes), 24, "cudaPointerAttributes")
# Graph param struct ABI guards — cudaMemcpy3DParms is the largest and most alignment-sensitive.
# All four values were verified against Python ctypes on a 64-bit Windows host (sizeof c_size_t=8).
_abi_guard(ctypes.sizeof(cudaPos), 24, "cudaPos")
_abi_guard(ctypes.sizeof(cudaPitchedPtr), 32, "cudaPitchedPtr")
_abi_guard(ctypes.sizeof(cudaExtent), 24, "cudaExtent")
_abi_guard(ctypes.sizeof(cudaMemcpy3DParms), 160, "cudaMemcpy3DParms")


# CUDA Error codes (subset)
class CUDAError:
    """CUDA runtime error codes."""

    SUCCESS = 0
    INVALID_VALUE = 1
    MEMORY_ALLOCATION = 2
    INVALID_DEVICE_POINTER = 17
    INVALID_DEVICE = 101
    INVALID_CONTEXT = 201  # Common in same-process IPC testing
    NOT_READY = 600
    PEER_ACCESS_ALREADY_ENABLED = 704

    @staticmethod
    def get_name(code: int) -> str:
        """Get human-readable error name."""
        names = {
            0: "SUCCESS",
            1: "INVALID_VALUE",
            2: "MEMORY_ALLOCATION",
            17: "INVALID_DEVICE_POINTER",
            101: "INVALID_DEVICE",
            201: "INVALID_CONTEXT",
            600: "NOT_READY",
            704: "PEER_ACCESS_ALREADY_ENABLED",
        }
        return names.get(code, f"UNKNOWN_ERROR_{code}")


# ---------------------------------------------------------------------------
# Library-specific exception hierarchy
# ---------------------------------------------------------------------------
#
# Previously every failure here raised a bare builtin (RuntimeError / ValueError /
# TimeoutError), so callers could not `except CudaLinkError` without also catching
# unrelated bugs that happen to raise the same builtin. Each subclass below ALSO
# inherits the builtin it replaces, so this is purely additive: every existing
# `except RuntimeError` / `except ValueError` / `except TimeoutError` call site in
# this codebase keeps catching these unchanged, while new code can narrow to
# `except CudaLinkError` (or a specific subclass) instead.


class CudaLinkError(RuntimeError):
    """Base class for all cuda-link library-specific exceptions."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code  # raw CUDA error code, when available — see CUDAError.get_name()


class CudaIpcError(CudaLinkError):
    """A CUDA driver/runtime call (alloc, IPC handle, event, memcpy, ...) failed."""


class ProtocolError(CudaLinkError, ValueError):
    """The SHM wire protocol was violated (bad magic, version, or field value).

    Also inherits ValueError so existing `except ValueError` sites keep working.
    """


class ProducerTimeoutError(CudaLinkError, TimeoutError):
    """Timed out waiting for the producer to signal a slot/doorbell.

    Also inherits TimeoutError so existing `except TimeoutError` sites
    (e.g. per-frame timeout handling in importer.py) keep working unchanged.
    """


# ---------------------------------------------------------------------------
# CUDA runtime enum constants — named replacements for magic integers
# ---------------------------------------------------------------------------


class MemcpyKind(IntEnum):
    """cudaMemcpyKind values used in cudaMemcpy / cudaMemcpyAsync calls."""

    HOST_TO_HOST = 0  # cudaMemcpyHostToHost
    HOST_TO_DEVICE = 1  # cudaMemcpyHostToDevice
    DEVICE_TO_HOST = 2  # cudaMemcpyDeviceToHost
    DEVICE_TO_DEVICE = 3  # cudaMemcpyDeviceToDevice


class StreamFlags(IntEnum):
    """cudaStreamFlags values passed to cudaStreamCreate* calls."""

    DEFAULT = 0  # cudaStreamDefault (inherits legacy synchronisation behaviour)
    NON_BLOCKING = 0x01  # cudaStreamNonBlocking


class StreamCaptureMode(IntEnum):
    """cudaStreamCaptureMode values for cudaStreamBeginCapture."""

    GLOBAL = 0  # cudaStreamCaptureModeGlobal
    THREAD_LOCAL = 1  # cudaStreamCaptureModeThreadLocal
    RELAXED = 2  # cudaStreamCaptureModeRelaxed


# cudaIpcMemLazyEnablePeerAccess — the only valid flag for cudaIpcOpenMemHandle.
IPC_MEM_LAZY_ENABLE_PEER_ACCESS: int = 1

# cudaHostAllocPortable — pinned allocation visible from any CUDA context in the process.
HOST_ALLOC_PORTABLE: int = 0x01
