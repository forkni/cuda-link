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

# Minimum cudart version that defines cudaDevAttrIpcEventSupport (125).
# CUDA 11.x tops out at 121 (cudaDevAttrDeferredMappingCudaArraySupported); querying 125
# against an 11.x runtime returns cudaErrorInvalidValue.
CUDART_IPC_EVENT_SUPPORT_MIN_VERSION = 12000

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

    ABI note: CUDA 13.x's driver_types.h added a fifth field, `long reserved[8]`
    ("Must be zero"), that CUDA 12.x and earlier do not have. Under MSVC x64
    (`long` = 4 bytes) that grows the struct from 24 to 56 bytes. Since this
    binding's loader (_load_cuda_runtime() in cuda_ipc_wrapper.py) may attach to
    either a 12.x or 13.x cudart at runtime, the buffer is always sized to the
    larger (13.x) layout: an older cudart simply writes fewer bytes into it,
    which is safe, whereas a 24-byte buffer handed to a 13.x cudart is an
    undersized out-parameter (out-of-bounds write risk). The four original
    fields keep their offsets, so .type/.device access is unaffected either way.

    Uses `c_int32` (not `ctypes.c_long`) for `reserved`: this struct models the
    target Windows/MSVC cudart ABI, where `long` is always 4 bytes, but
    `ctypes.c_long` tracks the *host* platform's C `long` — 8 bytes on Linux/LP64
    (e.g. the CI runner running this file's no-GPU tests). `c_long * 8` would
    silently produce an 88-byte struct there instead of 56.
    """

    _fields_ = [
        ("type", c_int),  # cudaMemoryType enum (2 = cudaMemoryTypeDevice)
        ("device", c_int),  # GPU device index owning this allocation
        ("devicePointer", c_void_p),
        ("hostPointer", c_void_p),
        ("reserved", ctypes.c_int32 * 8),  # CUDA 13.x+ only; unused, must be zero
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
_abi_guard(ctypes.sizeof(cudaPointerAttributes), 56, "cudaPointerAttributes")
# Note on what these guards do and don't prove: they only check that THIS module's
# ctypes _fields_ produce the byte size we intend (a self-consistency check against
# drift in this file). They cannot detect a struct layout mismatch against whichever
# cudart DLL is actually loaded at runtime — see the runtime-version check in
# CUDARuntimeAPI._load_cuda_runtime() / check_ipc_capability() in cuda_ipc_wrapper.py
# for the runtime-vs-binding side of that problem.
# Graph param struct ABI guards — cudaMemcpy3DParms is the largest and most alignment-sensitive.
# All values were verified against the CUDA 12.8 and 13.3 headers on a 64-bit Windows host
# (sizeof c_size_t=8); cudaPos/cudaPitchedPtr/cudaExtent/cudaMemcpy3DParms are unchanged
# between those two toolkit versions.
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
    # Runtime API name for 201 is cudaErrorDeviceUninitialized (driver_types.h). The similarly
    # numbered CUDA_ERROR_INVALID_CONTEXT belongs to the separate Driver API CUresult enum
    # (cuda.h) -- a different code space this module's cudaError_t-based error handling does
    # not use. Kept the old name as an alias for one release cycle since nothing in this
    # codebase branches on it by name (get_name() is diagnostic-only), but corrected the
    # canonical constant per the CUDA 12.8 driver_types.h ground truth.
    DEVICE_UNINITIALIZED = 201  # Common in same-process IPC testing
    INVALID_CONTEXT = DEVICE_UNINITIALIZED  # deprecated alias -- use DEVICE_UNINITIALIZED
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
            201: "DEVICE_UNINITIALIZED",
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

# --- cudaDeviceAttr values used with get_device_attribute() ---
# NOTE: cudaDevAttrAsyncEngineCount was previously documented (and used in tests) as 4.
# Per CUDA 12.8/13.3 driver_types.h it is 40; attribute 4 is cudaDevAttrMaxBlockDimZ, an
# unrelated value that a caller using the old constant would silently receive instead.
CUDA_DEV_ATTR_ASYNC_ENGINE_COUNT: int = 40  # cudaDevAttrAsyncEngineCount — DMA copy engine count
CUDA_DEV_ATTR_IPC_EVENT_SUPPORT: int = 125  # cudaDevAttrIpcEventSupport — 0 if this device/driver
# cannot support CUDA IPC events (used by check_ipc_capability() in cuda_ipc_wrapper.py)
