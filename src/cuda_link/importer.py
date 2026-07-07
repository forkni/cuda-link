"""
Importer — deep consumer-side module for CUDA IPC GPU memory.

Importer.open() is the single entry point. It returns a fully-initialized
Importer ready for get_frame*() calls. Each method returns an ImportResult
typed to the backend (torch.Tensor, np.ndarray, cp.ndarray).

Value objects (moved from cuda_ipc_importer.py, interface unchanged):
    Format          — parsed frame geometry + dtype; from_shm / from_overrides.
    IPCConnection   — live CUDA IPC connection (SHM handle, per-slot ptrs + events).
    TorchBuffers    — per-slot zero-copy torch.Tensor views (built eagerly).
    CupyBuffers     — per-slot zero-copy CuPy array views (built eagerly).
    NumpyBuffers    — pinned host buffer + D2H streams (built lazily).
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any, Protocol

from . import _doorbell, _nvtx
from ._importer_port import (
    ImportOutcome,
    ImportPolicy,
    ImportResult,
    ImportSpec,
)
from .cuda_runtime_types import IPC_MEM_LAZY_ENABLE_PEER_ACCESS, CudaIpcError, ProducerTimeoutError, ProtocolError
from .shm_protocol import (
    IPC_HANDLE_SIZE,
    MAGIC_OFFSET,
    MAGIC_SIZE,
    PROTOCOL_MAGIC,
    WRITE_IDX_OFFSET,
    AcquireResult,
    DtypeCodec,
    Metadata,
    SHMLayout,
    SlotState,
    acquire_slot,
    read_num_slots,
    read_version,
    read_write_idx,
)

if TYPE_CHECKING:
    from cuda_link._wait_backend import WaitBackend

    from ._importer_port import ImporterCudaPort
    from .nvml_observer import NVMLObserver

logger = logging.getLogger(__name__)

# Pre-built NVTX strings eliminate f-string allocation on the hot path.
_NVTX_GET_NAMES: tuple[str, ...] = _nvtx.slot_names("cudalink.importer.get_frame.slot")
_NVTX_NUMPY_NAMES: tuple[str, ...] = _nvtx.slot_names("cudalink.importer.get_frame_numpy.slot")
_NVTX_CUPY_NAMES: tuple[str, ...] = _nvtx.slot_names("cudalink.importer.get_frame_cupy.slot")


# Windows timer-resolution helper — reduces time.sleep floor from ~15ms to ~1ms.
# timeBeginPeriod/timeEndPeriod return their status directly (TIMERR_NOERROR=0,
# TIMERR_NOCANDO=97); they do NOT use GetLastError, so use_last_error is omitted.
_TIMERR_NOCANDO = 97  # mmsystem.h TIMERR_NOCANDO — period granularity unsupported
if sys.platform == "win32" and sys.version_info < (3, 11):
    # CPython 3.11+ uses a high-resolution waitable timer on Windows (100 ns class),
    # so timeBeginPeriod is unnecessary and the syscall pair would be wasted work.
    # On 3.10 and earlier, timeBeginPeriod(1) reduces time.sleep()'s floor from ~15 ms
    # to ~1 ms — see cpython/issues/89592 and What's New in Python 3.11.
    try:
        _winmm = ctypes.WinDLL("winmm")
        _winmm.timeBeginPeriod.argtypes = [ctypes.c_uint]
        _winmm.timeBeginPeriod.restype = ctypes.c_uint
        _winmm.timeEndPeriod.argtypes = [ctypes.c_uint]
        _winmm.timeEndPeriod.restype = ctypes.c_uint
    except OSError:
        _winmm = None
else:
    _winmm = None


class _HighResTimer:
    """Context manager: request 1ms timer resolution on Windows; no-op elsewhere."""

    __slots__ = ("_active",)

    def __enter__(self) -> _HighResTimer:
        self._active = _winmm is not None
        if self._active:
            r = _winmm.timeBeginPeriod(1)
            if r != 0:
                logger.debug(
                    "timeBeginPeriod(1) returned %d (TIMERR_NOCANDO=%d); "
                    "high-resolution timer unavailable — sleep floor stays at ~15ms",
                    r,
                    _TIMERR_NOCANDO,
                )
        return self

    def __exit__(self, *_: object) -> None:
        if self._active:
            _winmm.timeEndPeriod(1)


# Optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


def _numpy_dtype_for(dtype_str: str) -> object:
    """Return np.dtype for dtype_str, handling bfloat16 specially.

    Returns None when NumPy is unavailable.  For bfloat16 (DtypeCodec.numpy_name
    returns None), attempts ml_dtypes.bfloat16 and returns None if that package is
    absent (NumpyBuffers will raise a clear error rather than silently miscompute).
    """
    if not NUMPY_AVAILABLE:
        return None
    name = DtypeCodec.numpy_name(dtype_str)
    if name is None:  # bfloat16 — needs ml_dtypes
        try:
            import ml_dtypes  # noqa: PLC0415

            return np.dtype(ml_dtypes.bfloat16)
        except ImportError:
            return None
    return np.dtype(name)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Format:
    """Parsed frame format — shape, dtype, and precomputed derivations.

    Immutable after construction. Two constructors:
      from_shm():       parse extended metadata block in SharedMemory.
      from_overrides(): build from caller-supplied shape/dtype (no SHM read).
    """

    width: int
    height: int
    num_comps: int
    kind: int
    bits: int
    flags: int
    dtype_str: str
    shape: tuple
    numpy_dtype: object  # np.dtype | None when numpy not available
    frame_nbytes: int

    @classmethod
    def from_metadata(cls, md: Metadata) -> Format:
        """Build from an already-read Metadata value object.

        The caller is responsible for reading and validating the Metadata
        (e.g. checking md.data_size == md.expected_size) before calling this.
        Use from_shm() when reading directly from shared memory.
        """
        dtype_str = DtypeCodec.decode(md.format_kind, md.bits_per_comp, md.flags)
        itemsize = DtypeCodec.itemsize(dtype_str)
        shape = (md.height, md.width, md.num_comps)
        frame_nbytes = md.height * md.width * md.num_comps * itemsize
        numpy_dtype = _numpy_dtype_for(dtype_str)
        return cls(
            width=md.width,
            height=md.height,
            num_comps=md.num_comps,
            kind=md.format_kind,
            bits=md.bits_per_comp,
            flags=md.flags,
            dtype_str=dtype_str,
            shape=shape,
            numpy_dtype=numpy_dtype,
            frame_nbytes=frame_nbytes,
        )

    @classmethod
    def from_shm(cls, shm_buf: object, num_slots: int) -> Format | None:
        """Parse extended metadata block from shared memory.

        Returns None when the block is absent or contains zeros.
        Routes through shm_protocol.Metadata.read_from — do not replicate
        the struct decoding here.
        """
        layout = SHMLayout(num_slots)
        try:
            md = Metadata.read_from(shm_buf, layout)
            if md.width > 0 and md.height > 0 and md.num_comps > 0:
                return cls.from_metadata(md)
        except (struct.error, ValueError, IndexError):
            pass
        return None

    @classmethod
    def from_overrides(cls, shape: tuple, dtype_str: str) -> Format:
        """Build from caller-supplied shape/dtype (no SHM read).

        kind/bits/flags are 0 sentinels — diagnostic wire-format fields that
        cannot be known without reading the SHM metadata block.  These sentinel
        zeros participate in the auto-generated ``__eq__``, so do NOT use
        ``fmt_a == fmt_b`` to detect layout changes between an override-derived
        and an SHM-derived Format — use ``fmt_a.layout_differs_from(fmt_b)``
        instead (see ``_reinitialize`` for the established precedent).
        """
        height, width, num_comps = shape
        try:
            itemsize = DtypeCodec.itemsize(dtype_str)
        except KeyError:
            itemsize = 4  # safe fallback for caller-supplied unknown dtypes
        frame_nbytes = height * width * num_comps * itemsize
        numpy_dtype = _numpy_dtype_for(dtype_str)
        return cls(
            width=width,
            height=height,
            num_comps=num_comps,
            kind=0,
            bits=0,
            flags=0,
            dtype_str=dtype_str,
            shape=shape,
            numpy_dtype=numpy_dtype,
            frame_nbytes=frame_nbytes,
        )

    def layout_differs_from(self, other: Format) -> bool:
        """True when shape or dtype changed between two Formats.

        Do NOT use == (Format.__eq__) for layout-change detection — kind/bits/flags
        sentinel zeros in from_overrides() would false-positive against SHM-derived
        real values. Compare only the load-bearing layout fields.
        """
        return self.shape != other.shape or self.dtype_str != other.dtype_str


@dataclass
class IPCConnection:
    """Live CUDA IPC connection — runtime, SHM handle, per-slot GPU resources, layout.

    Mutable: dev_ptrs/ipc_events/ipc_handles are populated slot-by-slot during
    _open_ipc_slots(), then nulled in-place by close_ipc_handles() / close().
    """

    cuda: ImporterCudaPort
    shm_handle: object  # SharedMemory | None after close()
    ipc_version: int
    num_slots: int
    ipc_handles: list
    dev_ptrs: list  # [c_void_p | None]
    ipc_events: list  # [event_t | None]
    layout: object  # SHMLayout
    shutdown_offset: int
    timestamp_offset: int
    # R2: Win32 doorbell handle (None when CUDALINK_DOORBELL=0 or non-Windows)
    doorbell_handle: object = None

    def close_ipc_handles(self) -> None:
        """Close IPC mem handles and events. SharedMemory stays open."""
        for slot, dev_ptr in enumerate(self.dev_ptrs):
            if dev_ptr is not None:
                try:
                    self.cuda.ipc_close_mem_handle(dev_ptr)
                    logger.info("Closed IPC handle for slot %d", slot)
                except (RuntimeError, OSError) as e:
                    logger.error("Error closing IPC handle for slot %d: %s", slot, e)
                self.dev_ptrs[slot] = None

        for slot, event in enumerate(self.ipc_events):
            if event is not None:
                try:
                    self.cuda.destroy_event(event)
                    logger.info("Destroyed IPC event for slot %d", slot)
                except (RuntimeError, OSError) as e:
                    logger.error("Error destroying event for slot %d: %s", slot, e)
                self.ipc_events[slot] = None

        # R2: close doorbell handle alongside IPC handles
        if self.doorbell_handle is not None:
            _doorbell.close(self.doorbell_handle)
            self.doorbell_handle = None

    def close(self) -> None:
        """Close IPC handles and SharedMemory. Idempotent."""
        self.close_ipc_handles()
        if self.shm_handle is not None:
            try:
                self.shm_handle.close()
                logger.info("Closed SharedMemory")
            except (OSError, BufferError) as e:
                logger.error("Error closing SharedMemory: %s", e)
            self.shm_handle = None


class CUDAArrayWrapper:
    """Adapts a raw __cuda_array_interface__ dict for torch.as_tensor()."""

    def __init__(self, interface: dict) -> None:
        self.__cuda_array_interface__ = interface


@dataclass
class TorchBuffers:
    """Per-slot zero-copy torch.Tensor views of GPU memory (built eagerly at init)."""

    tensors: list
    wrappers: list  # GC keep-alive refs for __cuda_array_interface__ wrappers

    @classmethod
    def build(cls, conn: IPCConnection, fmt: Format) -> TorchBuffers:
        """Create one zero-copy tensor view per slot via __cuda_array_interface__.

        bfloat16: the CAI protocol has no bfloat16 typestr, so we use a uint16
        backing view ("<u2") and reinterpret with tensor.view(torch.bfloat16).
        DtypeCodec.typestr() returns "<u2" for bfloat16 exactly for this reason.
        """
        try:
            typestr = DtypeCodec.typestr(fmt.dtype_str)
        except KeyError:
            raise ValueError(f"Unsupported dtype for torch: {fmt.dtype_str}") from None

        tensors = []
        wrappers = []
        for slot in range(conn.num_slots):
            if conn.dev_ptrs[slot] is None:
                raise RuntimeError(f"Device pointer for slot {slot} not initialized")

            ptr_value = int(conn.dev_ptrs[slot].value) if conn.dev_ptrs[slot].value is not None else 0
            cuda_array_interface = {
                "shape": fmt.shape,
                "typestr": typestr,
                "data": (ptr_value, False),
                "version": 3,
                "strides": None,
            }

            wrapper = CUDAArrayWrapper(cuda_array_interface)
            tensor = torch.as_tensor(wrapper, device="cuda")
            if fmt.dtype_str == "bfloat16":
                tensor = tensor.view(torch.bfloat16)
            wrappers.append(wrapper)
            tensors.append(tensor)

        return cls(tensors=tensors, wrappers=wrappers)


@dataclass
class CupyBuffers:
    """Per-slot zero-copy CuPy array views of GPU memory (built eagerly at init)."""

    arrays: list

    @classmethod
    def build(cls, conn: IPCConnection, fmt: Format) -> CupyBuffers:
        """Create one zero-copy CuPy array view per slot via UnownedMemory.

        bfloat16 is not supported by CuPy (no cp.bfloat16 dtype); callers
        should use get_frame() to receive a torch.Tensor for bfloat16 data.
        DtypeCodec.cupy_name() returns None for bfloat16 to signal this.
        """
        cupy_name = DtypeCodec.cupy_name(fmt.dtype_str)
        if cupy_name is None:
            raise ValueError(
                f"{fmt.dtype_str} is not supported by the CuPy consumer "
                f"(CuPy has no {fmt.dtype_str} dtype). "
                "Use get_frame() to retrieve a torch.Tensor instead."
            )
        try:
            cp_dtype = cp.dtype(cupy_name)
        except (TypeError, AttributeError):
            raise ValueError(f"Unsupported dtype for CuPy: {fmt.dtype_str}") from None

        arrays = []
        for slot in range(conn.num_slots):
            if conn.dev_ptrs[slot] is None:
                raise RuntimeError(f"Device pointer for slot {slot} not initialized")
            ptr_value = int(conn.dev_ptrs[slot].value)
            mem = cp.cuda.UnownedMemory(ptr_value, fmt.frame_nbytes, owner=conn)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            arrays.append(cp.ndarray(fmt.shape, dtype=cp_dtype, memptr=memptr))

        return cls(arrays=arrays)


@dataclass
class NumpyBuffers:
    """Pinned host buffer + D2H streams for numpy frame consumption (built lazily).

    NumpyBuffers owns the CUDA streams and pinned host allocation.
    close() tears them down idempotently.
    """

    cuda: ImporterCudaPort
    fmt: Format
    buffer: object  # np.ndarray — reusable D2H destination
    pinned_ptr: object  # c_void_p | None
    host_registered_arr: object  # np.ndarray | None for cudaHostRegister fallback
    pinned_memory_available: bool
    primary_stream: object
    d2h_streams: list
    num_streams: int
    chunk_plan: list  # [(offset, size), ...] for multi-stream D2H; empty when num_streams <= 1
    buffer_ptr: object  # ctypes.c_void_p | None — precomputed dst for D2H copy (fixed for buffer lifetime)
    # pipelined D2H double-buffer state (all None / False when not pipelined)
    back_buffer: object = None  # second pinned host buffer
    back_pinned_ptr: object = None  # CUDA ptr from malloc_host_alloc for back buffer
    back_host_registered_arr: object = None  # ndarray from cudaHostRegister for back buffer
    back_buffer_ptr: object = None  # precomputed c_void_p for back_buffer
    pipelined: bool = False  # pipelined D2H active
    priming: bool = False  # True before the first copy completes

    @classmethod
    def build(
        cls,
        conn: IPCConnection,
        fmt: Format,
        num_streams: int,
        high_priority: bool = False,
        allow_pageable: bool = False,
        pipelined: bool = False,
    ) -> NumpyBuffers:
        """Allocate pinned host buffer + D2H streams.

        Allocation ladder: cudaMallocHost (portable pinned) → cudaHostRegister
        (page-locked) → pageable fallback (when allow_pageable=True).

        Raises ValueError for dtypes with no NumPy representation (bfloat16
        without ml_dtypes installed).  Use get_frame() for those dtypes.
        """
        if fmt.numpy_dtype is None:
            raise ValueError(
                f"NumPy consumer cannot be used for dtype {fmt.dtype_str!r}: "
                "no compatible NumPy dtype is available. "
                "For bfloat16, install ml_dtypes; or use get_frame() (torch) instead."
            )
        cuda = conn.cuda
        nbytes = fmt.frame_nbytes

        if high_priority:
            primary_stream = cuda.create_stream_with_priority(flags=0x01)
            d2h_streams = [primary_stream] + [
                cuda.create_stream_with_priority(flags=0x01) for _ in range(num_streams - 1)
            ]
            logger.info("D2H streams created at HIGH priority")
        else:
            primary_stream = cuda.create_stream(flags=0x01)
            d2h_streams = [primary_stream] + [cuda.create_stream(flags=0x01) for _ in range(num_streams - 1)]
        logger.debug("Created D2H stream: primary=%r", primary_stream)

        if num_streams > 1:
            logger.info("Multi-stream D2H enabled: %d streams", num_streams)

        pinned_ptr = None
        host_registered_arr = None
        buffer = None
        pinned_memory_available = False

        try:
            pinned_ptr = cuda.malloc_host_alloc(nbytes, flags=0x01)
            buf = (ctypes.c_ubyte * nbytes).from_address(pinned_ptr.value)
            buffer = np.frombuffer(buf, dtype=fmt.numpy_dtype).reshape(fmt.shape)
            pinned_memory_available = True
            logger.debug("Allocated portable pinned numpy buffer: %s, %s", fmt.shape, fmt.dtype_str)
        except (RuntimeError, OSError) as e:
            if not allow_pageable:
                raise RuntimeError(
                    f"Pinned-memory allocation failed for {nbytes} bytes ({nbytes / 1_048_576:.1f} MB). "
                    f"Set allow_pageable_fallback=True in ImportPolicy to allow ~2x slower pageable fallback. "
                    f"Original error: {e}"
                ) from e
            logger.warning(
                "cudaMallocHost failed for %d bytes (%.1f MB); trying cudaHostRegister: %s",
                nbytes,
                nbytes / 1_048_576,
                e,
            )
            try:
                fallback_arr = np.empty(fmt.shape, dtype=fmt.numpy_dtype)
                cuda.host_register(fallback_arr.ctypes.data, fallback_arr.nbytes)
                host_registered_arr = fallback_arr
                buffer = fallback_arr
                pinned_memory_available = True
                logger.info("cudaHostRegister succeeded — using registered pinned memory")
            except (RuntimeError, OSError) as e2:
                logger.warning(
                    "cudaHostRegister also failed — falling back to pageable memory "
                    "(expect ~2x slower D2H bandwidth): %s",
                    e2,
                )
                buffer = np.empty(fmt.shape, dtype=fmt.numpy_dtype)
                pinned_memory_available = False

        chunk_plan: list[tuple[int, int]] = []
        if num_streams > 1:
            chunk = ((nbytes + num_streams - 1) // num_streams + 15) & ~15
            for i in range(num_streams):
                offset = i * chunk
                size = min(chunk, nbytes - offset)
                if size <= 0:
                    break
                chunk_plan.append((offset, size))

        # allocate a second pinned buffer using the same path that succeeded.
        back_buffer = None
        back_pinned_ptr = None
        back_host_registered_arr = None
        if pipelined:
            if pinned_ptr is not None:
                back_pinned_ptr = cuda.malloc_host_alloc(nbytes, flags=0x01)
                _back_mem = (ctypes.c_ubyte * nbytes).from_address(back_pinned_ptr.value)
                back_buffer = np.frombuffer(_back_mem, dtype=fmt.numpy_dtype).reshape(fmt.shape)
            elif host_registered_arr is not None:
                _back_arr = np.empty(fmt.shape, dtype=fmt.numpy_dtype)
                cuda.host_register(_back_arr.ctypes.data, _back_arr.nbytes)
                back_host_registered_arr = _back_arr
                back_buffer = _back_arr
            else:
                back_buffer = np.empty(fmt.shape, dtype=fmt.numpy_dtype)
            logger.debug("Allocated pipelined D2H back buffer: %s %s", fmt.shape, fmt.dtype_str)

        return cls(
            cuda=cuda,
            fmt=fmt,
            buffer=buffer,
            pinned_ptr=pinned_ptr,
            host_registered_arr=host_registered_arr,
            pinned_memory_available=pinned_memory_available,
            primary_stream=primary_stream,
            d2h_streams=d2h_streams,
            num_streams=num_streams,
            chunk_plan=chunk_plan,
            buffer_ptr=ctypes.c_void_p(buffer.ctypes.data) if buffer is not None else None,
            back_buffer=back_buffer,
            back_pinned_ptr=back_pinned_ptr,
            back_host_registered_arr=back_host_registered_arr,
            back_buffer_ptr=ctypes.c_void_p(back_buffer.ctypes.data) if back_buffer is not None else None,
            pipelined=pipelined,
            priming=pipelined,
        )

    def needs_rebuild(self, fmt: Format) -> bool:
        """True when the pre-allocated buffer doesn't match the new format."""
        return self.buffer.shape != fmt.shape or self.buffer.dtype != fmt.numpy_dtype

    def rotate(self) -> None:
        """Swap front/back buffers and their precomputed pointers (pipelined mode)."""
        self.buffer, self.back_buffer = self.back_buffer, self.buffer
        self.buffer_ptr, self.back_buffer_ptr = self.back_buffer_ptr, self.buffer_ptr

    def drain(self) -> None:
        """Block until every enqueued D2H copy completes. Best-effort, idempotent.

        Pipelined mode leaves one async copy in flight after each materialize();
        callers must drain before freeing the pinned destination (close) or
        closing the IPC source mapping (Importer._reinitialize) — otherwise the
        copy can still be running when its endpoint is released (CUDA 719-class
        source-lifetime race).
        """
        if self.primary_stream is not None:
            with contextlib.suppress(RuntimeError, OSError):
                self.cuda.stream_synchronize(self.primary_stream)

    def close(self) -> None:
        """Idempotent teardown: free pinned allocation, destroy streams."""
        self.drain()  # in-flight pipelined copy writes into the buffer freed below
        if self.pinned_ptr is not None:
            try:
                self.cuda.free_host(self.pinned_ptr)
                logger.debug("Freed pinned numpy buffer")
            except (RuntimeError, OSError) as e:
                logger.debug("free_host skipped (context gone): %s", e)
            self.pinned_ptr = None
            # Drop the numpy view that aliases the now-freed CUDA-pinned allocation.
            # ctypes docs: a ctypes buffer's Python owner must outlive every view into it.
            # Clearing here ensures post-close access fails loudly (AttributeError on None)
            # instead of silently touching freed GPU memory.
            self.buffer = None
            self.buffer_ptr = None  # precomputed pointer into freed allocation — clear it

        if self.host_registered_arr is not None:
            try:
                self.cuda.host_unregister(self.host_registered_arr.ctypes.data)
            except (RuntimeError, OSError) as e:
                logger.debug("host_unregister failed: %s", e)
            self.host_registered_arr = None

        for i, stream in enumerate(self.d2h_streams):
            if i == 0:
                continue  # primary_stream destroyed below
            if stream is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.destroy_stream(stream)
        self.d2h_streams.clear()

        if self.primary_stream is not None:
            try:
                self.cuda.destroy_stream(self.primary_stream)
                logger.debug("Destroyed D2H stream")
            except (RuntimeError, OSError) as e:
                logger.debug("D2H stream destroy skipped (context gone): %s", e)
            self.primary_stream = None

        # free pipelined back buffer.
        if self.back_pinned_ptr is not None:
            try:
                self.cuda.free_host(self.back_pinned_ptr)
                logger.debug("Freed pipelined back buffer")
            except (RuntimeError, OSError) as e:
                logger.debug("free_host(back) skipped: %s", e)
            self.back_pinned_ptr = None
            self.back_buffer = None
            self.back_buffer_ptr = None
        if self.back_host_registered_arr is not None:
            try:
                self.cuda.host_unregister(self.back_host_registered_arr.ctypes.data)
            except (RuntimeError, OSError) as e:
                logger.debug("host_unregister(back) failed: %s", e)
            self.back_host_registered_arr = None
            self.back_buffer = None
            self.back_buffer_ptr = None


# ---------------------------------------------------------------------------
# Reconnect state machine
# ---------------------------------------------------------------------------


@dataclass
class _RetryState:
    """Retry policy and transient counters for the connection-attempt loop.

    Mirrors RetryState in td_exporter/TDReceiver.py — keep fields in sync.
    """

    connect_attempts: int = 0
    max_connect_attempts: int = 20
    backoff_intervals: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 120)
    retry_interval_frames: int = 1
    frames_since_last_retry: int = 0

    def request_immediate_reconnect(self) -> None:
        """Force the next get_frame*() call to attempt reconnection without waiting."""
        self.frames_since_last_retry = self.retry_interval_frames


# ---------------------------------------------------------------------------
# IPC-event pointer helper
# ---------------------------------------------------------------------------


def _event_to_int(evt: object) -> int:
    """Resolve an opened IPC event to its integer cudaEvent_t pointer.

    Normally a CUDAEvent_t (ctypes c_uint64); be robust to a raw bytes handle
    (8-byte little-endian pointer) or a plain int, so the cupy backend matches
    the torch/CPU paths that hand the event straight to ctypes.
    """
    val = getattr(evt, "value", evt)
    if isinstance(val, (bytes, bytearray)):
        return int.from_bytes(val, "little")
    if isinstance(val, int):
        return val
    return int(val)  # type: ignore[arg-type]  # ctypes c_uint64.value is int at runtime


# ---------------------------------------------------------------------------
# Frame-consume backends (private, one instance per get_frame* call)
# ---------------------------------------------------------------------------


class _FrameBackend(Protocol):
    """Private per-call backend for Importer._consume_frame. One instance per call.

    Concrete implementations: _TorchBackend, _NumpyBackend, _CupyBackend.
    The interface is the test surface — each backend is independently verifiable.
    """

    nvtx_slot_names: tuple[str, ...]
    nvtx_color: str
    backend_name: str  # passed to _log_frame_stats

    def prepare(self, importer: Importer) -> None:
        """Pre-NVTX setup. Called before the outer NVTX range is pushed.

        NumpyBuffers lazy rebuild goes here. Torch and CuPy are no-ops.
        May mutate importer state (e.g. importer._numpy).
        """

    def wait(self, conn: IPCConnection, read_slot: int) -> float:
        """Wait for the producer event for *read_slot*.

        Returns microseconds waited (for debug telemetry via total_wait_event_time).
        CPU-spin backends raise TimeoutError on timeout.
        GPU-side backends (cupy) return 0.0 immediately — the stream waits on the GPU.
        """

    def materialize(self, conn: IPCConnection, read_slot: int) -> Any:
        """Return the frame after wait(). Called inside the outer NVTX range.

        May add inner NVTX sub-ranges (e.g. numpy adds a d2h_copy range).
        """


class _TorchBackend:
    """Zero-copy torch.Tensor backend. CPU-spin or GPU-side wait depending on stream."""

    nvtx_slot_names = _NVTX_GET_NAMES
    nvtx_color = "purple"
    backend_name = "torch"

    def __init__(self, importer: Importer, stream: object | None) -> None:
        self._imp = importer
        self._stream = stream
        # Cached raw CUDA stream handle for torch_gpu_wait path.  Populated lazily
        # on first call to avoid paying the torch.cuda.current_stream() Python→C++
        # crossing on every frame.  Valid as long as the caller uses a single default
        # stream per device (the common case).  Pass stream= to get_frame() explicitly
        # if you switch CUDA streams between calls.
        self._gpu_wait_cs: int | None = None

    def prepare(self, importer: Importer) -> None:  # noqa: ARG002
        pass

    def wait(self, conn: IPCConnection, read_slot: int) -> float:
        evt = conn.ipc_events[read_slot]
        if self._stream is not None:
            # Explicit stream= provided: GPU-side wait (existing behaviour).
            cs = self._imp._resolve_stream(self._stream)
            if evt:
                conn.cuda.stream_wait_event(cs, evt, 0)
            else:
                # No IPC event for this slot — fall back to full sync so the
                # GPU read is not unordered.
                self._imp._wait_for_slot(read_slot)
            return 0.0
        if self._imp._policy.torch_gpu_wait or self._imp._gpu_wait_active:
            # R1 opt-in (explicit) or R1-adaptive (auto-latched): GPU-side wait on torch.cuda.current_stream().
            # CPU returns immediately; ordering enforced by the GPU scheduler.
            # TIMEOUT is unreachable on this path (hung producer = stalled stream).
            # Stream handle cached after first call to skip repeated Python→C++ crossing.
            import torch

            if torch.cuda.is_available():
                if self._gpu_wait_cs is None:
                    self._gpu_wait_cs = torch.cuda.current_stream().cuda_stream
                cs = self._gpu_wait_cs
                if evt:
                    conn.cuda.stream_wait_event(cs, evt, 0)
                    return 0.0
                # No IPC event — fall through to CPU wait.
        return self._imp._wait_for_slot(read_slot)  # may raise TimeoutError

    def materialize(self, conn: IPCConnection, read_slot: int) -> Any:  # noqa: ARG002
        return self._imp._torch.tensors[read_slot]


class _NumpyBackend:
    """D2H-copy numpy ndarray backend. CPU-spin wait; lazy NumpyBuffers build."""

    nvtx_slot_names = _NVTX_NUMPY_NAMES
    nvtx_color = "orange"
    backend_name = "numpy"

    def __init__(self, importer: Importer) -> None:
        self._imp = importer

    def prepare(self, importer: Importer) -> None:
        fmt = importer._format
        if importer._numpy is None or importer._numpy.needs_rebuild(fmt):
            if importer._numpy is not None:
                importer._numpy.close()
            importer._numpy = NumpyBuffers.build(
                importer._conn,
                fmt,
                num_streams=importer._policy.d2h_num_streams,
                high_priority=importer._policy.d2h_stream_high_priority,
                allow_pageable=importer._policy.allow_pageable_fallback,
                pipelined=importer._policy.d2h_pipelined,
            )

    def wait(self, conn: IPCConnection, read_slot: int) -> float:
        return self._imp._wait_for_slot(read_slot)  # may raise TimeoutError

    def materialize(self, conn: IPCConnection, read_slot: int) -> Any:
        nb = self._imp._numpy
        fmt = self._imp._format
        nbytes = fmt.frame_nbytes
        with _nvtx.verbose_range("cudalink.importer.d2h_copy", self.nvtx_color):
            if nb.pipelined:
                if nb.priming:
                    # First call: enqueue copy into back buffer; no prior data to return.
                    conn.cuda.memcpy_async(
                        dst=nb.back_buffer_ptr,
                        src=conn.dev_ptrs[read_slot],
                        count=nbytes,
                        kind=2,
                        stream=nb.primary_stream,
                    )
                    nb.priming = False
                    # Rotate: back (just filled) becomes new front for next call.
                    nb.rotate()
                    conn.cuda.check_sticky_error("get_frame_numpy[pipelined-prime]")
                    return None  # get_frame_numpy surfaces as NO_FRAME
                # Steady state: sync the previous call's copy (now in nb.buffer = front),
                # enqueue this call's copy into nb.back_buffer, then return front.
                conn.cuda.stream_synchronize(nb.primary_stream)
                conn.cuda.memcpy_async(
                    dst=nb.back_buffer_ptr,
                    src=conn.dev_ptrs[read_slot],
                    count=nbytes,
                    kind=2,
                    stream=nb.primary_stream,
                )
                result = nb.buffer
                nb.rotate()
                conn.cuda.check_sticky_error("get_frame_numpy[pipelined]")
                return result
            n_streams = nb.num_streams
            if n_streams <= 1:
                conn.cuda.memcpy_async(
                    dst=nb.buffer_ptr,  # precomputed c_void_p; stable for buffer lifetime
                    src=conn.dev_ptrs[read_slot],
                    count=nbytes,
                    kind=2,  # cudaMemcpyDeviceToHost
                    stream=nb.primary_stream,
                )
                conn.cuda.stream_synchronize(nb.primary_stream)
            else:
                dst_base = nb.buffer.ctypes.data
                src_base = conn.dev_ptrs[read_slot].value
                for i, (offset, size) in enumerate(nb.chunk_plan):
                    conn.cuda.memcpy_async(
                        dst=ctypes.c_void_p(dst_base + offset),
                        src=ctypes.c_void_p(src_base + offset),
                        count=size,
                        kind=2,
                        stream=nb.d2h_streams[i],
                    )
                for stream in nb.d2h_streams[: len(nb.chunk_plan)]:
                    conn.cuda.stream_synchronize(stream)
        conn.cuda.check_sticky_error("get_frame_numpy")
        return nb.buffer


class _CupyBackend:
    """Zero-copy CuPy ndarray backend. GPU-side streamWaitEvent; TIMEOUT unreachable."""

    nvtx_slot_names = _NVTX_CUPY_NAMES
    nvtx_color = "green"
    backend_name = "cupy"

    def __init__(self, importer: Importer, stream: object | None) -> None:
        self._imp = importer
        self._stream = stream

    def prepare(self, importer: Importer) -> None:  # noqa: ARG002
        pass

    def wait(self, conn: IPCConnection, read_slot: int) -> float:
        evt = conn.ipc_events[read_slot]
        if not evt:
            # No IPC event for this slot — fall back to CPU wait so the GPU
            # read is not unordered.  TimeoutError may be raised.
            return self._imp._wait_for_slot(read_slot)
        stream = self._stream
        if stream is None:
            stream = cp.cuda.get_current_stream()
        elif not isinstance(stream, cp.cuda.Stream):
            cuda_stream_ptr = self._imp._resolve_stream(stream)
            stream = cp.cuda.ExternalStream(cuda_stream_ptr)
        cp.cuda.runtime.streamWaitEvent(stream.ptr, _event_to_int(evt), 0)
        return 0.0  # GPU-side wait — CPU returns immediately; TimeoutError unreachable

    def materialize(self, conn: IPCConnection, read_slot: int) -> Any:  # noqa: ARG002
        return self._imp._cupy.arrays[read_slot]


# ---------------------------------------------------------------------------
# Importer
# ---------------------------------------------------------------------------


class Importer:
    """Deep consumer-side importer for CUDA IPC GPU memory.

    Do not construct directly — use Importer.open().

    Responsibilities:
    - Read IPC handles from SharedMemory (once at open)
    - Open handles via cudaIpcOpenMemHandle (once per connect)
    - Per-frame: acquire the next updated ring slot, wait on its IPC event,
      and return the frame as a typed ImportResult.

    Performance:
    - Per-frame (torch/cupy): < 1 μs (zero-copy pointer return)
    - Per-frame (numpy): ~300 μs–5 ms (GPU→CPU D2H copy)
    """

    def __init__(
        self,
        spec: ImportSpec,
        policy: ImportPolicy,
        cuda: ImporterCudaPort,
    ) -> None:
        """Internal constructor. Use Importer.open() instead."""
        self._spec = spec
        self._policy = policy
        self._cuda = cuda

        # Connected state (populated by _connect; None until then)
        self._conn: IPCConnection | None = None
        self._format: Format | None = None
        self._torch: TorchBuffers | None = None
        self._cupy: CupyBuffers | None = None
        self._numpy: NumpyBuffers | None = None
        self._initialized = False

        # A1 fix (metadata-race): True when self._format came from _resolve_format()'s
        # hard-coded (512, 512, 4)/float32 fallback because SHM metadata wasn't written
        # yet at connect time. The first NEW_FRAME after connect guarantees real
        # metadata is present (see _try_acquire), so that's when we re-resolve and
        # clear this flag -- never per-frame once resolved.
        self._format_provisional = False

        # Cached backend instances — each holds only a back-ref to this Importer
        # plus (for torch/cupy) the per-call stream, refreshed in-place rather than
        # constructing a new object each frame.  Avoids one allocation per get_frame* call.
        self._numpy_backend: _NumpyBackend | None = None
        self._torch_backend: _TorchBackend | None = None
        self._cupy_backend: _CupyBackend | None = None

        # R5: resolved native wait backend (compiled _native_waiter extension,
        # built into this wheel on Windows; see _native_loader.py).
        # None until/unless _open_ipc_slots resolves one; re-resolved on every
        # (re)connect, including from _reinitialize(), so a producer restart
        # never leaves a stale backend/address behind. See _wait_for_slot().
        self._wait_backend: WaitBackend | None = None

        # Reconnect state machine (None when reconnect_enabled=False)
        self._retry: _RetryState | None = (
            _RetryState(
                max_connect_attempts=policy.reconnect_max_attempts,
                backoff_intervals=policy.reconnect_backoff_frames,
            )
            if policy.reconnect_enabled
            else None
        )

        # Frame tracking
        self._last_write_idx = 0
        self.frame_count = 0
        self.last_latency = 0.0

        # NVML telemetry (attached after open() via attach_nvml_observer())
        self._nvml_observer: NVMLObserver | None = None

        # Performance metrics (debug mode)
        self.total_wait_event_time = 0.0
        self.total_get_frame_time = 0.0
        self.total_shm_read_us = 0.0
        self.total_wait_spin_us = 0.0
        self.total_wait_sleep_us = 0.0
        self.wait_spin_hits = 0
        self.wait_sleep_hits = 0

        # R1-adaptive: one-way latch state
        # _gpu_wait_active flips True once and stays True for the session.
        # Precompute the minimum sleeps needed to trigger (ceil of window × pct / 100).
        self._gpu_wait_active = False
        self._gpu_wait_latched_frame: int | None = None
        self._adaptive_frames = 0
        self._adaptive_sleeps = 0
        _w = policy.gpu_wait_adaptive_window
        _p = policy.gpu_wait_adaptive_sleep_pct
        self._adaptive_min_sleeps: int = max(1, -(-_w * _p // 100))  # ceil without math.ceil

    @classmethod
    def open(
        cls,
        spec: ImportSpec,
        *,
        policy: ImportPolicy | None = None,
        cuda: ImporterCudaPort | None = None,
    ) -> Importer:
        """Open a CUDA IPC channel and return a connected Importer.

        Args:
            spec:   Channel geometry, SHM routing, and timeout.
            policy: Behavioural knobs (spin-wait, D2H streams, etc.).
                    Defaults to ImportPolicy.from_env().
            cuda:   CUDA Port adapter. Defaults to CTypesCUDAAdapter (production).
                    Pass FakeCUDAAdapter() in tests to avoid requiring a GPU.
        """
        if policy is None:
            policy = ImportPolicy.from_env()
        if cuda is None:
            from ._cuda_adapters import CTypesCUDAAdapter

            cuda = CTypesCUDAAdapter.for_device(device=spec.device)

        imp = cls(spec, policy, cuda)
        if policy.reconnect_enabled:
            try:
                imp._connect()
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as e:
                logger.info("Producer not yet available (%s) — will retry on get_frame*()", e)
        else:
            imp._connect()
        return imp

    @classmethod
    def from_connection(
        cls,
        spec: ImportSpec,
        policy: ImportPolicy,
        conn: IPCConnection,
        fmt: Format,
        *,
        cuda: ImporterCudaPort | None = None,
        torch: Any | None = None,
        cupy: Any | None = None,
        numpy: Any | None = None,
        last_write_idx: int = 0,
    ) -> Importer:
        """Wrap an already-open IPCConnection into a connected Importer.

        Intended for GPU-free tests (pass a FakeCUDAAdapter-backed IPCConnection via
        ``fakes.make_connected_importer``) and for advanced callers that open a
        connection out-of-band.  Production code uses ``Importer.open()`` instead.

        Args:
            spec:           Channel geometry, SHM routing, and timeout (must match conn).
            policy:         Behavioural knobs (spin-wait, D2H streams, etc.).
            conn:           A live IPCConnection (dev_ptrs already opened). The Importer
                            takes ownership — ``conn.close()`` is called by
                            ``Importer.close()``.
            fmt:            Format describing the frame geometry and dtype.
            cuda:           CUDA adapter used for operations *after* the connection (e.g.
                            ``_reinitialize``). Defaults to ``conn.cuda`` if not given,
                            which is always correct in production. Tests may pass
                            ``FakeCUDAAdapter()`` explicitly so that ``_reinitialize``
                            uses a proper fake rather than the MagicMock stored on the
                            connection.
            torch:          Pre-built TorchBuffers, or None (skips torch frame returns).
            cupy:           Pre-built CupyBuffers, or None (skips cupy frame returns).
            numpy:          Pre-built NumpyBuffers, or None (built lazily on first
                            ``get_frame_numpy()`` call).
            last_write_idx: The write-index this Importer should treat as already-consumed
                            at connect time (default 0). Pass a non-zero value to simulate
                            an Importer that has already seen some frames — useful in tests
                            that verify NO_FRAME / new-frame edge cases without private
                            attribute injection.
        """
        imp = cls(spec, policy, conn.cuda if cuda is None else cuda)
        imp._adopt_connection(conn, fmt, torch=torch, cupy=cupy, numpy=numpy, last_write_idx=last_write_idx)
        return imp

    # ------------------------------------------------------------------
    # Connection internals
    # ------------------------------------------------------------------

    def _adopt_connection(
        self,
        conn: IPCConnection,
        fmt: Format,
        *,
        torch: Any | None = None,
        cupy: Any | None = None,
        numpy: Any | None = None,
        last_write_idx: int = 0,
        format_provisional: bool = False,
    ) -> None:
        """Wire an already-open IPCConnection into this Importer's connected state.

        Single authoritative definition of 'entered connected state'. Called by
        ``_connect()`` (normal production path) and ``from_connection()`` (advanced /
        test path). Callers build TorchBuffers / CupyBuffers and pass them in; numpy
        stays None by default and is built lazily on the first ``get_frame_numpy()``
        call.

        Args:
            last_write_idx: Initial value of ``_last_write_idx`` (default 0, meaning
                            "no frames consumed yet"). The production path always uses 0;
                            ``from_connection`` forwards this to support tests that need
                            an importer that has already consumed some frames.
            format_provisional: Forwarded from ``_resolve_format()`` (A1 fix) — True
                            when ``fmt`` is the hard-coded fallback, not SHM-derived.
                            ``from_connection`` (test path) never sets this — a
                            test-supplied fmt is always treated as resolved.
        """
        self._conn = conn
        self._format = fmt
        self._torch = torch
        self._cupy = cupy
        self._numpy = numpy
        self._last_write_idx = last_write_idx
        self._format_provisional = format_provisional
        self._initialized = True

    def _connect(self) -> None:
        """Open SHM, read IPC handles, build buffer views. Called once by open()."""
        shm, num_slots, ipc_version = self._open_and_validate_shm()
        try:
            fmt, provisional = self._resolve_format(shm, num_slots)
            conn = self._open_ipc_slots(shm, num_slots, ipc_version, fmt)
        except BaseException:
            # shm is still ours here — _open_ipc_slots only rolls back the
            # per-slot resources it opened, not the SharedMemory it was handed.
            shm.close()
            raise
        self._adopt_connection(
            conn,
            fmt,
            torch=TorchBuffers.build(conn, fmt) if TORCH_AVAILABLE else None,
            cupy=CupyBuffers.build(conn, fmt) if CUPY_AVAILABLE else None,
            format_provisional=provisional,
        )
        logger.info("Importer ready — device %d, shm=%r", self._spec.device, self._spec.shm_name)

    def _open_and_validate_shm(self) -> tuple[SharedMemory, int, int]:
        """Open SharedMemory and validate protocol magic, version, num_slots, shutdown."""
        try:
            shm = SharedMemory(name=self._spec.shm_name)
        except FileNotFoundError:
            logger.error("SharedMemory %r not found — producer must be running first", self._spec.shm_name)
            raise

        logger.info("Opened SharedMemory: %s", self._spec.shm_name)

        try:
            magic = struct.unpack("<I", bytes(shm.buf[MAGIC_OFFSET : MAGIC_OFFSET + MAGIC_SIZE]))[0]
        except (struct.error, ValueError, IndexError):
            shm.close()
            raise

        if magic != PROTOCOL_MAGIC:
            shm.close()
            raise ProtocolError(
                f"Protocol magic mismatch: expected 0x{PROTOCOL_MAGIC:08X}, got 0x{magic:08X}. "
                "Update both TD and Python sides to the same protocol version."
            )

        ipc_version = read_version(shm.buf)
        num_slots = read_num_slots(shm.buf)

        if num_slots == 0 or num_slots > 10:
            shm.close()
            raise ProtocolError(f"Invalid num_slots={num_slots} in SHM (expected 1–10)")

        shutdown_offset = SHMLayout(num_slots).shutdown_offset
        try:
            shutdown_flag = shm.buf[shutdown_offset]
        except (OSError, BufferError, IndexError) as e:
            shm.close()
            raise CudaIpcError(f"Could not read shutdown flag: {e}") from e

        if shutdown_flag == 1:
            shm.close()
            raise ProtocolError("Producer shutdown flag set — SharedMemory is stale")

        logger.info("Ring buffer with %d slots (v%d)", num_slots, ipc_version)
        return shm, num_slots, ipc_version

    def _resolve_format(self, shm: SharedMemory, num_slots: int) -> tuple[Format, bool]:
        """Determine frame geometry from spec overrides + SHM metadata.

        Returns:
            (format, provisional). provisional=True means the sender hadn't written
            real metadata yet (SHM metadata block was all-zero) at connect time, so
            a hard-coded (512, 512, 4)/float32 fallback was used (A1 fix — this used
            to be indistinguishable from a genuine detection; _try_acquire()'s first
            NEW_FRAME now re-resolves via this same method when provisional is True).
        """
        spec = self._spec
        if spec.shape is None or spec.dtype is None:
            fmt_from_shm = Format.from_shm(shm.buf, num_slots)
            if fmt_from_shm is not None:
                shape = spec.shape if spec.shape is not None else fmt_from_shm.shape
                dtype_str = spec.dtype if spec.dtype is not None else fmt_from_shm.dtype_str
                if shape != fmt_from_shm.shape or dtype_str != fmt_from_shm.dtype_str:
                    fmt = Format.from_overrides(shape, dtype_str)
                else:
                    fmt = fmt_from_shm
                if spec.shape is None:
                    logger.info("Auto-detected shape: %s", fmt.shape)
                if spec.dtype is None:
                    logger.info("Auto-detected dtype: %s", fmt.dtype_str)
                return fmt, False
            else:
                logger.warning("Could not detect metadata — using fallback (512, 512, 4) / float32")
                shape = spec.shape or (512, 512, 4)
                dtype_str = spec.dtype or "float32"
                fmt = Format.from_overrides(shape, dtype_str)
                return fmt, True
        else:
            fmt = Format.from_overrides(spec.shape, spec.dtype)
            return fmt, False

    def _open_ipc_slots(
        self,
        shm: SharedMemory,
        num_slots: int,
        ipc_version: int,
        fmt: Format,
    ) -> IPCConnection:
        """Open all IPC mem + event handles; return a live IPCConnection.

        Caller (``_connect`` / ``_reinitialize``) owns ``shm`` — on failure this
        method rolls back only the resources it opened itself (dev_ptrs, ipc_events,
        doorbell), so a partial slot-open never leaks a GPU IPC mapping.
        """
        from .cuda_runtime_types import cudaIpcEventHandle_t, cudaIpcMemHandle_t

        cuda = self._cuda
        ipc_handles: list = [None] * num_slots
        dev_ptrs: list = [None] * num_slots
        ipc_events: list = [None] * num_slots
        layout = SHMLayout(num_slots)
        db_handle = None

        try:
            for slot in range(num_slots):
                mem_off = layout.mem_handle_offset(slot)
                evt_off = layout.event_handle_offset(slot)

                mem_handle_bytes = bytes(shm.buf[mem_off : mem_off + IPC_HANDLE_SIZE])
                logger.debug("Slot %d IPC read handle prefix: %s...", slot, mem_handle_bytes[:16].hex())
                ipc_handles[slot] = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)
                dev_ptrs[slot] = cuda.ipc_open_mem_handle(ipc_handles[slot], flags=IPC_MEM_LAZY_ENABLE_PEER_ACCESS)

                event_handle_bytes = bytes(shm.buf[evt_off : evt_off + IPC_HANDLE_SIZE])
                if any(event_handle_bytes):
                    try:
                        ipc_event_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                        ipc_events[slot] = cuda.ipc_open_event_handle(ipc_event_handle)
                    except (RuntimeError, OSError) as e:
                        logger.debug("Failed to open IPC event for slot %d: %s", slot, e)

                logger.info(
                    "Slot %d: GPU ptr=0x%016x, event=%s",
                    slot,
                    dev_ptrs[slot].value,
                    "YES" if ipc_events[slot] else "NO",
                )

            logger.info("Opened %d IPC buffer slots", num_slots)

            # R5: the native wait backend needs the doorbell to reach its <10us target
            # (its block phase has nothing faster to wait on otherwise) — so wanting
            # native forces the doorbell open even if policy.doorbell is False.
            want_native = self._policy.wait_backend in ("auto", "native") and os.name == "nt"

            # R2: open the doorbell event if the policy (or R5 above) requests it
            if self._policy.doorbell or want_native:
                db_handle = _doorbell.open_doorbell(_doorbell.doorbell_event_name(self._spec.shm_name))
                if db_handle is not None:
                    logger.info("Doorbell opened: %r", _doorbell.doorbell_event_name(self._spec.shm_name))
                elif self._policy.doorbell:
                    logger.warning(
                        "Doorbell event not found (producer not ready or not Windows); falling back to poll-sleep"
                    )

            # R5: resolve the native wait backend now, while db_handle is known — a
            # backend without a doorbell can't beat the Python fallback, so only
            # resolve when one was actually opened above.
            self._wait_backend = self._resolve_wait_backend() if (want_native and db_handle is not None) else None
            if self._wait_backend is None and want_native and not self._policy.doorbell and db_handle is not None:
                # Doorbell was opened speculatively for native only (policy.doorbell
                # was False) and native failed to resolve — don't leave a doorbell
                # handle open that nothing will wait on.
                _doorbell.close(db_handle)
                db_handle = None
        except BaseException:
            # Roll back whatever this call opened so far — shm itself is not ours
            # to close (the caller owns it and may retry against the same mapping).
            IPCConnection(
                cuda=cuda,
                shm_handle=None,
                ipc_version=ipc_version,
                num_slots=num_slots,
                ipc_handles=ipc_handles,
                dev_ptrs=dev_ptrs,
                ipc_events=ipc_events,
                layout=layout,
                shutdown_offset=layout.shutdown_offset,
                timestamp_offset=layout.timestamp_offset,
                doorbell_handle=db_handle,
            ).close_ipc_handles()
            raise

        return IPCConnection(
            cuda=cuda,
            shm_handle=shm,
            ipc_version=ipc_version,
            num_slots=num_slots,
            ipc_handles=ipc_handles,
            dev_ptrs=dev_ptrs,
            ipc_events=ipc_events,
            layout=layout,
            shutdown_offset=layout.shutdown_offset,
            timestamp_offset=layout.timestamp_offset,
            doorbell_handle=db_handle,
        )

    def _resolve_wait_backend(self) -> WaitBackend | None:
        """Best-effort resolve the native wait backend (cuda_link._native_waiter).

        Returns None on any failure (extension not built for this platform,
        this CUDA adapter can't supply a cudaEventQuery pointer, or the
        extension rejects it) — callers must treat None as "use the existing
        Python spin/sleep path", never as an error.

        The resolved function-pointer address is obtained from *this
        connection's own* CUDA adapter (`self._cuda`) and handed to the
        native module explicitly, rather than letting the native module
        rediscover cudart independently — a bare-name GetModuleHandle lookup
        there is not guaranteed to find the same cudart instance this
        process is actually using when more than one same-named cudart DLL
        is loaded from different directories. Diagnosed 2026-07-04 (PLAN-002
        R5): an earlier version of this seam resolved the wrong instance and
        silently misreported a genuinely-pending CUDA event as complete.
        """
        try:
            from cuda_link._native_loader import load_native_backend  # noqa: PLC0415
        except ImportError:
            logger.info("cuda_link._native_waiter not built for this platform; using Python wait path")
            return None
        try:
            fn_ptr = self._cuda.cudart_event_query_fn_ptr()
        except AttributeError:
            # e.g. FakeCUDAAdapter in tests — never expected in practice since
            # ImportPolicy.for_testing() forces wait_backend="python", but stay
            # defensive rather than raise.
            logger.info("CUDA adapter has no cudart_event_query_fn_ptr(); using Python wait path")
            return None
        try:
            backend = load_native_backend(fn_ptr)
        except RuntimeError as e:
            logger.info("Native wait backend unavailable (%s); using Python wait path", e)
            return None
        logger.info("Native wait backend engaged")
        return backend

    # ------------------------------------------------------------------
    # Reconnect helpers
    # ------------------------------------------------------------------

    def _connect_silent(self) -> bool:
        """Attempt _connect(); return True on success, False on any connection failure."""
        try:
            self._connect()
            return True
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as e:
            logger.debug("Connect attempt failed: %s", e)
            return False

    def _drive_retry(self) -> bool:
        """Advance the reconnect state machine by one frame. Returns True if just connected."""
        retry = self._retry
        retry.frames_since_last_retry += 1
        if retry.frames_since_last_retry < retry.retry_interval_frames:
            return False

        retry.frames_since_last_retry = 0
        retry.connect_attempts += 1

        if self._connect_silent():
            return True

        backoff_idx = min(retry.connect_attempts, len(retry.backoff_intervals) - 1)
        retry.retry_interval_frames = retry.backoff_intervals[backoff_idx]
        if retry.connect_attempts <= retry.max_connect_attempts:
            logger.info(
                "Waiting for producer... (attempt %d, next retry in %d frames)",
                retry.connect_attempts,
                retry.retry_interval_frames,
            )
        elif retry.connect_attempts == retry.max_connect_attempts + 1:
            logger.warning("Producer not found. Will keep retrying silently.")
        return False

    def _partial_cleanup_for_reconnect(self) -> None:
        """Release IPC handles and reset connection state; keep Importer alive for retry."""
        self._release_connection_state()
        if self._retry is None:
            self._retry = _RetryState(
                max_connect_attempts=self._policy.reconnect_max_attempts,
                backoff_intervals=self._policy.reconnect_backoff_frames,
            )
        self._retry.request_immediate_reconnect()
        logger.info("Partial cleanup done — waiting for producer restart")

    def request_immediate_reconnect(self) -> None:
        """Force the next get_frame*() call to attempt reconnection without waiting."""
        if self._retry is not None:
            self._retry.request_immediate_reconnect()

    # ------------------------------------------------------------------
    # Slot acquisition
    # ------------------------------------------------------------------

    def _try_acquire(self) -> tuple[AcquireResult | None, ImportOutcome]:
        """Try to acquire the next updated ring slot.

        Returns:
            (AcquireResult, NEW_FRAME) when a new slot is ready.
            (None, outcome)            for all other states.
        """
        try:
            result = acquire_slot(
                self._conn.shm_handle.buf,
                self._conn.layout,
                self._last_write_idx,
                self._conn.ipc_version,
            )
        except (OSError, BufferError) as e:
            logger.debug("SHM buffer inaccessible: %s", e)
            return None, ImportOutcome.NO_FRAME

        if result.state is SlotState.SHUTDOWN:
            logger.info("Producer shutdown detected")
            if self._policy.reconnect_enabled:
                self._partial_cleanup_for_reconnect()
            else:
                self.close()
            return None, ImportOutcome.SHUTDOWN

        if result.state is SlotState.VERSION_CHANGED:
            logger.debug(
                "Producer re-initialized (v%d → v%d), reopening IPC handles",
                self._conn.ipc_version,
                result.new_version,
            )
            self._reinitialize()
            return None, ImportOutcome.RECONNECTING

        if result.state is SlotState.NO_FRAME:
            return None, ImportOutcome.NO_FRAME

        self._last_write_idx = result.write_idx

        # A1 fix: the first NEW_FRAME after a connect that had to guess geometry
        # (_format_provisional, see _resolve_format) guarantees the producer has by
        # now written real metadata under the SAME version — a case _reinitialize()
        # never sees, since it only fires on VERSION_CHANGED. Re-resolve once here;
        # no cost on subsequent frames once the flag is cleared.
        if self._format_provisional:
            resolved_fmt = Format.from_shm(self._conn.shm_handle.buf, self._conn.num_slots)
            if resolved_fmt is not None:
                self._apply_format_change(resolved_fmt)
                self._format_provisional = False
            # else: metadata still not visible (shouldn't happen once NEW_FRAME is
            # reported) — keep the flag set and retry on the next NEW_FRAME.

        return result, ImportOutcome.NEW_FRAME

    def _adaptive_observe_wait(self, slept: bool) -> None:
        """Record one cpu-spin/sleep outcome and latch gpu-wait when the sleep rate crosses the threshold.

        Called from _wait_for_slot only — a no-op once latched or when the feature is disabled.
        The latch is one-way: once _gpu_wait_active is True it stays True for the session.
        """
        if not self._policy.torch_gpu_wait_adaptive or self._gpu_wait_active:
            return
        self._adaptive_frames += 1
        if slept:
            self._adaptive_sleeps += 1
        if self._adaptive_sleeps >= self._adaptive_min_sleeps:
            self._gpu_wait_active = True
            self._gpu_wait_latched_frame = self.frame_count
        elif self._adaptive_frames >= self._policy.gpu_wait_adaptive_window:
            # Full window passed without hitting the threshold — reset and keep watching.
            self._adaptive_frames = 0
            self._adaptive_sleeps = 0

    def _wait_for_slot(self, slot: int) -> float:
        """CPU-side wait until producer signals the slot event.

        Returns wait time in microseconds. Raises TimeoutError on timeout.
        """
        conn = self._conn
        policy = self._policy
        wait_start = time.perf_counter()
        deadline = wait_start + self._spec.timeout_ms / 1000

        # R5: native wait backend — only engaged when resolved AND this slot has
        # both an IPC event and an open doorbell (the backend needs both to reach
        # its <10us target). Everything below this block is the pre-R5 path,
        # unchanged, so CUDALINK_WAIT_BACKEND=python (or any resolution failure)
        # reproduces today's behavior exactly.
        if self._wait_backend is not None and conn.ipc_events[slot] and conn.doorbell_handle is not None:
            event_ptr = int(conn.ipc_events[slot].value)
            db_handle = int(conn.doorbell_handle)
            # Recomputed fresh every call, never cached: a producer restart tears
            # down and remaps the SHM segment at a new address (_reinitialize),
            # and _wait_backend/doorbell are re-resolved at that same point, but
            # the address itself must still be read live here to be correct even
            # within a single connection's lifetime (the buffer object is stable
            # per-connection, but computing it once and reusing it buys nothing
            # and adds a class of staleness bug for free).
            write_idx_addr = ctypes.addressof(ctypes.c_char.from_buffer(conn.shm_handle.buf)) + WRITE_IDX_OFFSET
            result = self._wait_backend.wait_slot(
                event_ptr,
                db_handle,
                write_idx_addr,
                self._last_write_idx,
                policy.wait_spin_us,
                int(self._spec.timeout_ms),
            )
            status_name = result.status.name  # avoid importing WaitStatus in core
            if status_name == "TIMEOUT":
                raise ProducerTimeoutError(f"IPC event wait timed out after {self._spec.timeout_ms}ms (slot={slot})")
            if status_name == "READY_SPIN":
                self.total_wait_spin_us += result.waited_us
                self.wait_spin_hits += 1
                self._adaptive_observe_wait(False)
            else:  # READY_DOORBELL / READY_POLL / READY_LATE(retired) all count as the blocked/slept bucket
                self.total_wait_sleep_us += result.waited_us
                self.wait_sleep_hits += 1
                self._adaptive_observe_wait(status_name == "READY_DOORBELL")
            return result.waited_us

        if conn.ipc_events[slot]:
            evt = conn.ipc_events[slot]
            query = conn.cuda.query_event

            if policy.wait_spin_us > 0:
                # Clamp spin_deadline to the overall deadline so the loop needs
                # only one perf_counter() call per iteration — no inner deadline
                # check required (spin budget << timeout_ms in all practical cases).
                spin_deadline = min(
                    wait_start + policy.wait_spin_us / 1_000_000,
                    deadline,
                )
                while (now := time.perf_counter()) < spin_deadline:
                    if query(evt):
                        spin_us = (now - wait_start) * 1_000_000
                        self.total_wait_spin_us += spin_us
                        self.wait_spin_hits += 1
                        self._adaptive_observe_wait(False)
                        return spin_us
                # If we exited because spin_deadline == deadline, we timed out.
                if time.perf_counter() >= deadline:
                    raise ProducerTimeoutError(
                        f"IPC event wait timed out after {self._spec.timeout_ms}ms (slot={slot})"
                    )

            phase2_start = time.perf_counter()
            did_sleep = False
            with _HighResTimer():
                while True:
                    if query(evt):
                        break
                    if time.perf_counter() >= deadline:
                        raise ProducerTimeoutError(
                            f"IPC event wait timed out after {self._spec.timeout_ms}ms (slot={slot})"
                        )
                    time.sleep(0.0001)
                    did_sleep = True
            self.total_wait_sleep_us += (time.perf_counter() - phase2_start) * 1_000_000
            self.wait_sleep_hits += 1
            self._adaptive_observe_wait(did_sleep)
        else:
            conn.cuda.synchronize()

        return (time.perf_counter() - wait_start) * 1_000_000

    # ------------------------------------------------------------------
    # R2: Doorbell wait primitive
    # ------------------------------------------------------------------

    def wait_for_doorbell(self, timeout_ms: float) -> bool:
        """Block until the producer signals a new frame or timeout_ms elapses.

        Lost-wakeup-safe: before blocking the function checks whether a new
        frame has already arrived (write_idx advanced) and returns True
        immediately if so, consuming no signal from the kernel event.

        Returns:
            True  — a new frame *may* be waiting; caller should call get_frame().
            False — timed out, or doorbell is disabled / not opened / non-Windows.
                    Caller should fall back to its poll-sleep path.

        Notes:
            * This is a single-consumer primitive (auto-reset event).
            * The 2 ms default cap in the example loop keeps the lost-wakeup
              window bounded: worst-case latency is one timeout_ms slice, not
              infinite.  Caller always re-checks with get_frame() after True.
            * Returns False immediately when CUDALINK_DOORBELL=0 or on
              non-Windows — existing behaviour is preserved without any guards
              in the caller.
        """
        conn = self._conn
        h = conn.doorbell_handle if conn is not None else None
        if h is None:
            return False  # disabled / not opened / non-Windows → caller polls

        # Early return: frame already arrived since last get_frame() call.
        cur = read_write_idx(conn.shm_handle.buf)
        if cur != 0 and cur != self._last_write_idx:
            return True

        # Block until signaled; auto-reset clears the event on wake.
        return _doorbell.wait(h, int(timeout_ms))

    # ------------------------------------------------------------------
    # Frame consumers
    # ------------------------------------------------------------------

    def _consume_frame(self, backend: _FrameBackend) -> ImportResult:
        """Shared frame-consume core used by all get_frame* methods.

        Owns: _begin_frame preamble, backend.prepare(), outer NVTX push/pop,
        event_wait verbose_range, TimeoutError handling, frame_count increment,
        and debug telemetry. Backend owns: pre-NVTX buffer setup (prepare),
        event wait (wait), and frame materialisation (materialize).
        """
        early, read_slot, latency_ms, debug, frame_start = self._begin_frame()
        if early is not None:
            return early
        self.last_latency = latency_ms
        conn = self._conn

        backend.prepare(self)

        _nvtx.push_range(backend.nvtx_slot_names[read_slot], backend.nvtx_color)
        with _nvtx.verbose_range("cudalink.importer.event_wait", backend.nvtx_color):
            try:
                wait_us = backend.wait(conn, read_slot)
                if debug:
                    self.total_wait_event_time += wait_us
            except TimeoutError:
                logger.error("Producer timeout — slot %d", read_slot)
                _nvtx.pop_range()
                return ImportResult(outcome=ImportOutcome.TIMEOUT)

        frame = backend.materialize(conn, read_slot)

        self.frame_count += 1
        if debug:
            frame_time = (time.perf_counter() - frame_start) * 1_000_000
            self.total_get_frame_time += frame_time
            self._log_frame_stats(backend.backend_name, read_slot, conn)

        _nvtx.pop_range()
        return ImportResult(outcome=ImportOutcome.NEW_FRAME, frame=frame)

    def _begin_frame(self) -> tuple:
        """Common preamble for all get_frame* methods.

        Returns ``(early_result, read_slot, latency_ms, debug, frame_start)``.
        When ``early_result`` is not None the caller must return it immediately.
        All three paths — torch, numpy, cupy — open with::

            early, read_slot, latency_ms, debug, frame_start = self._begin_frame()
            if early is not None:
                return early
            self.last_latency = latency_ms
        """
        if not self._initialized and (self._retry is None or not self._drive_retry()):
            return ImportResult(outcome=ImportOutcome.RECONNECTING), -1, 0.0, False, 0.0

        debug = self._policy.debug
        frame_start = time.perf_counter() if debug else 0.0

        slot_result, outcome = self._try_acquire()
        if outcome is not ImportOutcome.NEW_FRAME:
            return ImportResult(outcome=outcome), -1, 0.0, debug, frame_start

        read_slot = slot_result.slot
        producer_timestamp = slot_result.timestamp
        latency_ms = (time.perf_counter() - producer_timestamp) * 1000 if producer_timestamp > 0 else 0.0
        return None, read_slot, latency_ms, debug, frame_start

    def get_frame(self, stream: object | None = None) -> ImportResult:
        """Get current frame as a zero-copy torch.Tensor on GPU.

        Args:
            stream: Optional CUDA stream (torch.cuda.Stream, cupy.cuda.Stream,
                    or int). When provided, issues cudaStreamWaitEvent (GPU-side
                    ordering, non-blocking CPU). When None, the wait mode depends
                    on ImportPolicy.torch_gpu_wait:

                    False (default): CPU spins on cudaEventQuery then sleeps.
                        get_frame() blocks until the producer event fires.
                        ImportOutcome.TIMEOUT is raised after timeout_ms.

                    True (opt-in, CUDALINK_TORCH_GPU_WAIT=1): cudaStreamWaitEvent
                        on torch.cuda.current_stream(). CPU returns immediately;
                        the tensor is valid in **stream order**, not at return.
                        ImportOutcome.TIMEOUT is unreachable on this path —
                        a hung producer stalls the stream instead of raising.
                        See get_frame_cupy() for the same trade-off.

        Returns:
            ImportResult[torch.Tensor] with outcome NEW_FRAME, NO_FRAME,
            SHUTDOWN, RECONNECTING, or TIMEOUT.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for get_frame(). Use get_frame_numpy() instead.")
        if self._torch_backend is None:
            self._torch_backend = _TorchBackend(self, stream)
        else:
            self._torch_backend._stream = stream
        return self._consume_frame(self._torch_backend)

    def get_frame_numpy(self) -> ImportResult:
        """Get current frame as a numpy ndarray (CPU; involves D2H copy).

        Returns:
            ImportResult[np.ndarray] with outcome NEW_FRAME, NO_FRAME,
            SHUTDOWN, RECONNECTING, or TIMEOUT.

        When ImportPolicy.d2h_pipelined is True the first call returns NO_FRAME
        (priming the back buffer) and each subsequent call returns the previous
        frame with +1 frame latency, hiding D2H copy time behind consumer CPU work.
        A producer restart re-primes: the first call after RECONNECTING returns
        NO_FRAME again rather than the previous session's last frame.
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for get_frame_numpy()")
        if self._numpy_backend is None:
            self._numpy_backend = _NumpyBackend(self)
        result = self._consume_frame(self._numpy_backend)
        # pipelined path signals the priming call with frame=None; surface as NO_FRAME.
        if self._policy.d2h_pipelined and result.outcome is ImportOutcome.NEW_FRAME and result.frame is None:
            return ImportResult(outcome=ImportOutcome.NO_FRAME)
        return result

    def get_frame_cupy(self, stream: object | None = None) -> ImportResult:
        """Get current frame as a zero-copy CuPy ndarray on GPU.

        Args:
            stream: Optional CuPy/torch stream or int. When provided, issues
                    cudaStreamWaitEvent. When None, uses CuPy's current stream.

        Returns:
            ImportResult[cp.ndarray] with outcome NEW_FRAME, NO_FRAME,
            SHUTDOWN, or RECONNECTING.

        Note:
            TIMEOUT is not reachable via this path — streamWaitEvent is a
            non-blocking CPU call; the stream waits on the GPU side. See
            _CupyBackend.wait() for details. Use get_frame() or
            get_frame_numpy() if producer-timeout detection is required.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("cupy is required for get_frame_cupy(). Install: pip install cupy-cuda12x")
        if self._cupy_backend is None:
            self._cupy_backend = _CupyBackend(self, stream)
        else:
            self._cupy_backend._stream = stream
        return self._consume_frame(self._cupy_backend)

    # ------------------------------------------------------------------
    # Re-initialization (producer restarted with new IPC handles)
    # ------------------------------------------------------------------

    def _apply_format_change(self, new_fmt: Format) -> None:
        """Update self._format and rebuild any already-materialized buffer views for it.

        Shared tail for the two places a resolved geometry can supersede
        self._format: ``_reinitialize()`` (after a full IPC handle reopen, producer
        restart) and ``_try_acquire()``'s first-NEW_FRAME provisional re-resolve (A1
        fix — same IPC memory, only the metadata was unavailable at connect time).
        Callers differ in whether IPC handles were reopened; rebuilding torch/numpy
        views for the new geometry is identical either way, so it lives here once.

        Compares via layout_differs_from(), not Format.__eq__ — __eq__ would
        false-positive when self._format is override-derived (kind=bits=flags=0
        sentinels) against an SHM-derived new_fmt with real non-zero values.
        """
        if new_fmt.layout_differs_from(self._format):
            logger.info(
                "Format changed: %s %s → %s %s",
                self._format.shape,
                self._format.dtype_str,
                new_fmt.shape,
                new_fmt.dtype_str,
            )
            if self._numpy is not None:
                self._numpy.close()
                self._numpy = None

        self._format = new_fmt
        if self._torch is not None:
            self._torch = TorchBuffers.build(self._conn, new_fmt)

    def _reinitialize(self) -> None:
        """Reopen all IPC handles after producer restart. Internal; callers see RECONNECTING."""
        old_conn = self._conn
        shm = old_conn.shm_handle

        if self._numpy is not None and self._numpy.pipelined:
            # The previous get_frame_numpy() left an async D2H copy in flight whose
            # source is an IPC mapping closed just below — drain before releasing it.
            self._numpy.drain()
            # Re-prime so the first call of the new session returns NO_FRAME instead
            # of surfacing the dead session's last frame as NEW_FRAME.
            self._numpy.priming = True

        old_conn.close_ipc_handles()

        # self._conn still points at old_conn here (handles closed, shm still open)
        # until _open_ipc_slots succeeds below — on failure, fall through to the
        # same partial-cleanup path used for producer shutdown rather than leaving
        # self._conn referencing a connection whose dev_ptrs/events are all None.
        try:
            new_ipc_version = read_version(shm.buf)
            new_num_slots = read_num_slots(shm.buf)

            # Route through Format.from_shm — same decoder used at connect time.
            # Falls back to current format if SHM metadata is absent or all-zeros.
            resolved_fmt = Format.from_shm(shm.buf, new_num_slots)
            new_fmt = resolved_fmt or self._format

            new_conn = self._open_ipc_slots(shm, new_num_slots, new_ipc_version, new_fmt)
            self._conn = new_conn

            self._apply_format_change(new_fmt)
            if resolved_fmt is not None:
                # A1: a genuine SHM read (not the "still absent, kept old format" fallback
                # above) means geometry is now authoritative — clear any stale provisional
                # flag from the original connect so _try_acquire() doesn't re-resolve again.
                self._format_provisional = False

            logger.debug("Reopened %d IPC handles v%d", new_num_slots, new_ipc_version)
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as e:
            logger.info("Reinitialize failed (%s) — waiting for producer restart", e)
            self._partial_cleanup_for_reconnect()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _release_connection_state(self) -> None:
        """Release IPC/numpy resources and reset per-connection state. Idempotent."""
        if getattr(self, "_numpy", None) is not None:
            self._numpy.close()
            self._numpy = None
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
        self._torch = None
        self._cupy = None
        self._format = None
        self._initialized = False

    def close(self) -> None:
        """Release all CUDA and SHM resources. Idempotent."""
        self._release_connection_state()
        logger.info("Importer closed")

    def __del__(self) -> None:
        if getattr(self, "_initialized", False):
            self.close()

    def __enter__(self) -> Importer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_stream(self, stream: object) -> int:
        """Extract raw CUDA stream pointer from torch/cupy stream or int."""
        if isinstance(stream, int):
            return stream
        if TORCH_AVAILABLE and hasattr(stream, "cuda_stream"):
            return stream.cuda_stream
        if hasattr(stream, "ptr"):
            return stream.ptr
        raise TypeError(
            f"Unsupported stream type: {type(stream)}. Expected torch.cuda.Stream, cupy.cuda.Stream, or int."
        )

    def _log_frame_stats(self, mode: str, slot: int, conn: IPCConnection) -> None:
        n = self.frame_count
        if n > 0 and n % 97 == 0:
            sync_mode = "GPU-Events" if all(conn.ipc_events) else "CPU-Sync"
            spin_hit_pct = 100.0 * self.wait_spin_hits / n if n > 0 else 0.0
            logger.debug(
                "Frame %d [%s/%s]: wait=%.1fus total=%.1fus latency=%.2fms "
                "spin_hit=%.0f%% avg_spin=%.1fus avg_sleep=%.1fus",
                n,
                mode,
                sync_mode,
                self.total_wait_event_time / n,
                self.total_get_frame_time / n,
                self.last_latency,
                spin_hit_pct,
                self.total_wait_spin_us / self.wait_spin_hits if self.wait_spin_hits > 0 else 0.0,
                self.total_wait_sleep_us / self.wait_sleep_hits if self.wait_sleep_hits > 0 else 0.0,
            )

    def is_ready(self) -> bool:
        """True when connected and all slot pointers are open."""
        if not self._initialized or self._conn is None:
            return False
        return len(self._conn.dev_ptrs) > 0 and all(ptr is not None for ptr in self._conn.dev_ptrs)

    @property
    def wait_backend_name(self) -> str:
        """Which wait path ``_wait_for_slot`` uses: ``"native"`` or ``"python"``.

        Resolved per-connection (re-resolved on every reconnect, see
        ``_open_ipc_slots``); reads ``"python"`` before the first successful
        connect and whenever the native sidecar failed to resolve.
        """
        return "native" if self._wait_backend is not None else "python"

    def attach_nvml_observer(self, observer: NVMLObserver) -> None:
        """Attach an NVMLObserver for GPU telemetry in get_stats()."""
        self._nvml_observer = observer

    def get_stats(self) -> dict[str, object]:
        """Return current importer statistics."""
        conn = self._conn
        dev_ptrs = conn.dev_ptrs if conn is not None else []
        num_slots = conn.num_slots if conn is not None else 0
        tensors = self._torch.tensors if self._torch is not None else []
        spec = self._spec

        stats: dict[str, object] = {
            "initialized": self._initialized,
            "shm_name": spec.shm_name,
            "shape": self._format.shape if self._format is not None else spec.shape,
            "dtype": self._format.dtype_str if self._format is not None else spec.dtype,
            "device": spec.device,
            "num_slots": num_slots,
            "frame_count": self.frame_count,
            "torch_available": TORCH_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in dev_ptrs],
            "tensor_device": (
                str(tensors[0].device) if TORCH_AVAILABLE and tensors and tensors[0] is not None else "N/A"
            ),
            "wait_backend": self.wait_backend_name,
            "wait_spin_hits": self.wait_spin_hits,
            "wait_sleep_hits": self.wait_sleep_hits,
            "avg_spin_us": self.total_wait_spin_us / self.wait_spin_hits if self.wait_spin_hits > 0 else 0.0,
            "avg_sleep_us": self.total_wait_sleep_us / self.wait_sleep_hits if self.wait_sleep_hits > 0 else 0.0,
            "gpu_wait_active": self._gpu_wait_active,
            "gpu_wait_latched_frame": self._gpu_wait_latched_frame,
        }
        if self._nvml_observer is not None:
            stats["nvml"] = self._nvml_observer.snapshot()
        return stats
