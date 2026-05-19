"""
CUDA IPC Importer for Python Process
Imports GPU memory from TouchDesigner via CUDA IPC handles

Usage:
    # PyTorch tensor (GPU, zero-copy)
    importer = CUDAIPCImporter(shm_name="cudalink_output_ipc", shape=(512, 512, 4))
    tensor = importer.get_frame()  # torch.Tensor on GPU

    # Numpy array (CPU, D2H copy)
    importer = CUDAIPCImporter(shm_name="cudalink_output_ipc", shape=(512, 512, 4))
    array = importer.get_frame_numpy()  # numpy array on CPU

Architecture:
    TouchDesigner Process → IPC Handle in SharedMemory
                                ↓
    Python Process → Open Handle → torch.as_tensor() or numpy D2H copy
                     (once)         (zero-copy)        (GPU→CPU)

Value objects:
    IPCConnection — CUDA runtime, SHM handle, per-slot dev_ptrs/ipc_events, layout.
    Format        — Parsed metadata (shape, dtype, frame_nbytes, numpy_dtype).
    TorchBuffers  — Per-slot zero-copy tensor views (built eagerly).
    CupyBuffers   — Per-slot zero-copy CuPy array views (built eagerly).
    NumpyBuffers  — Pinned host buffer + D2H streams (built lazily on first get_frame_numpy).
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

from . import _nvtx

logger = logging.getLogger(__name__)

# Pre-built NVTX range name strings per slot — eliminates f-string allocation on every get_frame call.
_NVTX_IMPORTER_GET_NAMES: tuple[str, ...] = tuple(f"cudalink.importer.get_frame.slot{i}" for i in range(10))
_NVTX_IMPORTER_NUMPY_NAMES: tuple[str, ...] = tuple(f"cudalink.importer.get_frame_numpy.slot{i}" for i in range(10))

if TYPE_CHECKING:
    from .nvml_observer import NVMLObserver

# Windows timer-resolution helper — reduces time.sleep floor from ~15ms to ~1ms.
# The winmm DLL handle is cached at module level so the load cost is paid once.
if sys.platform == "win32":
    try:
        _winmm = ctypes.WinDLL("winmm")
    except OSError:
        _winmm = None
else:
    _winmm = None


class _HighResTimer:
    """Context manager that requests 1ms timer resolution on Windows.

    On Windows, the default system timer tick is ~15.6ms, making
    ``time.sleep(0.0001)`` wake up 15-150x later than intended. Calling
    ``timeBeginPeriod(1)`` drops the floor to ~1ms for the duration of the
    with-block, then restores the default on exit. No-op on non-Windows.
    """

    __slots__ = ("_active",)

    def __enter__(self) -> _HighResTimer:
        self._active = _winmm is not None
        if self._active:
            _winmm.timeBeginPeriod(1)
        return self

    def __exit__(self, *_: object) -> None:
        if self._active:
            _winmm.timeEndPeriod(1)


# Optional dependencies with fallback
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from .cuda_ipc_wrapper import CUDARuntimeAPI, get_cuda_runtime  # noqa: E402
from .cuda_runtime_types import cudaIpcEventHandle_t, cudaIpcMemHandle_t  # noqa: E402

# Byte size per dtype — module-level constant avoids dict construction on every _dtype_itemsize() call
_DTYPE_SIZES: dict = {"float32": 4, "float16": 2, "bfloat16": 2, "uint8": 1, "uint16": 2, "int8": 1, "int16": 2}

from .shm_protocol import (  # noqa: E402
    _ST_BBH,
    MAGIC_OFFSET,
    MAGIC_SIZE,
    NUM_SLOTS_OFFSET,
    NUM_SLOTS_SIZE,
    PROTOCOL_MAGIC,
    SHM_HEADER_SIZE,
    SLOT_SIZE,
    VERSION_OFFSET,
    VERSION_SIZE,
    AcquireResult,
    DtypeCodec,
    SHMLayout,
    SlotState,
    acquire_slot,
)


def _decode_dtype_str(kind: int, bits: int, flags: int) -> str:
    return DtypeCodec.decode(kind, bits, flags)


# ============================================================
# Value objects
# ============================================================


@dataclass(frozen=True)
class Format:
    """Parsed frame format — shape, dtype, and precomputed derivations.

    Immutable after construction. Two constructors:
    - from_shm(): parse the extended metadata block in SharedMemory.
    - from_overrides(): build from caller-supplied shape/dtype (no SHM read).
    """

    width: int
    height: int
    num_comps: int
    kind: int
    bits: int
    flags: int
    dtype_str: str
    shape: tuple
    numpy_dtype: object  # np.dtype or None when numpy not available
    frame_nbytes: int

    @classmethod
    def from_shm(cls, shm_buf: object, num_slots: int) -> Format | None:
        """Parse extended metadata block from shared memory.

        Returns None when the block is absent or contains zeros (sender not yet
        written metadata).
        """
        layout = SHMLayout(num_slots)
        metadata_offset = layout.metadata_offset
        try:
            width = struct.unpack("<I", bytes(shm_buf[metadata_offset : metadata_offset + 4]))[0]
            height = struct.unpack("<I", bytes(shm_buf[metadata_offset + 4 : metadata_offset + 8]))[0]
            num_comps = struct.unpack("<I", bytes(shm_buf[metadata_offset + 8 : metadata_offset + 12]))[0]
            kind, bits, flags = _ST_BBH.unpack(bytes(shm_buf[metadata_offset + 12 : metadata_offset + 16]))
            if width > 0 and height > 0 and num_comps > 0:
                dtype_str = _decode_dtype_str(kind, bits, flags)
                shape = (height, width, num_comps)
                itemsize = _DTYPE_SIZES.get(dtype_str, bits // 8 or 4)
                frame_nbytes = height * width * num_comps * itemsize
                numpy_dtype = np.dtype(dtype_str) if NUMPY_AVAILABLE else None
                return cls(
                    width=width,
                    height=height,
                    num_comps=num_comps,
                    kind=kind,
                    bits=bits,
                    flags=flags,
                    dtype_str=dtype_str,
                    shape=shape,
                    numpy_dtype=numpy_dtype,
                    frame_nbytes=frame_nbytes,
                )
        except (struct.error, ValueError, IndexError):
            pass
        return None

    @classmethod
    def from_overrides(cls, shape: tuple, dtype_str: str) -> Format:
        """Build from caller-supplied shape/dtype (no SHM read).

        kind/bits/flags are left as 0 sentinels — they are diagnostic fields only
        and are not used by frame consumers.
        """
        height, width, num_comps = shape
        itemsize = _DTYPE_SIZES.get(dtype_str, 4)
        frame_nbytes = height * width * num_comps * itemsize
        numpy_dtype = np.dtype(dtype_str) if NUMPY_AVAILABLE else None
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


@dataclass
class IPCConnection:
    """Live CUDA IPC connection — runtime, SHM handle, per-slot GPU resources, layout.

    Mutable: dev_ptrs/ipc_events/ipc_handles are populated slot-by-slot during
    _open_ipc_slots(), then nulled in-place by close_ipc_handles() / close().
    """

    cuda: object  # CUDARuntimeAPI
    shm_handle: object  # SharedMemory or None after close()
    ipc_version: int
    num_slots: int
    ipc_handles: list  # [cudaIpcMemHandle_t | None]
    dev_ptrs: list  # [c_void_p | None]
    ipc_events: list  # [event_t | None]
    layout: object  # SHMLayout
    shutdown_offset: int
    timestamp_offset: int

    def close_ipc_handles(self) -> None:
        """Close IPC mem handles and events. SharedMemory stays open (used by _reinitialize)."""
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


@dataclass
class TorchBuffers:
    """Per-slot zero-copy torch.Tensor views of GPU memory (built eagerly at init)."""

    tensors: list  # [torch.Tensor]
    wrappers: list  # GC keep-alive refs for __cuda_array_interface__ wrappers

    @classmethod
    def build(cls, conn: IPCConnection, fmt: Format) -> TorchBuffers:
        """Create one zero-copy tensor view per slot via __cuda_array_interface__."""
        typestr_map = {"float32": "<f4", "float16": "<f2", "uint8": "|u1", "uint16": "<u2"}
        typestr = typestr_map.get(fmt.dtype_str)
        if typestr is None:
            raise ValueError(f"Unsupported dtype for torch: {fmt.dtype_str}")

        tensors = []
        wrappers = []
        for slot in range(conn.num_slots):
            if conn.dev_ptrs[slot] is None:
                raise RuntimeError(f"Device pointer for slot {slot} not initialized")

            ptr_value = int(conn.dev_ptrs[slot].value) if conn.dev_ptrs[slot].value is not None else 0
            cuda_array_interface = {
                "shape": fmt.shape,
                "typestr": typestr,
                "data": (ptr_value, False),  # (ptr, read_only)
                "version": 3,
                "strides": None,  # Contiguous C-order
            }

            class CUDAArrayWrapper:
                """Minimal wrapper exposing __cuda_array_interface__ for zero-copy tensor creation."""

                def __init__(self, interface: dict) -> None:
                    self.__cuda_array_interface__ = interface

            wrapper = CUDAArrayWrapper(cuda_array_interface)
            tensor = torch.as_tensor(wrapper, device="cuda")
            wrappers.append(wrapper)
            tensors.append(tensor)

        return cls(tensors=tensors, wrappers=wrappers)


@dataclass
class CupyBuffers:
    """Per-slot zero-copy CuPy array views of GPU memory (built eagerly at init)."""

    arrays: list  # [cp.ndarray]

    @classmethod
    def build(cls, conn: IPCConnection, fmt: Format) -> CupyBuffers:
        """Create one zero-copy CuPy array view per slot via UnownedMemory."""
        dtype_map = {"float32": cp.float32, "float16": cp.float16, "uint8": cp.uint8, "uint16": cp.uint16}
        cp_dtype = dtype_map.get(fmt.dtype_str)
        if cp_dtype is None:
            raise ValueError(f"Unsupported dtype for CuPy: {fmt.dtype_str}")

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

    NumpyBuffers owns the CUDA streams and pinned host allocation. close() tears
    them down idempotently.
    """

    cuda: object  # CUDARuntimeAPI (same instance as IPCConnection.cuda)
    fmt: Format
    buffer: object  # np.ndarray — reusable D2H destination
    pinned_ptr: object  # cudaMallocHost result or None
    host_registered_arr: object  # cudaHostRegister fallback array or None
    pinned_memory_available: bool
    primary_stream: object  # primary D2H CUDA stream (also d2h_streams[0])
    d2h_streams: list  # one per CUDALINK_D2H_STREAMS value; slot 0 == primary_stream
    d2h_events: list  # join-barrier sync events, one per stream
    num_streams: int
    chunk_plan: list  # pre-computed [(offset, size), ...] for multi-stream D2H; empty when num_streams <= 1

    @classmethod
    def build(cls, conn: IPCConnection, fmt: Format, num_streams: int) -> NumpyBuffers:
        """Allocate pinned host buffer + D2H streams.

        Allocation ladder: cudaMallocHost (portable pinned) → cudaHostRegister
        (page-locked) → pageable fallback. Matches current _setup_numpy_buffer logic.
        """
        cuda = conn.cuda
        nbytes = fmt.frame_nbytes

        # Create streams
        primary_stream = cuda.create_stream(flags=0x01)  # cudaStreamNonBlocking
        logger.debug("Created numpy stream: 0x%016x", int(primary_stream.value))
        d2h_streams = [primary_stream] + [cuda.create_stream(flags=0x01) for _ in range(num_streams - 1)]
        d2h_events = [cuda.create_sync_event() for _ in range(num_streams)]
        if num_streams > 1:
            logger.info("Multi-stream D2H enabled: %d streams (CUDALINK_D2H_STREAMS=%d)", num_streams, num_streams)

        pinned_ptr = None
        host_registered_arr = None
        buffer = None
        pinned_memory_available = False

        try:
            # cudaHostAllocPortable (0x01) makes the allocation accessible from any
            # CUDA context in the process — needed when PyTorch and CuPy coexist.
            pinned_ptr = cuda.malloc_host_alloc(nbytes, flags=0x01)
            buf = (ctypes.c_ubyte * nbytes).from_address(pinned_ptr.value)
            buffer = np.frombuffer(buf, dtype=fmt.numpy_dtype).reshape(fmt.shape)
            pinned_memory_available = True
            logger.debug("Allocated portable pinned numpy buffer: %s, %s", fmt.shape, fmt.dtype_str)
        except (RuntimeError, OSError) as e:
            logger.warning(
                "cudaMallocHost failed for %d bytes (%.1f MB) — trying cudaHostRegister: %s",
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

        # Pre-compute (offset, size) pairs for multi-stream D2H; avoids per-frame math.
        chunk_plan: list[tuple[int, int]] = []
        if num_streams > 1:
            chunk = ((nbytes + num_streams - 1) // num_streams + 15) & ~15
            for i in range(num_streams):
                offset = i * chunk
                size = min(chunk, nbytes - offset)
                if size <= 0:
                    break
                chunk_plan.append((offset, size))

        return cls(
            cuda=cuda,
            fmt=fmt,
            buffer=buffer,
            pinned_ptr=pinned_ptr,
            host_registered_arr=host_registered_arr,
            pinned_memory_available=pinned_memory_available,
            primary_stream=primary_stream,
            d2h_streams=d2h_streams,
            d2h_events=d2h_events,
            num_streams=num_streams,
            chunk_plan=chunk_plan,
        )

    def needs_rebuild(self, fmt: Format) -> bool:
        """True when the pre-allocated buffer doesn't match the new format."""
        return self.buffer.shape != fmt.shape or self.buffer.dtype != fmt.numpy_dtype

    def close(self) -> None:
        """Idempotent teardown: free pinned allocation, destroy streams and events."""
        if self.pinned_ptr is not None:
            try:
                self.cuda.free_host(self.pinned_ptr)
                logger.debug("Freed pinned numpy buffer")
            except (RuntimeError, OSError) as e:
                logger.debug("free_host skipped (context gone): %s", e)
            self.pinned_ptr = None

        if self.host_registered_arr is not None:
            try:
                self.cuda.host_unregister(self.host_registered_arr.ctypes.data)
            except (RuntimeError, OSError) as e:
                logger.debug("host_unregister failed: %s", e)
            self.host_registered_arr = None

        for evt in self.d2h_events:
            if evt is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.destroy_event(evt)
        self.d2h_events.clear()

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
                logger.debug("Destroyed numpy stream")
            except (RuntimeError, OSError) as e:
                logger.debug("numpy stream destroy skipped (context gone): %s", e)
            self.primary_stream = None


# ============================================================
# Importer
# ============================================================


class CUDAIPCImporter:
    """Python-side importer for CUDA IPC GPU memory.

    Responsibilities:
    - Read 64-byte IPC handle from SharedMemory (once at startup)
    - Open handle using cudaIpcOpenMemHandle() (once)
    - Create persistent torch.Tensor view (zero-copy) or numpy array (D2H copy)
    - Return tensor/array for each frame

    Performance:
    - Initialization: ~10-100μs (one-time handle opening)
    - Per-frame (torch): < 1μs (just return existing tensor)
    - Per-frame (numpy): ~300μs-5ms depending on resolution and dtype (GPU→CPU D2H copy)
    """

    def __init__(
        self,
        shm_name: str = "cudalink_output_ipc",
        shape: tuple[int, int, int] | None = None,
        dtype: str | None = None,
        debug: bool = False,
        timeout_ms: float = 5000.0,
        device: int = 0,
    ) -> None:
        """Initialize CUDA IPC importer.

        Args:
            shm_name: SharedMemory name where IPC handle is stored
            shape: Expected tensor shape (height, width, channels). If None, auto-detect from metadata.
            dtype: Data type as string: "float32", "float16", or "uint8". If None, auto-detect from metadata.
            debug: Enable verbose debug logging (default: False)
            timeout_ms: Timeout for waiting on producer events in milliseconds (default: 5000.0)
            device: CUDA device index (default: 0). Must match the sender's device.
                    IPC handles are device-scoped; opening a handle on the wrong device
                    causes error 400 (cudaErrorInvalidValue).
        """
        # Construction config (kept in sync with _format after init)
        self.shm_name = shm_name
        self.shape = shape  # May be None initially (will be auto-detected)
        self.dtype = dtype  # May be None initially (will be auto-detected)
        self.debug = debug
        self.timeout_ms = timeout_ms
        self.device = device

        # N1: spin-then-sleep configuration.
        # Phase 1: tight cudaEventQuery spin for up to _spin_us microseconds (no sleep).
        # Phase 2: existing time.sleep(0.0001) poll loop (unchanged).
        # CUDALINK_WAIT_SPIN_US=0 disables Phase 1, restoring pre-batch-2 behaviour.
        self._spin_us: int = int(os.getenv("CUDALINK_WAIT_SPIN_US", "200"))

        # Multi-stream D2H config (NumpyBuffers reads this at build time)
        self._d2h_num_streams: int = max(1, int(os.getenv("CUDALINK_D2H_STREAMS", "1")))

        # Initialization gate
        self._initialized = False

        # Value-object references (all None until _initialize() succeeds)
        self._conn: IPCConnection | None = None
        self._format: Format | None = None
        self._torch: TorchBuffers | None = None
        self._cupy: CupyBuffers | None = None
        self._numpy: NumpyBuffers | None = None

        # Frame tracking
        self.frame_count = 0
        self._last_write_idx = 0

        # Performance metrics
        self.total_wait_event_time = 0.0
        self.total_get_frame_time = 0.0
        self.total_shm_read_us: float = 0.0
        self.last_latency = 0.0
        # N1: spin-phase vs sleep-phase breakdown counters
        self.total_wait_spin_us: float = 0.0
        self.total_wait_sleep_us: float = 0.0
        self.wait_spin_hits: int = 0
        self.wait_sleep_hits: int = 0

        # _numpy_dtype() cache (for pre-init or post-cleanup calls)
        self._cached_dtype_str: str = ""
        self._cached_numpy_dtype: object = None

        # Auto-initialize
        self._initialize()

    # ------------------------------------------------------------------
    # Convenience dtype methods (read self.dtype; kept for backward compat)
    # ------------------------------------------------------------------

    def _dtype_itemsize(self) -> int:
        """Get byte size per element for the configured dtype."""
        return _DTYPE_SIZES[self.dtype]

    def _numpy_dtype(self) -> np.dtype:
        """Get numpy dtype from string dtype (cached)."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required but not installed")
        if self.dtype != self._cached_dtype_str:
            self._cached_numpy_dtype = np.dtype(self.dtype)
            self._cached_dtype_str = self.dtype
        return self._cached_numpy_dtype

    def _torch_dtype(self) -> torch.dtype:
        """Get torch dtype from string dtype."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required but not installed")
        mapping = {"float32": torch.float32, "float16": torch.float16, "uint8": torch.uint8}
        if hasattr(torch, "uint16"):
            mapping["uint16"] = torch.uint16
        dtype = mapping.get(self.dtype)
        if dtype is None:
            raise RuntimeError(
                f"dtype '{self.dtype}' requires PyTorch >= 2.5 (torch.uint16 not available). "
                "Use get_frame_numpy() instead, or upgrade PyTorch."
            )
        return dtype

    def _resolve_stream(self, stream: object) -> int | None:
        """Extract raw CUDA stream pointer from torch/cupy stream or int."""
        if stream is None:
            return None
        if isinstance(stream, int):
            return stream
        if TORCH_AVAILABLE and hasattr(stream, "cuda_stream"):
            return stream.cuda_stream
        if hasattr(stream, "ptr"):
            return stream.ptr
        raise TypeError(
            f"Unsupported stream type: {type(stream)}. Expected torch.cuda.Stream, cupy.cuda.Stream, or int."
        )

    # ------------------------------------------------------------------
    # Phase methods (each returns its piece; orchestrator assembles them)
    # ------------------------------------------------------------------

    def _setup_runtime(self) -> CUDARuntimeAPI:
        """Load CUDA runtime on self.device; raise on device mismatch."""
        cuda = get_cuda_runtime(device=self.device)
        actual_device = cuda.get_device()
        if actual_device != self.device:
            raise RuntimeError(
                f"Device mismatch: requested device {self.device} but CUDA context "
                f"is bound to device {actual_device}. Sender and receiver must use "
                "the same device index."
            )
        logger.info("Loaded CUDA runtime on device %d", actual_device)
        return cuda

    def _open_and_validate_shm(self) -> tuple:
        """Open SharedMemory and validate protocol magic, version, num_slots, shutdown flag.

        Returns:
            (shm, num_slots, ipc_version) on success. Raises on any failure.
        """
        try:
            shm = SharedMemory(name=self.shm_name)
        except FileNotFoundError:
            logger.error("SharedMemory '%s' not found", self.shm_name)
            logger.error("Make sure TouchDesigner CUDAIPCExporter is initialized first")
            raise

        logger.info("Opened SharedMemory: %s", self.shm_name)

        # Validate protocol magic
        try:
            magic = struct.unpack("<I", bytes(shm.buf[MAGIC_OFFSET : MAGIC_OFFSET + MAGIC_SIZE]))[0]
        except (struct.error, ValueError, IndexError):
            logger.error("Cannot read protocol magic.")
            logger.error("  Sender may be using old protocol version (pre-magic).")
            shm.close()
            raise

        if magic != PROTOCOL_MAGIC:
            logger.error("Protocol magic mismatch!")
            logger.error("  Expected: 0x%08X ('CIPD')", PROTOCOL_MAGIC)
            logger.error("  Got:      0x%08X", magic)
            logger.error("  Sender using incompatible protocol version. Please update both TD and Python sides.")
            shm.close()
            raise RuntimeError(f"Protocol magic mismatch: expected 0x{PROTOCOL_MAGIC:08X}, got 0x{magic:08X}")

        ipc_version = struct.unpack("<Q", bytes(shm.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE]))[0]
        num_slots = struct.unpack("<I", bytes(shm.buf[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + NUM_SLOTS_SIZE]))[0]

        if num_slots == 0 or num_slots > 10:
            logger.error(
                "Invalid num_slots=%d read from SharedMemory. Protocol error or corrupted SHM (expected 1-10).",
                num_slots,
            )
            shm.close()
            raise ValueError(f"Invalid num_slots={num_slots}")

        shutdown_offset = SHM_HEADER_SIZE + num_slots * SLOT_SIZE
        try:
            shutdown_flag = shm.buf[shutdown_offset]
        except (OSError, BufferError, IndexError) as e:
            logger.error("Could not read shutdown flag: %s", e)
            shm.close()
            raise

        if shutdown_flag == 1:
            logger.warning("Sender shutdown flag detected - SharedMemory is stale")
            shm.close()
            raise RuntimeError("Sender shutdown flag set — SharedMemory is stale")

        logger.info("Ring buffer with %d slots (v%d)", num_slots, ipc_version)
        return shm, num_slots, ipc_version

    def _parse_format(self, shm: object, num_slots: int) -> Format:
        """Read extended metadata block and return a Format.

        Uses caller-supplied shape/dtype overrides when provided; falls back to
        SHM metadata, then to (512,512,4)/'float32' on parse failure.
        Updates self.shape and self.dtype to stay in sync with the returned Format.
        """
        if self.shape is None or self.dtype is None:
            fmt_from_shm = Format.from_shm(shm.buf, num_slots)
            if fmt_from_shm is not None:
                shape = self.shape if self.shape is not None else fmt_from_shm.shape
                dtype_str = self.dtype if self.dtype is not None else fmt_from_shm.dtype_str
                if shape != fmt_from_shm.shape or dtype_str != fmt_from_shm.dtype_str:
                    # Override one dimension but parsed the other — rebuild from overrides
                    fmt = Format.from_overrides(shape, dtype_str)
                else:
                    fmt = fmt_from_shm
                if self.shape is None:
                    logger.info("Auto-detected shape: %s", fmt.shape)
                if self.dtype is None:
                    logger.info("Auto-detected dtype: %s", fmt.dtype_str)
            else:
                logger.warning("Could not auto-detect metadata; using fallback: shape=(512,512,4), dtype='float32'")
                shape = self.shape or (512, 512, 4)
                dtype_str = self.dtype or "float32"
                fmt = Format.from_overrides(shape, dtype_str)
        else:
            # Both provided by caller — no SHM metadata read needed
            fmt = Format.from_overrides(self.shape, self.dtype)

        # Keep construction hints in sync with the resolved format
        self.shape = fmt.shape
        self.dtype = fmt.dtype_str
        return fmt

    def _open_ipc_slots(
        self,
        cuda: CUDARuntimeAPI,
        shm: object,
        num_slots: int,
        ipc_version: int,
        fmt: Format,
    ) -> IPCConnection:
        """Open all IPC mem + event handles; return a live IPCConnection."""
        ipc_handles: list = [None] * num_slots
        dev_ptrs: list = [None] * num_slots
        ipc_events: list = [None] * num_slots

        for slot in range(num_slots):
            base_offset = SHM_HEADER_SIZE + slot * SLOT_SIZE

            # Read + open memory handle (64 bytes)
            mem_handle_bytes = bytes(shm.buf[base_offset : base_offset + 64])
            ipc_handles[slot] = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)
            # Flag 1 = cudaIpcMemLazyEnablePeerAccess
            dev_ptrs[slot] = cuda.ipc_open_mem_handle(ipc_handles[slot], flags=1)

            # Read + open event handle (64 bytes) if present
            event_handle_bytes = bytes(shm.buf[base_offset + 64 : base_offset + 128])
            if any(event_handle_bytes):
                try:
                    ipc_event_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                    ipc_events[slot] = cuda.ipc_open_event_handle(ipc_event_handle)
                except (RuntimeError, OSError) as e:
                    logger.debug("Failed to open IPC event for slot %d: %s", slot, e)
                    ipc_events[slot] = None

            tensor_info = f"tensor shape={fmt.shape}" if TORCH_AVAILABLE else "torch N/A"
            logger.info(
                "Slot %d: GPU at 0x%016x, %s, event=%s",
                slot,
                dev_ptrs[slot].value,
                tensor_info,
                "YES" if ipc_events[slot] else "NO",
            )

        logger.info("Opened %d IPC buffer slots with GPU-side sync", num_slots)

        layout = SHMLayout(num_slots)
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
        )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def _initialize(self) -> bool:
        """Initialize CUDA IPC resources.

        Returns True on success; False on any failure (already logged).
        """
        if self._initialized:
            logger.debug("Already initialized")
            return True

        try:
            cuda = self._setup_runtime()
            shm, num_slots, ipc_version = self._open_and_validate_shm()
            fmt = self._parse_format(shm, num_slots)
            conn = self._open_ipc_slots(cuda, shm, num_slots, ipc_version, fmt)

            self._conn = conn
            self._format = fmt
            self._torch = TorchBuffers.build(conn, fmt) if TORCH_AVAILABLE else None
            self._cupy = CupyBuffers.build(conn, fmt) if CUPY_AVAILABLE else None
            self._numpy = None  # lazy — built on first get_frame_numpy()
            self._last_write_idx = 0
            self._initialized = True
            logger.info("Initialization complete - ready for zero-copy GPU access")
            return True

        except (OSError, RuntimeError, ValueError, struct.error, IndexError) as e:
            logger.error("Initialization failed: %s", e)
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Slot acquisition + wait
    # ------------------------------------------------------------------

    def _try_acquire(self) -> AcquireResult | None:
        """Acquire next frame via acquire_slot(); dispatch on state.

        Returns:
            AcquireResult on NEW_FRAME (slot/timestamp/write_idx populated), else None.
        Side-effects:
            cleanup() on SHUTDOWN; _reinitialize() on VERSION_CHANGED (single-tick stall).
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
            return None
        if result.state is SlotState.SHUTDOWN:
            logger.info("Producer shutdown detected - cleaning up gracefully")
            self.cleanup()
            return None
        if result.state is SlotState.VERSION_CHANGED:
            logger.debug(
                "TD re-initialized (v%d -> v%d), reopening IPC handle...",
                self._conn.ipc_version,
                result.new_version,
            )
            self._reinitialize()
            return None  # pick up frame next call
        if result.state is SlotState.NO_FRAME:
            return None
        self._last_write_idx = result.write_idx
        return result

    def _wait_for_slot(self, slot: int) -> float:
        """Wait for producer to finish writing slot, with timeout.

        Returns:
            Wait time in microseconds.

        Raises:
            TimeoutError: If wait exceeds timeout_ms.
        """
        conn = self._conn
        wait_start = time.perf_counter()

        if conn.ipc_events[slot]:
            evt = conn.ipc_events[slot]
            query = conn.cuda.query_event
            deadline = wait_start + self.timeout_ms / 1000

            if self._spin_us > 0:
                spin_deadline = wait_start + self._spin_us / 1_000_000
                while time.perf_counter() < spin_deadline:
                    if query(evt):
                        spin_us = (time.perf_counter() - wait_start) * 1_000_000
                        self.total_wait_spin_us += spin_us
                        self.wait_spin_hits += 1
                        return spin_us
                    if time.perf_counter() >= deadline:
                        raise TimeoutError(
                            f"IPC event wait timed out after {self.timeout_ms}ms (slot={slot}) — producer may have crashed"
                        )

            phase2_start = time.perf_counter()
            with _HighResTimer():
                while True:
                    if query(evt):
                        break
                    if time.perf_counter() >= deadline:
                        raise TimeoutError(
                            f"IPC event wait timed out after {self.timeout_ms}ms (slot={slot}) — producer may have crashed"
                        )
                    time.sleep(0.0001)
            self.total_wait_sleep_us += (time.perf_counter() - phase2_start) * 1_000_000
            self.wait_sleep_hits += 1
        elif TORCH_AVAILABLE:
            torch.cuda.synchronize()
        else:
            conn.cuda.synchronize()

        return (time.perf_counter() - wait_start) * 1_000_000

    # ------------------------------------------------------------------
    # Frame consumers
    # ------------------------------------------------------------------

    def get_frame(self, stream: object | None = None) -> torch.Tensor | None:
        """Get current frame as torch.Tensor (GPU, zero-copy).

        Args:
            stream: Optional CUDA stream (torch.cuda.Stream, cupy.cuda.Stream, int, or None).
                    If provided, issues cudaStreamWaitEvent on this stream
                    (non-blocking to CPU). If None, falls back to blocking
                    cudaEventSynchronize for backward compatibility.

        Returns:
            Zero-copy torch.Tensor on GPU, or None if not initialized or no new frame.

        Raises:
            RuntimeError: If torch is not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for get_frame(). Use get_frame_numpy() instead.")
        debug = self.debug
        frame_start = 0.0
        _shm_t = 0.0
        wait_start = 0.0
        if debug:
            frame_start = time.perf_counter()
        if not self._initialized:
            logger.warning("Not initialized - call _initialize() first")
            return None

        if debug:
            _shm_t = time.perf_counter()
        result = self._try_acquire()
        if result is None:
            return None
        read_slot = result.slot
        producer_timestamp = result.timestamp
        if debug:
            self.total_shm_read_us += (time.perf_counter() - _shm_t) * 1_000_000

        if producer_timestamp > 0:
            self.last_latency = (time.perf_counter() - producer_timestamp) * 1000
        else:
            self.last_latency = 0.0

        conn = self._conn
        if debug:
            wait_start = time.perf_counter()

        _nvtx.push_range(_NVTX_IMPORTER_GET_NAMES[read_slot], "purple")
        with _nvtx.verbose_range("cudalink.importer.event_wait", "purple"):
            if stream is not None:
                cuda_stream = self._resolve_stream(stream)
                if conn.ipc_events[read_slot]:
                    conn.cuda.stream_wait_event(cuda_stream, conn.ipc_events[read_slot], 0)
            else:
                try:
                    self._wait_for_slot(read_slot)
                except TimeoutError:
                    logger.error("Producer timeout — returning None")
                    _nvtx.pop_range()
                    return None

        if debug:
            self.total_wait_event_time += (time.perf_counter() - wait_start) * 1_000_000

        self.frame_count += 1

        if debug:
            frame_time = (time.perf_counter() - frame_start) * 1_000_000
            self.total_get_frame_time += frame_time

            if self.frame_count % 97 == 0:
                n = self.frame_count
                sync_mode = "GPU-Events" if all(conn.ipc_events) else "CPU-Sync"
                spin_hit_pct = 100.0 * self.wait_spin_hits / n if n > 0 else 0.0
                logger.debug(
                    "Frame %d [%s]: shm_read=%.1fus stream_wait=%.1fus total=%.1fus "
                    "latency=%.2fms | spin_hit=%.0f%% avg_spin=%.1fus avg_sleep=%.1fus",
                    n,
                    sync_mode,
                    self.total_shm_read_us / n,
                    self.total_wait_event_time / n,
                    self.total_get_frame_time / n,
                    self.last_latency,
                    spin_hit_pct,
                    self.total_wait_spin_us / self.wait_spin_hits if self.wait_spin_hits > 0 else 0.0,
                    self.total_wait_sleep_us / self.wait_sleep_hits if self.wait_sleep_hits > 0 else 0.0,
                )

        _nvtx.pop_range()
        return self._torch.tensors[read_slot]

    def get_frame_numpy(self) -> np.ndarray | None:
        """Get current frame as numpy array (CPU, involves D2H copy).

        Returns:
            Numpy array on CPU, or None if not initialized or no new frame.

        Raises:
            RuntimeError: If numpy is not available
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for get_frame_numpy()")
        debug = self.debug
        frame_start = 0.0
        _shm_t = 0.0
        _wait_t = 0.0
        _d2h_t = 0.0
        d2h_time = 0.0
        if debug:
            frame_start = time.perf_counter()
        if not self._initialized:
            logger.warning("Not initialized - call _initialize() first")
            return None

        if debug:
            _shm_t = time.perf_counter()
        result = self._try_acquire()
        if result is None:
            return None
        read_slot = result.slot
        producer_timestamp = result.timestamp
        if debug:
            self.total_shm_read_us += (time.perf_counter() - _shm_t) * 1_000_000

        if producer_timestamp > 0:
            self.last_latency = (time.perf_counter() - producer_timestamp) * 1000
        else:
            self.last_latency = 0.0

        conn = self._conn
        fmt = self._format
        nbytes = fmt.frame_nbytes

        # Lazily build (or rebuild) NumpyBuffers when format changes
        if self._numpy is None or self._numpy.needs_rebuild(fmt):
            if self._numpy is not None:
                self._numpy.close()
            self._numpy = NumpyBuffers.build(conn, fmt, self._d2h_num_streams)

        nb = self._numpy

        # CPU-side event poll + async D2H + synchronize.
        # Uses _wait_for_slot (query_event CPU poll) rather than stream_wait_event because
        # cudaStreamWaitEvent on a cross-process IPC event has high kernel-mode latency on
        # Windows (~100-300ms when followed by stream_synchronize). The producer records
        # the IPC event BEFORE publishing write_idx (improvement #2), so the event is always
        # pre-signaled when the consumer reads write_idx — query_event returns True on the
        # first call with no polling delay.
        _nvtx.push_range(_NVTX_IMPORTER_NUMPY_NAMES[read_slot], "orange")
        if debug:
            _wait_t = time.perf_counter()
        with _nvtx.verbose_range("cudalink.importer.event_wait", "orange"):
            try:
                self._wait_for_slot(read_slot)
            except TimeoutError:
                logger.error("Producer timeout — returning None")
                _nvtx.pop_range()
                return None
        if debug:
            self.total_wait_event_time += (time.perf_counter() - _wait_t) * 1_000_000

        if debug:
            _d2h_t = time.perf_counter()
        with _nvtx.verbose_range("cudalink.importer.d2h_copy", "orange"):
            n_streams = nb.num_streams
            if n_streams <= 1:
                conn.cuda.memcpy_async(
                    dst=ctypes.c_void_p(nb.buffer.ctypes.data),
                    src=conn.dev_ptrs[read_slot],
                    count=nbytes,
                    kind=2,  # cudaMemcpyDeviceToHost
                    stream=nb.primary_stream,
                )
                conn.cuda.stream_synchronize(nb.primary_stream)
            else:
                dst_base = nb.buffer.ctypes.data
                src_base = conn.dev_ptrs[read_slot].value
                issued = 0
                for i, (offset, size) in enumerate(nb.chunk_plan):
                    conn.cuda.memcpy_async(
                        dst=ctypes.c_void_p(dst_base + offset),
                        src=ctypes.c_void_p(src_base + offset),
                        count=size,
                        kind=2,
                        stream=nb.d2h_streams[i],
                    )
                    conn.cuda.record_event(nb.d2h_events[i], stream=nb.d2h_streams[i])
                    issued = i + 1
                for i in range(issued):
                    conn.cuda.wait_event(nb.d2h_events[i])
        conn.cuda.check_sticky_error("get_frame_numpy")
        if debug:
            d2h_time = (time.perf_counter() - _d2h_t) * 1_000_000

        self.frame_count += 1

        if debug:
            frame_time = (time.perf_counter() - frame_start) * 1_000_000
            self.total_get_frame_time += frame_time

            if self.frame_count % 97 == 0:
                n = self.frame_count
                logger.debug(
                    "Frame %d (numpy): shm_read=%.1fus wait=%.1fus d2h=%.1fus total=%.1fus latency=%.2fms",
                    n,
                    self.total_shm_read_us / n,
                    self.total_wait_event_time / n,
                    d2h_time,
                    self.total_get_frame_time / n,
                    self.last_latency,
                )

        _nvtx.pop_range()
        return nb.buffer

    def get_frame_cupy(self, stream: object | None = None) -> cp.ndarray | None:
        """Get current frame as CuPy GPU array (zero-copy).

        Args:
            stream: Optional CuPy stream (cupy.cuda.Stream, torch.cuda.Stream, int, or None).
                    If provided, issues cudaStreamWaitEvent on this stream
                    (non-blocking to CPU). If None, uses CuPy's current stream.

        Returns:
            Zero-copy CuPy array on GPU, or None if not initialized

        Raises:
            RuntimeError: If CuPy is not available
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("cupy is required for get_frame_cupy(). Install: pip install cupy-cuda12x")

        if not self._initialized:
            logger.warning("Not initialized - call _initialize() first")
            return None

        result = self._try_acquire()
        if result is None:
            return None
        read_slot = result.slot
        producer_timestamp = result.timestamp
        if producer_timestamp > 0:
            self.last_latency = (time.perf_counter() - producer_timestamp) * 1000
        else:
            self.last_latency = 0.0

        conn = self._conn

        if stream is None:
            stream = cp.cuda.get_current_stream()
        else:
            if not isinstance(stream, cp.cuda.Stream):
                cuda_stream_ptr = self._resolve_stream(stream)
                stream = cp.cuda.ExternalStream(cuda_stream_ptr)

        if conn.ipc_events[read_slot]:
            cp.cuda.runtime.streamWaitEvent(stream.ptr, int(conn.ipc_events[read_slot]), 0)

        return self._cupy.arrays[read_slot]

    # ------------------------------------------------------------------
    # Re-initialization (TD sender restarted with new IPC handles)
    # ------------------------------------------------------------------

    def _reinitialize(self) -> None:
        """Re-open all IPC handles after TD re-initialization."""
        old_conn = self._conn
        shm = old_conn.shm_handle  # keep SHM alive across handle close

        # Close old IPC handles only (SHM stays open)
        old_conn.close_ipc_handles()

        # Re-read version and num_slots
        new_ipc_version = struct.unpack("<Q", bytes(shm.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE]))[0]
        new_num_slots = struct.unpack("<I", bytes(shm.buf[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + NUM_SLOTS_SIZE]))[0]

        # Re-parse format metadata (shape/dtype may have changed)
        new_layout = SHMLayout(new_num_slots)
        metadata_offset = new_layout.metadata_offset
        new_fmt = self._format  # fallback: keep existing
        try:
            width = struct.unpack("<I", bytes(shm.buf[metadata_offset : metadata_offset + 4]))[0]
            height = struct.unpack("<I", bytes(shm.buf[metadata_offset + 4 : metadata_offset + 8]))[0]
            num_comps = struct.unpack("<I", bytes(shm.buf[metadata_offset + 8 : metadata_offset + 12]))[0]
            kind, bits, flags = _ST_BBH.unpack(bytes(shm.buf[metadata_offset + 12 : metadata_offset + 16]))
            if width > 0 and height > 0 and num_comps > 0:
                new_dtype_str = _decode_dtype_str(kind, bits, flags)
                new_shape = (height, width, num_comps)
                itemsize = _DTYPE_SIZES.get(new_dtype_str, bits // 8 or 4)
                new_fmt = Format(
                    width=width,
                    height=height,
                    num_comps=num_comps,
                    kind=kind,
                    bits=bits,
                    flags=flags,
                    dtype_str=new_dtype_str,
                    shape=new_shape,
                    numpy_dtype=np.dtype(new_dtype_str) if NUMPY_AVAILABLE else None,
                    frame_nbytes=height * width * num_comps * itemsize,
                )
        except (struct.error, ValueError, IndexError) as e:
            logger.debug("Could not re-read metadata during reinit: %s", e)

        if new_fmt != self._format:
            logger.info(
                "Metadata changed on reinit: %s %s -> %s %s",
                self._format.shape,
                self._format.dtype_str,
                new_fmt.shape,
                new_fmt.dtype_str,
            )
            # Tear down numpy buffers — will be rebuilt lazily on next get_frame_numpy()
            if self._numpy is not None:
                self._numpy.close()
                self._numpy = None
            self.shape = new_fmt.shape
            self.dtype = new_fmt.dtype_str

        self._format = new_fmt

        # Rebuild IPC connection (reusing the still-open SHM handle)
        new_conn = self._open_ipc_slots(old_conn.cuda, shm, new_num_slots, new_ipc_version, new_fmt)
        self._conn = new_conn

        # Rebuild torch buffers (cupy not rebuilt — matches pre-refactor behavior)
        if TORCH_AVAILABLE:
            self._torch = TorchBuffers.build(new_conn, new_fmt)

        logger.debug("Reopened %d IPC handles v%d", new_num_slots, new_ipc_version)
        for slot in range(new_num_slots):
            logger.debug("Slot %d: GPU at 0x%016x", slot, new_conn.dev_ptrs[slot].value)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Cleanup CUDA IPC resources."""
        if getattr(self, "_numpy", None) is not None:
            self._numpy.close()
            self._numpy = None
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
        # TorchBuffers and CupyBuffers hold zero-copy views; GC reclaims on deref
        self._torch = None
        self._cupy = None
        self._format = None
        self._initialized = False
        logger.info("Cleanup complete")

    def __del__(self) -> None:
        if getattr(self, "_initialized", False):
            self.cleanup()

    def __enter__(self) -> CUDAIPCImporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.cleanup()
        return None

    # ------------------------------------------------------------------
    # Status / stats
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Check if importer is ready for frame access."""
        if not self._initialized or self._conn is None:
            return False
        return len(self._conn.dev_ptrs) > 0 and all(ptr is not None for ptr in self._conn.dev_ptrs)

    def attach_nvml_observer(self, observer: NVMLObserver) -> None:
        """Attach an NVMLObserver for GPU telemetry in get_stats()."""
        self._nvml_observer = observer

    def get_stats(self) -> dict[str, object]:
        """Get importer statistics."""
        conn = self._conn
        dev_ptrs = conn.dev_ptrs if conn is not None else []
        num_slots = conn.num_slots if conn is not None else 0
        tensors = self._torch.tensors if self._torch is not None else []

        stats: dict[str, object] = {
            "initialized": self._initialized,
            "shape": self.shape,
            "dtype": self.dtype,
            "frame_count": self.frame_count,
            "shm_name": self.shm_name,
            "num_slots": num_slots,
            "torch_available": TORCH_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in dev_ptrs],
            "tensor_device": (
                str(tensors[0].device) if TORCH_AVAILABLE and tensors and tensors[0] is not None else "N/A"
            ),
            "wait_spin_hits": self.wait_spin_hits,
            "wait_sleep_hits": self.wait_sleep_hits,
            "avg_spin_us": self.total_wait_spin_us / self.wait_spin_hits if self.wait_spin_hits > 0 else 0.0,
            "avg_sleep_us": self.total_wait_sleep_us / self.wait_sleep_hits if self.wait_sleep_hits > 0 else 0.0,
        }
        observer = getattr(self, "_nvml_observer", None)
        if observer is not None:
            stats["nvml"] = observer.snapshot()
        return stats
