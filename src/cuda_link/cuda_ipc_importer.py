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
"""

from __future__ import annotations

import ctypes
import logging
import struct
import time
import traceback
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)

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

from .cuda_ipc_wrapper import cudaIpcEventHandle_t, cudaIpcMemHandle_t, get_cuda_runtime  # noqa: E402

# Protocol layout constants (must match td_exporter/CUDAIPCExtension.py)
PROTOCOL_MAGIC = 0x43495043  # "CIPC" - protocol validation magic number
MAGIC_OFFSET = 0
MAGIC_SIZE = 4
VERSION_OFFSET = 4
VERSION_SIZE = 8
NUM_SLOTS_OFFSET = 12
NUM_SLOTS_SIZE = 4
WRITE_IDX_OFFSET = 16
WRITE_IDX_SIZE = 4
SHM_HEADER_SIZE = 20  # Total header: 4+8+4+4 (was 16, now 20 with magic)
SLOT_SIZE = 128  # 64B mem_handle + 64B event_handle
SHUTDOWN_FLAG_SIZE = 1
METADATA_SIZE = 20  # 4B width + 4B height + 4B num_comps + 4B dtype_code + 4B buffer_size
TIMESTAMP_SIZE = 8  # 8B float64 producer timestamp (for latency measurement)
# TIMESTAMP_OFFSET calculated at runtime: SHM_HEADER_SIZE + (num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE


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
    ):
        """Initialize CUDA IPC importer.

        Args:
            shm_name: SharedMemory name where IPC handle is stored
            shape: Expected tensor shape (height, width, channels). If None, auto-detect from metadata.
            dtype: Data type as string: "float32", "float16", or "uint8". If None, auto-detect from metadata.
            debug: Enable verbose debug logging (default: False)
            timeout_ms: Timeout for waiting on producer events in milliseconds (default: 5000.0)
        """
        self.shm_name = shm_name
        self.shape = shape  # May be None initially (will be auto-detected)
        self.dtype = dtype  # May be None initially (will be auto-detected)
        self.debug = debug
        self.timeout_ms = timeout_ms

        # CUDA runtime API
        self.cuda = None
        self._initialized = False

        # IPC state
        self.shm_handle = None
        self.ipc_version = 0

        # Ring buffer state (initialized after reading SharedMemory)
        self.num_slots = 0  # Number of buffer slots (read from SharedMemory)
        self.ipc_handles = []  # List of IPC mem handles
        self.dev_ptrs = []  # List of GPU buffer pointers
        self.ipc_events = []  # List of IPC events for GPU-side sync

        # Tensor state (arrays for ring buffer) - only if torch available
        self.tensors = []  # List of zero-copy tensor views (one per slot)
        self._wrappers = []  # Keep wrappers alive to prevent GC

        # CuPy state (arrays for ring buffer) - only if cupy available
        self.cupy_arrays = []  # List of zero-copy CuPy arrays (one per slot)

        # Numpy state (pre-allocated buffer for D2H copy)
        self._numpy_buffer = None  # Reused numpy array (avoids per-frame allocation)
        self._pinned_ptr = None  # Pinned host memory pointer (cudaMallocHost)

        # Frame tracking
        self.frame_count = 0
        self._last_write_idx = 0  # Track last read write_idx for new-frame detection

        # Performance metrics
        self.total_wait_event_time = 0.0
        self.total_get_frame_time = 0.0
        self.total_shm_read_us: float = 0.0
        self.last_latency = 0.0  # End-to-end latency from producer timestamp (ms)

        # Cached SharedMemory offsets (computed once in _initialize(), constant thereafter)
        self._shutdown_offset: int = 0
        self._timestamp_offset: int = 0

        # Auto-initialize
        self._initialize()

    def _dtype_itemsize(self) -> int:
        """Get byte size per element for the configured dtype."""
        sizes = {"float32": 4, "float16": 2, "uint8": 1, "uint16": 2}
        return sizes[self.dtype]

    def _numpy_dtype(self) -> np.dtype:
        """Get numpy dtype from string dtype."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required but not installed")
        return np.dtype(self.dtype)

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
        """Extract raw CUDA stream pointer from torch/cupy stream or int.

        Args:
            stream: torch.cuda.Stream, cupy.cuda.Stream, or raw int pointer

        Returns:
            int: Raw CUDA stream pointer value

        Raises:
            TypeError: If stream type is unsupported
        """
        if stream is None:
            return None
        if isinstance(stream, int):
            return stream
        # Try torch.cuda.Stream
        if TORCH_AVAILABLE and hasattr(stream, "cuda_stream"):
            return stream.cuda_stream
        # Try cupy.cuda.Stream (via CUPY_AVAILABLE check if added later)
        if hasattr(stream, "ptr"):
            return stream.ptr
        raise TypeError(
            f"Unsupported stream type: {type(stream)}. Expected torch.cuda.Stream, cupy.cuda.Stream, or int."
        )

    def _initialize(self) -> bool:
        """Initialize CUDA IPC resources.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.debug("Already initialized")
            return True

        try:
            # Load CUDA runtime
            self.cuda = get_cuda_runtime()
            logger.info("Loaded CUDA runtime")

            # Create internal stream for numpy async D2H operations
            self._numpy_stream = self.cuda.create_stream(flags=0x01)  # cudaStreamNonBlocking
            logger.debug("Created numpy stream: 0x%016x", int(self._numpy_stream.value))

            # Open SharedMemory to read IPC handle
            try:
                self.shm_handle = SharedMemory(name=self.shm_name)
                logger.info("Opened SharedMemory: %s", self.shm_name)
            except FileNotFoundError:
                logger.error("SharedMemory '%s' not found", self.shm_name)
                logger.error("Make sure TouchDesigner CUDAIPCExporter is initialized first")
                return False

            # Read header from SharedMemory (magic + version + num_slots + write_idx)

            # Validate protocol magic number (new in this version)
            try:
                magic = struct.unpack("<I", bytes(self.shm_handle.buf[MAGIC_OFFSET : MAGIC_OFFSET + MAGIC_SIZE]))[0]
                if magic != PROTOCOL_MAGIC:
                    logger.error("Protocol magic mismatch!")
                    logger.error("  Expected: 0x%08X ('CIPC')", PROTOCOL_MAGIC)
                    logger.error("  Got:      0x%08X", magic)
                    logger.error(
                        "  Sender using incompatible protocol version. Please update both TD and Python sides."
                    )
                    self.shm_handle.close()
                    self.shm_handle = None
                    return False
            except (struct.error, ValueError, IndexError):
                logger.error("Cannot read protocol magic.")
                logger.error("  Sender may be using old protocol version (pre-magic).")
                self.shm_handle.close()
                self.shm_handle = None
                return False

            self.ipc_version = struct.unpack(
                "<Q", bytes(self.shm_handle.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE])
            )[0]
            self.num_slots = struct.unpack(
                "<I", bytes(self.shm_handle.buf[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + NUM_SLOTS_SIZE])
            )[0]
            logger.info("Ring buffer with %d slots (v%d)", self.num_slots, self.ipc_version)

            # Validate num_slots bounds (matches Receiver validation in CUDAIPCExtension)
            if self.num_slots == 0 or self.num_slots > 10:
                logger.error(
                    "Invalid num_slots=%d read from SharedMemory. Protocol error or corrupted SHM (expected 1-10).",
                    self.num_slots,
                )
                self.shm_handle.close()
                self.shm_handle = None
                return False

            # Check if sender has shut down (stale SharedMemory with invalid IPC handles)
            shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
            try:
                shutdown_flag = self.shm_handle.buf[shutdown_offset]
                if shutdown_flag == 1:
                    logger.warning("Sender shutdown flag detected - SharedMemory is stale")
                    self.shm_handle.close()
                    self.shm_handle = None
                    return False
            except (OSError, BufferError, IndexError) as e:
                logger.error("Could not read shutdown flag: %s", e)
                self.shm_handle.close()
                self.shm_handle = None
                return False

            # Auto-detect shape and dtype from extended metadata if not provided
            if self.shape is None or self.dtype is None:
                shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
                metadata_offset = shutdown_offset + 1  # After shutdown flag

                # Try to read extended metadata
                try:
                    width = struct.unpack("<I", bytes(self.shm_handle.buf[metadata_offset : metadata_offset + 4]))[0]
                    height = struct.unpack("<I", bytes(self.shm_handle.buf[metadata_offset + 4 : metadata_offset + 8]))[
                        0
                    ]
                    num_comps = struct.unpack(
                        "<I", bytes(self.shm_handle.buf[metadata_offset + 8 : metadata_offset + 12])
                    )[0]
                    dtype_code = struct.unpack(
                        "<I", bytes(self.shm_handle.buf[metadata_offset + 12 : metadata_offset + 16])
                    )[0]

                    if width > 0 and height > 0 and num_comps > 0:
                        if self.shape is None:
                            self.shape = (height, width, num_comps)
                            logger.info("Auto-detected shape: %s", self.shape)
                        if self.dtype is None:
                            dtype_map = {0: "float32", 1: "float16", 2: "uint8", 3: "uint16"}
                            self.dtype = dtype_map.get(dtype_code, "float32")
                            logger.info("Auto-detected dtype: %s", self.dtype)
                    else:
                        raise ValueError("Metadata contains zeros")
                except (struct.error, ValueError, IndexError) as e:
                    if self.shape is None or self.dtype is None:
                        logger.warning("Could not auto-detect metadata: %s", e)
                        logger.warning("Using fallback: shape=(512,512,4), dtype='float32'")
                        self.shape = self.shape or (512, 512, 4)
                        self.dtype = self.dtype or "float32"

            # Initialize arrays for ring buffer
            self.ipc_handles = [None] * self.num_slots
            self.dev_ptrs = [None] * self.num_slots
            self.ipc_events = [None] * self.num_slots
            self.tensors = [None] * self.num_slots
            self._wrappers = [None] * self.num_slots
            self.cupy_arrays = [None] * self.num_slots
            self._last_write_idx = 0  # Reset frame tracking on re-init

            # Open all IPC handles
            for slot in range(self.num_slots):
                base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

                # Read memory handle (64 bytes)
                mem_handle_bytes = bytes(self.shm_handle.buf[base_offset : base_offset + 64])
                self.ipc_handles[slot] = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)

                # Open IPC memory handle (ONCE - expensive operation)
                # Flag 1 = cudaIpcMemLazyEnablePeerAccess
                self.dev_ptrs[slot] = self.cuda.ipc_open_mem_handle(self.ipc_handles[slot], flags=1)

                # Read event handle (64 bytes)
                event_handle_bytes = bytes(self.shm_handle.buf[base_offset + 64 : base_offset + 128])
                if any(event_handle_bytes):
                    try:
                        ipc_event_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                        self.ipc_events[slot] = self.cuda.ipc_open_event_handle(ipc_event_handle)
                    except (RuntimeError, OSError) as e:
                        logger.debug("Failed to open IPC event for slot %d: %s", slot, e)
                        self.ipc_events[slot] = None

                # Create tensor view for this slot if torch available
                if TORCH_AVAILABLE:
                    self.tensors[slot] = self._create_tensor_view(slot)
                    tensor_info = f"tensor shape={self.tensors[slot].shape}"
                else:
                    tensor_info = "torch N/A"

                # Create CuPy array view for this slot if cupy available
                if CUPY_AVAILABLE:
                    self.cupy_arrays[slot] = self._create_cupy_view(slot)
                    logger.debug("Slot %d: Created CuPy array shape=%s", slot, self.cupy_arrays[slot].shape)

                logger.info(
                    "Slot %d: GPU at 0x%016x, %s, event=%s",
                    slot,
                    self.dev_ptrs[slot].value,
                    tensor_info,
                    "YES" if self.ipc_events[slot] else "NO",
                )

            logger.info("Opened %d IPC buffer slots with GPU-side sync", self.num_slots)

            # Cache constant SharedMemory offsets so hot paths avoid per-frame arithmetic
            self._shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
            self._timestamp_offset = self._shutdown_offset + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

            self._initialized = True
            logger.info("Initialization complete - ready for zero-copy GPU access")
            return True

        except (OSError, RuntimeError, ValueError, struct.error, IndexError) as e:
            logger.error("Initialization failed: %s", e)
            traceback.print_exc()
            return False

    def _create_tensor_view(self, slot: int) -> torch.Tensor:
        """Create zero-copy torch.Tensor view of GPU memory for a specific slot.

        Uses __cuda_array_interface__ for zero-copy tensor creation.

        Args:
            slot: Buffer slot index (0 to num_slots-1)

        Returns:
            Torch tensor backed by IPC GPU memory for this slot
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for tensor views")

        if self.dev_ptrs[slot] is None:
            raise RuntimeError(f"Device pointer for slot {slot} not initialized")

        # Calculate total elements
        height, width, channels = self.shape

        # Determine typestr based on dtype
        if self.dtype == "float32":
            typestr = "<f4"  # Little-endian float32
        elif self.dtype == "float16":
            typestr = "<f2"  # Little-endian float16
        elif self.dtype == "uint8":
            typestr = "|u1"  # Unsigned 8-bit integer
        elif self.dtype == "uint16":
            typestr = "<u2"  # Little-endian unsigned 16-bit integer
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        # Create __cuda_array_interface__ descriptor
        ptr_value = int(self.dev_ptrs[slot].value) if self.dev_ptrs[slot].value is not None else 0

        cuda_array_interface = {
            "shape": self.shape,
            "typestr": typestr,
            "data": (ptr_value, False),  # (ptr, read_only)
            "version": 3,
            "strides": None,  # Contiguous C-order
        }

        # Create wrapper with __cuda_array_interface__
        class CUDAArrayWrapper:
            """Minimal wrapper exposing __cuda_array_interface__ for zero-copy tensor creation."""

            def __init__(self, interface: dict) -> None:
                self.__cuda_array_interface__ = interface

        wrapper = CUDAArrayWrapper(cuda_array_interface)
        self._wrappers[slot] = wrapper  # Keep alive to prevent GC

        # Create torch tensor from CUDA array interface (zero-copy)
        tensor = torch.as_tensor(wrapper, device="cuda")

        # Validate shape and dtype
        if tensor.shape != self.shape:
            raise ValueError(f"Shape mismatch: {tensor.shape} != {self.shape}")
        torch_dtype = self._torch_dtype()
        if tensor.dtype != torch_dtype:
            raise ValueError(f"Dtype mismatch: {tensor.dtype} != {torch_dtype}")

        return tensor

    def _create_cupy_view(self, slot: int) -> cp.ndarray:
        """Create zero-copy CuPy array view of GPU memory for a specific slot.

        Uses cp.cuda.UnownedMemory for zero-copy array creation from external pointer.

        Args:
            slot: Buffer slot index (0 to num_slots-1)

        Returns:
            CuPy array backed by IPC GPU memory for this slot
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("cupy is required for CuPy views")

        if self.dev_ptrs[slot] is None:
            raise RuntimeError(f"Device pointer for slot {slot} not initialized")

        # Calculate buffer size
        height, width, channels = self.shape
        itemsize = self._dtype_itemsize()
        nbytes = height * width * channels * itemsize

        # Determine CuPy dtype
        dtype_map = {"float32": cp.float32, "float16": cp.float16, "uint8": cp.uint8, "uint16": cp.uint16}
        cp_dtype = dtype_map.get(self.dtype)
        if cp_dtype is None:
            raise ValueError(f"Unsupported dtype for CuPy: {self.dtype}")

        # Create UnownedMemory (zero-copy, no ownership transfer)
        ptr_value = int(self.dev_ptrs[slot].value)
        mem = cp.cuda.UnownedMemory(ptr_value, nbytes, owner=self)
        memptr = cp.cuda.MemoryPointer(mem, 0)

        # Create CuPy array from memory pointer
        cupy_array = cp.ndarray(self.shape, dtype=cp_dtype, memptr=memptr)

        return cupy_array

    def _get_read_slot(self) -> int | None:
        """Read write_idx from SharedMemory and compute read slot.

        Returns:
            Slot index to read from, or None if no new frame available
        """
        write_idx = struct.unpack_from("<I", self.shm_handle.buf, WRITE_IDX_OFFSET)[0]

        if write_idx == 0:
            return None  # No frames written yet

        if write_idx == self._last_write_idx:
            return None  # Same frame as last read

        self._last_write_idx = write_idx
        return (write_idx - 1) % self.num_slots

    def _wait_for_slot(self, slot: int) -> float:
        """Wait for producer to finish writing slot, with timeout.

        Args:
            slot: Slot index to wait for

        Returns:
            Wait time in microseconds

        Raises:
            TimeoutError: If wait exceeds timeout_ms
        """
        wait_start = time.perf_counter()

        # GPU-side synchronization if event available
        if self.ipc_events[slot]:
            # Poll with timeout instead of blocking indefinitely
            deadline = wait_start + self.timeout_ms / 1000
            while True:
                if self.cuda.query_event(self.ipc_events[slot]):
                    break
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        f"IPC event wait timed out after {self.timeout_ms}ms (slot={slot}) — producer may have crashed"
                    )
                time.sleep(0.0001)  # 100µs polling interval
        elif TORCH_AVAILABLE:
            # Fallback: CPU synchronization with torch
            torch.cuda.synchronize()
        else:
            # Fallback: CPU synchronization with CUDA runtime
            self.cuda.synchronize()

        return (time.perf_counter() - wait_start) * 1_000_000  # microseconds

    def get_frame(self, stream: object | None = None) -> torch.Tensor | None:
        """Get current frame as torch.Tensor (GPU, zero-copy).

        Args:
            stream: Optional CUDA stream (torch.cuda.Stream, cupy.cuda.Stream, int, or None).
                    If provided, issues cudaStreamWaitEvent on this stream
                    (non-blocking to CPU). If None, falls back to blocking
                    cudaEventSynchronize for backward compatibility.

        Returns:
            Zero-copy torch.Tensor on GPU, or None if not initialized

        Raises:
            RuntimeError: If torch is not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for get_frame(). Use get_frame_numpy() instead.")

        # Start frame timer (only if debug)
        if self.debug:
            frame_start = time.perf_counter()

        if not self._initialized:
            logger.warning("Not initialized - call _initialize() first")
            return None

        # Check for producer shutdown + version change + read slot (all SharedMemory reads)
        if self.debug:
            _shm_t = time.perf_counter()
        try:
            if self.shm_handle.buf[self._shutdown_offset] == 1:
                logger.info("Producer shutdown detected - cleaning up gracefully")
                self.cleanup()
                return None
        except (OSError, BufferError) as e:
            logger.debug("Could not read shutdown flag: %s", e)

        current_version = struct.unpack_from("<Q", self.shm_handle.buf, VERSION_OFFSET)[0]
        if current_version != self.ipc_version:
            logger.debug("TD re-initialized (v%d -> v%d), reopening IPC handle...", self.ipc_version, current_version)
            self._reinitialize()

        read_slot = self._get_read_slot()
        if read_slot is None:
            return None  # No new frame available

        producer_timestamp = struct.unpack_from("<d", self.shm_handle.buf, self._timestamp_offset)[0]
        if self.debug:
            self.total_shm_read_us += (time.perf_counter() - _shm_t) * 1_000_000

        if producer_timestamp > 0:  # Will be 0.0 on first frame before sender writes
            self.last_latency = (time.perf_counter() - producer_timestamp) * 1000
        else:
            self.last_latency = 0.0

        # Wait for slot to be ready (stream-ordered if stream provided, else CPU-blocking)
        if self.debug:
            wait_start = time.perf_counter()

        if stream is not None:
            # Stream-ordered wait (non-blocking to CPU)
            cuda_stream = self._resolve_stream(stream)
            if self.ipc_events[read_slot]:
                self.cuda.stream_wait_event(cuda_stream, self.ipc_events[read_slot], 0)
            # Note: No fallback sync needed - downstream GPU work will naturally wait on stream
        else:
            # Backward-compatible blocking wait
            try:
                self._wait_for_slot(read_slot)
            except TimeoutError:
                logger.error("Producer timeout — returning None")
                return None

        if self.debug:
            self.total_wait_event_time += (time.perf_counter() - wait_start) * 1_000_000

        # Frame tracking
        self.frame_count += 1

        if self.debug:
            frame_time = (time.perf_counter() - frame_start) * 1_000_000
            self.total_get_frame_time += frame_time

            if self.frame_count % 97 == 0:
                n = self.frame_count
                sync_mode = "GPU-Events" if all(self.ipc_events) else "CPU-Sync"
                logger.debug(
                    "Frame %d [%s]: shm_read=%.1fus stream_wait=%.1fus total=%.1fus latency=%.2fms",
                    n,
                    sync_mode,
                    self.total_shm_read_us / n,
                    self.total_wait_event_time / n,
                    self.total_get_frame_time / n,
                    self.last_latency,
                )

        # Return tensor for this slot (zero-copy, no allocation)
        return self.tensors[read_slot]

    def get_frame_numpy(self) -> np.ndarray | None:
        """Get current frame as numpy array (CPU, involves D2H copy).

        Returns:
            Numpy array on CPU, or None if not initialized

        Raises:
            RuntimeError: If numpy is not available
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for get_frame_numpy()")

        # Start frame timer (only if debug)
        if self.debug:
            frame_start = time.perf_counter()

        if not self._initialized:
            logger.warning("Not initialized - call _initialize() first")
            return None

        # Check for producer shutdown + version change + read slot (all SharedMemory reads)
        if self.debug:
            _shm_t = time.perf_counter()
        try:
            if self.shm_handle.buf[self._shutdown_offset] == 1:
                logger.info("Producer shutdown detected - cleaning up gracefully")
                self.cleanup()
                return None
        except (OSError, BufferError) as e:
            logger.debug("Could not read shutdown flag: %s", e)

        current_version = struct.unpack_from("<Q", self.shm_handle.buf, VERSION_OFFSET)[0]
        if current_version != self.ipc_version:
            logger.debug("TD re-initialized (v%d -> v%d), reopening IPC handle...", self.ipc_version, current_version)
            self._reinitialize()

        read_slot = self._get_read_slot()
        if read_slot is None:
            return None  # No new frame available

        producer_timestamp = struct.unpack_from("<d", self.shm_handle.buf, self._timestamp_offset)[0]
        if self.debug:
            self.total_shm_read_us += (time.perf_counter() - _shm_t) * 1_000_000

        if producer_timestamp > 0:  # Will be 0.0 on first frame before sender writes
            self.last_latency = (time.perf_counter() - producer_timestamp) * 1000
        else:
            self.last_latency = 0.0

        # Wait for slot to be ready (CPU-blocking poll + event query)
        if self.debug:
            _wait_t = time.perf_counter()
        try:
            self._wait_for_slot(read_slot)
        except TimeoutError:
            logger.error("Producer timeout — returning None")
            return None
        if self.debug:
            self.total_wait_event_time += (time.perf_counter() - _wait_t) * 1_000_000

        # D2H copy via CUDA memcpy (inherently not zero-copy)
        height, width, channels = self.shape
        itemsize = self._dtype_itemsize()
        nbytes = height * width * channels * itemsize

        # Pre-allocate numpy buffer (reuse across frames to avoid allocation overhead)
        if (
            self._numpy_buffer is None
            or self._numpy_buffer.shape != self.shape
            or self._numpy_buffer.dtype != self._numpy_dtype()
        ):
            # Free previous pinned allocation if shape or dtype changed
            if self._pinned_ptr is not None:
                try:
                    self.cuda.free_host(self._pinned_ptr)
                except (RuntimeError, OSError) as e:
                    logger.debug("free_host failed during reshape: %s", e)
                self._pinned_ptr = None

            try:
                self._pinned_ptr = self.cuda.malloc_host(nbytes)
                buf = (ctypes.c_ubyte * nbytes).from_address(self._pinned_ptr.value)
                self._numpy_buffer = np.frombuffer(buf, dtype=self._numpy_dtype()).reshape(self.shape)
                logger.debug("Allocated pinned numpy buffer: %s, %s", self.shape, self.dtype)
            except (RuntimeError, OSError) as e:
                logger.debug("cudaMallocHost failed — falling back to pageable memory: %s", e)
                self._numpy_buffer = np.empty(self.shape, dtype=self._numpy_dtype())

        # Async D2H copy on dedicated stream + synchronize (CPU-blocking until copy completes)
        if self.debug:
            _d2h_t = time.perf_counter()
        self.cuda.memcpy_async(
            dst=ctypes.c_void_p(self._numpy_buffer.ctypes.data),
            src=self.dev_ptrs[read_slot],
            count=nbytes,
            kind=2,  # cudaMemcpyDeviceToHost
            stream=self._numpy_stream,
        )
        self.cuda.stream_synchronize(self._numpy_stream)
        if self.debug:
            d2h_time = (time.perf_counter() - _d2h_t) * 1_000_000

        # Frame tracking
        self.frame_count += 1

        if self.debug:
            frame_time = (time.perf_counter() - frame_start) * 1_000_000
            self.total_get_frame_time += frame_time

            if self.frame_count % 97 == 0:
                n = self.frame_count
                logger.debug(
                    "Frame %d (numpy): shm_read=%.1fus wait=%.1fus d2h=%.1fus total=%.1fus latency=%.2fms",
                    n,
                    self.total_shm_read_us / n,
                    self.total_wait_event_time / n,
                    d2h_time,  # last frame only (not accumulated)
                    self.total_get_frame_time / n,
                    self.last_latency,
                )

        # Return pre-allocated buffer (NOTE: caller must not hold reference across frames)
        return self._numpy_buffer

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

        # Check for producer shutdown + version change + read slot (all SharedMemory reads)
        try:
            if self.shm_handle.buf[self._shutdown_offset] == 1:
                logger.info("Producer shutdown detected - cleaning up gracefully")
                self.cleanup()
                return None
        except (OSError, BufferError) as e:
            logger.debug("Could not read shutdown flag: %s", e)

        current_version = struct.unpack_from("<Q", self.shm_handle.buf, VERSION_OFFSET)[0]
        if current_version != self.ipc_version:
            logger.debug("TD re-initialized (v%d -> v%d), reopening IPC handle...", self.ipc_version, current_version)
            self._reinitialize()

        read_slot = self._get_read_slot()
        if read_slot is None:
            return None  # No new frame available

        producer_timestamp = struct.unpack_from("<d", self.shm_handle.buf, self._timestamp_offset)[0]
        if producer_timestamp > 0:  # Will be 0.0 on first frame before sender writes
            self.last_latency = (time.perf_counter() - producer_timestamp) * 1000
        else:
            self.last_latency = 0.0

        # Stream-ordered wait using CuPy's stream system
        if stream is None:
            stream = cp.cuda.get_current_stream()
        else:
            # Resolve stream pointer if torch.cuda.Stream or raw int
            if not isinstance(stream, cp.cuda.Stream):
                cuda_stream_ptr = self._resolve_stream(stream)
                # Wrap raw pointer as CuPy stream for consistency
                stream = cp.cuda.ExternalStream(cuda_stream_ptr)

        # Issue stream wait event (GPU-side, non-blocking to CPU)
        if self.ipc_events[read_slot]:
            cp.cuda.runtime.streamWaitEvent(stream.ptr, int(self.ipc_events[read_slot]), 0)
        # Note: No fallback needed - CuPy operations will naturally wait on stream

        # Return pre-created zero-copy CuPy array for this slot
        return self.cupy_arrays[read_slot]

    def _reinitialize(self) -> None:
        """Re-open all IPC handles after TD re-initialization."""

        # Close old handles
        for slot, dev_ptr in enumerate(self.dev_ptrs):
            if dev_ptr is not None:
                try:
                    self.cuda.ipc_close_mem_handle(dev_ptr)
                    logger.debug("Closed old IPC handle for slot %d", slot)
                except (RuntimeError, OSError) as e:
                    logger.warning("Error closing slot %d: %s", slot, e)

        # Destroy old events before re-opening (fix per NVIDIA simpleIPC)
        for slot, event in enumerate(self.ipc_events):
            if event is not None:
                try:
                    self.cuda.destroy_event(event)
                    logger.debug("Destroyed old IPC event for slot %d", slot)
                except (RuntimeError, OSError) as e:
                    logger.warning("Error destroying event for slot %d: %s", slot, e)

        # Read new version and num_slots
        self.ipc_version = struct.unpack(
            "<Q", bytes(self.shm_handle.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE])
        )[0]
        self.num_slots = struct.unpack(
            "<I", bytes(self.shm_handle.buf[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + NUM_SLOTS_SIZE])
        )[0]

        # Recompute cached offsets (num_slots may have changed)
        self._shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
        self._timestamp_offset = self._shutdown_offset + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

        # Re-read metadata (shape/dtype may have changed since last init)
        metadata_offset = self._shutdown_offset + SHUTDOWN_FLAG_SIZE
        try:
            width = struct.unpack("<I", bytes(self.shm_handle.buf[metadata_offset : metadata_offset + 4]))[0]
            height = struct.unpack("<I", bytes(self.shm_handle.buf[metadata_offset + 4 : metadata_offset + 8]))[0]
            num_comps = struct.unpack("<I", bytes(self.shm_handle.buf[metadata_offset + 8 : metadata_offset + 12]))[0]
            dtype_code = struct.unpack("<I", bytes(self.shm_handle.buf[metadata_offset + 12 : metadata_offset + 16]))[0]
            if width > 0 and height > 0 and num_comps > 0:
                new_shape = (height, width, num_comps)
                dtype_map = {0: "float32", 1: "float16", 2: "uint8", 3: "uint16"}
                new_dtype = dtype_map.get(dtype_code, "float32")
                if new_shape != self.shape or new_dtype != self.dtype:
                    logger.info(
                        "Metadata changed on reinit: %s %s -> %s %s",
                        self.shape,
                        self.dtype,
                        new_shape,
                        new_dtype,
                    )
                    self.shape = new_shape
                    self.dtype = new_dtype
                    # Free pinned allocation before invalidating pointer (avoids memory leak)
                    if self._pinned_ptr is not None:
                        try:
                            self.cuda.free_host(self._pinned_ptr)
                        except (RuntimeError, OSError) as e:
                            logger.debug("free_host failed during reinit: %s", e)
                    self._pinned_ptr = None
                    self._numpy_buffer = None  # Force reallocation on next get_frame_numpy()
        except (struct.error, ValueError, IndexError) as e:
            logger.debug("Could not re-read metadata during reinit: %s", e)

        # Reinitialize arrays
        self.ipc_handles = [None] * self.num_slots
        self.dev_ptrs = [None] * self.num_slots
        self.ipc_events = [None] * self.num_slots
        self.tensors = [None] * self.num_slots
        self._wrappers = [None] * self.num_slots

        # Reopen all handles
        for slot in range(self.num_slots):
            base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

            # Read memory handle (64 bytes)
            mem_handle_bytes = bytes(self.shm_handle.buf[base_offset : base_offset + 64])
            self.ipc_handles[slot] = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)

            # Open IPC memory handle
            self.dev_ptrs[slot] = self.cuda.ipc_open_mem_handle(self.ipc_handles[slot], flags=1)

            # Read event handle (64 bytes)
            event_handle_bytes = bytes(self.shm_handle.buf[base_offset + 64 : base_offset + 128])
            if any(event_handle_bytes):
                try:
                    ipc_event_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                    self.ipc_events[slot] = self.cuda.ipc_open_event_handle(ipc_event_handle)
                except (RuntimeError, OSError) as e:
                    logger.debug("Failed to open IPC event for slot %d: %s", slot, e)
                    self.ipc_events[slot] = None

            # Create tensor view for this slot if torch available
            if TORCH_AVAILABLE:
                self.tensors[slot] = self._create_tensor_view(slot)

        logger.debug("Reopened %d IPC handles v%d", self.num_slots, self.ipc_version)
        for slot in range(self.num_slots):
            logger.debug("Slot %d: GPU at 0x%016x", slot, self.dev_ptrs[slot].value)

    def cleanup(self) -> None:
        """Cleanup CUDA IPC resources."""
        # Close all IPC handles
        if hasattr(self, "dev_ptrs") and self.cuda:
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr is not None:
                    try:
                        self.cuda.ipc_close_mem_handle(dev_ptr)
                        logger.info("Closed IPC handle for slot %d", slot)
                    except (RuntimeError, OSError) as e:
                        logger.error("Error closing IPC handle for slot %d: %s", slot, e)

        # Destroy all IPC events (fix per NVIDIA simpleIPC: even opened handles consume resources)
        if hasattr(self, "ipc_events") and self.cuda is not None:
            for slot, event in enumerate(self.ipc_events):
                if event is not None:
                    try:
                        self.cuda.destroy_event(event)
                        logger.info("Destroyed IPC event for slot %d", slot)
                    except (RuntimeError, OSError) as e:
                        logger.error("Error destroying event for slot %d: %s", slot, e)

        # Free pinned host memory
        if hasattr(self, "_pinned_ptr") and self._pinned_ptr is not None and self.cuda:
            try:
                self.cuda.free_host(self._pinned_ptr)
                logger.debug("Freed pinned numpy buffer")
            except (RuntimeError, OSError) as e:
                logger.debug("free_host skipped (context gone): %s", e)
            self._pinned_ptr = None
            self._numpy_buffer = None

        # Destroy numpy stream
        if hasattr(self, "_numpy_stream") and self.cuda:
            try:
                self.cuda.destroy_stream(self._numpy_stream)
                logger.debug("Destroyed numpy stream")
            except (RuntimeError, OSError) as e:
                # Expected when producer has already cleaned up its CUDA context
                # (e.g. error 400: invalid resource handle in cross-process IPC teardown)
                logger.debug("numpy stream destroy skipped (context gone): %s", e)

        # Close SharedMemory
        if self.shm_handle is not None:
            try:
                self.shm_handle.close()
                # Note: Don't unlink - TouchDesigner owns it
                logger.info("Closed SharedMemory")
            except (OSError, BufferError) as e:
                logger.error("Error closing SharedMemory: %s", e)

        # Clear all state arrays (fix stale reference accumulation)
        self.tensors = []
        self._wrappers = []
        self.dev_ptrs = []
        self.ipc_handles = []
        self.ipc_events = []

        self._initialized = False
        logger.info("Cleanup complete")

    def __del__(self) -> None:
        """Destructor - cleanup on garbage collection."""
        if self._initialized:
            self.cleanup()

    def __enter__(self) -> CUDAIPCImporter:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager — cleanup resources."""
        self.cleanup()
        return None  # Don't suppress exceptions

    def is_ready(self) -> bool:
        """Check if importer is ready for frame access.

        Returns:
            True if initialized and ready, False otherwise
        """
        return self._initialized and len(self.dev_ptrs) > 0 and all(ptr is not None for ptr in self.dev_ptrs)

    def get_stats(self) -> dict[str, object]:
        """Get importer statistics.

        Returns:
            Dictionary with importer stats
        """
        return {
            "initialized": self._initialized,
            "shape": self.shape,
            "dtype": self.dtype,
            "frame_count": self.frame_count,
            "shm_name": self.shm_name,
            "num_slots": self.num_slots,
            "torch_available": TORCH_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "dev_ptrs": (
                [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self.dev_ptrs] if hasattr(self, "dev_ptrs") else []
            ),
            "tensor_device": (
                str(self.tensors[0].device)
                if TORCH_AVAILABLE and self.tensors and self.tensors[0] is not None
                else "N/A"
            ),
        }
