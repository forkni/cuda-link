"""
TDReceiver - Receiver engine for CUDAIPCExtension.

Owns all Receiver-mode CUDA IPC resources: SHM attachment, IPC handle opening,
per-frame GPU event sync, and Script TOP copyCUDAMemory calls.

textDAT name: TDReceiver  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import contextlib
import struct
import time
import traceback
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable

try:
    import numpy
except ImportError:
    numpy = None  # Will be imported at runtime in TD

from CUDAIPCWrapper import (  # noqa: E402
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
    get_cuda_runtime,
)
from TDConfig import TDSenderConfig  # noqa: E402
from TDHost import RealTOPHandle, TDHost  # noqa: E402

# Protocol layout constants (mirrors CUDAIPCExtension.py - both must stay in sync)
PROTOCOL_MAGIC = 0x43495044
MAGIC_OFFSET = 0
MAGIC_SIZE = 4
VERSION_OFFSET = 4
VERSION_SIZE = 8
NUM_SLOTS_OFFSET = 12
NUM_SLOTS_SIZE = 4
WRITE_IDX_OFFSET = 16
WRITE_IDX_SIZE = 4
SHM_HEADER_SIZE = 20
SLOT_SIZE = 128
SHUTDOWN_FLAG_SIZE = 1
METADATA_SIZE = 20
TIMESTAMP_SIZE = 8

FORMAT_KIND_SIGNED = 0
FORMAT_KIND_UNSIGNED = 1
FORMAT_KIND_FLOAT = 2
FLAGS_BFLOAT16 = 0x0001

_ST_U32 = struct.Struct("<I")
_ST_U64 = struct.Struct("<Q")
_ST_F64 = struct.Struct("<d")
_ST_BBH = struct.Struct("<BBH")

# CuPy import deferred (heavy; only needed for float16 receiver path)
CUPY_AVAILABLE: bool = False
cp = None


class TDReceiverEngine:
    """Receiver-mode engine: owns all GPU/SHM resources for the Receiver path.

    Constructed by the CUDAIPCExtension facade and replaced (not mutated) on
    mode switches - guaranteeing zero state leak between Sender and Receiver.
    """

    def __init__(
        self,
        host: TDHost,
        config: TDSenderConfig,
        cuda: Any,
        log_fn: Callable,
        num_slots: int,
        device: int,
        shm_name: str,
        verbose: bool,
    ) -> None:
        self._host = host
        self._config = config
        self.cuda = cuda
        self._log = log_fn
        self.num_slots = num_slots
        self.device = device
        self.shm_name = shm_name
        self.verbose_performance = verbose

        self._initialized = False

        self._rx_dev_ptrs = [None] * self.num_slots
        self._rx_ipc_handles = [None] * self.num_slots
        self._rx_ipc_events = [None] * self.num_slots
        self._rx_ipc_version = 0
        self._rx_num_slots = 0
        self._rx_shutdown_offset = 0
        self._rx_width = 0
        self._rx_height = 0
        self._rx_num_comps = 0
        self._rx_format_kind = FORMAT_KIND_FLOAT
        self._rx_bits_per_comp = 32
        self._rx_flags = 0
        self._rx_buffer_size = 0
        self._rx_last_write_idx = 0
        self._rx_cached_shape = None

        self._rx_connect_attempts = 0
        self._rx_max_connect_attempts = 20
        self._rx_backoff_intervals = [1, 2, 4, 8, 16, 32, 64, 120]
        self._diag_frames_since_reinit: int = 0
        self._rx_retry_interval_frames = 1
        self._rx_frames_since_last_retry = 0
        self._rx_needs_resolution_update = False
        self._rx_f16_cpu_buf = None
        self._rx_f32_cpu_buf = None
        self._rx_f16_pinned_ptr = None
        self._rx_cupy_f32_buf = None
        self._rx_cupy_f16_views: list = []

        self.shm_handle = None
        self._rx_stream = None

        # frame_count mirrored from sender SHM - exposed for get_stats()
        self.frame_count = 0

    def is_ready(self) -> bool:
        """True when initialized and all GPU buffer slots are open."""
        return self._initialized and len(self._rx_dev_ptrs) > 0 and all(ptr is not None for ptr in self._rx_dev_ptrs)

    def get_stats(self) -> dict:
        """Receiver statistics dict."""
        return {
            "mode": "Receiver",
            "initialized": self._initialized,
            "frame_count": self.frame_count,
            "shm_name": self.shm_name,
            "num_slots": self.num_slots,
            "rx_resolution": f"{self._rx_width}x{self._rx_height}x{self._rx_num_comps}"
            if self._rx_width > 0
            else "N/A",
            "rx_buffer_size_mb": self._rx_buffer_size / 1024 / 1024 if self._rx_buffer_size > 0 else 0,
            "rx_last_write_idx": self._rx_last_write_idx,
            "rx_dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self._rx_dev_ptrs],
        }

    def import_frame(self, import_buffer: TOP) -> bool:
        """Import frame from CUDA IPC into ImportBuffer (Script TOP).

        Can be called from:
        - Inside ImportBuffer's onCook callback (TD 2023+ compatibility)
        - Execute DAT onFrameStart with modoutsidecook enabled (TD 2025+)

        Args:
            import_buffer: The ImportBuffer Script TOP operator

        Returns:
            True if import successful, False otherwise.
        """
        # Check Active parameter (hot path via TDHost.is_active())
        if not self._host.is_active():
            return False

        # Lazy initialization with exponential backoff retry logic
        if not self._initialized:
            self._rx_frames_since_last_retry += 1
            if self._rx_frames_since_last_retry < self._rx_retry_interval_frames:
                return False  # Wait before retrying

            self._rx_frames_since_last_retry = 0
            self._rx_connect_attempts += 1

            if not self.initialize_receiver():
                backoff_idx = min(self._rx_connect_attempts, len(self._rx_backoff_intervals) - 1)
                self._rx_retry_interval_frames = self._rx_backoff_intervals[backoff_idx]
                if self._rx_connect_attempts <= self._rx_max_connect_attempts:
                    self._log(
                        f"Waiting for sender... (attempt {self._rx_connect_attempts}, "
                        f"next retry in {self._rx_retry_interval_frames} frames)"
                    )
                elif self._rx_connect_attempts == self._rx_max_connect_attempts + 1:
                    self._log("Sender not found. Will keep retrying silently.", force=True)
                return False

        # Wrap import_buffer for TOPHandle-style access (Phase 3 will pass handle directly)
        _ib = RealTOPHandle(import_buffer) if import_buffer is not None else None

        try:
            # Check for shutdown signal
            if self.shm_handle.buf[self._rx_shutdown_offset] == 1:
                self._log("Sender shutdown detected. Cleaning up.", force=True)
                self.cleanup_receiver()
                return False

            # Check for version change (sender re-initialized)
            current_version = _ST_U64.unpack_from(self.shm_handle.buf, VERSION_OFFSET)[0]
            if current_version != self._rx_ipc_version:
                self._log(
                    f"Sender re-initialized (v{self._rx_ipc_version} -> v{current_version}). Reconnecting...",
                    force=True,
                )
                self.cleanup_receiver()
                return False  # Will reinitialize on next frame

            # Read write_idx and calculate read slot
            write_idx = _ST_U32.unpack_from(self.shm_handle.buf, WRITE_IDX_OFFSET)[0]
            if write_idx == 0:
                return False  # No frames written yet

            read_slot = (write_idx - 1) % self._rx_num_slots

            _diag = self._diag_frames_since_reinit < 5
            _t_event = _t_copy = 0.0  # pre-init for static analyzers; only read when _diag is True
            if _diag:
                self._diag_frames_since_reinit += 1
                _t_event = time.perf_counter()

            # Wait on IPC event for this slot (stream-ordered, non-blocking to CPU)
            if self._rx_ipc_events[read_slot]:
                self.cuda.stream_wait_event(
                    self._rx_stream,
                    self._rx_ipc_events[read_slot],
                    0,
                )
            else:
                # Fallback when no IPC event: drain the stream now.
                # Note: float16 path will call stream_synchronize again below, but
                # synchronizing an already-idle stream is a no-op in CUDA.
                self.cuda.stream_synchronize(self._rx_stream)

            if _diag:
                _event_ms = (time.perf_counter() - _t_event) * 1000.0
                _t_copy = time.perf_counter()

            # Copy CUDA memory into ImportBuffer texture using cached shape
            address = self._rx_dev_ptrs[read_slot].value

            if (
                self._rx_format_kind == FORMAT_KIND_FLOAT
                and self._rx_bits_per_comp == 16
                and not (self._rx_flags & FLAGS_BFLOAT16)
            ):
                if CUPY_AVAILABLE and self._rx_cupy_f32_buf is not None:
                    # GPU-side float16→float32 conversion (Ch5: minimize PCIe traffic).
                    # stream_wait_event (enqueued above on _rx_stream) guarantees GPU data is ready.
                    # We create a zero-copy CuPy view of the IPC pointer, run an elementwise
                    # f16→f32 cast entirely on GPU via ExternalStream, then call copyCUDAMemory —
                    # eliminating two PCIe roundtrips and the CPU numpy.copyto call.
                    rx_stream_int = int(self._rx_stream.value)
                    f16_size = self._rx_buffer_size  # original float16 byte count
                    f32_size = f16_size * 2  # float32 = 2× bytes

                    cupy_f16 = self._rx_cupy_f16_views[read_slot]
                    # Run conversion on _rx_stream so copyCUDAMemory (also on _rx_stream)
                    # automatically serializes after the elementwise cast kernel.
                    with cp.cuda.ExternalStream(rx_stream_int):
                        cp.copyto(self._rx_cupy_f32_buf, cupy_f16, casting="same_kind")

                    _ib.copy_cuda_memory(
                        self._rx_cupy_f32_buf.data.ptr,
                        f32_size,
                        self._rx_cached_shape,  # dataType=float32 set during initialize_receiver()
                        stream=rx_stream_int,
                    )
                else:
                    # CPU fallback: D2H + numpy convert + copyNumpyArray.
                    # Used when CuPy is not installed or GPU buffer allocation failed.
                    if self._rx_f16_cpu_buf is None or self._rx_f32_cpu_buf is None:
                        debug("[CUDAIPCLink] float16 CPU buffers not allocated — skipping frame")
                        return False

                    # D2H on _rx_stream: stream_wait_event (enqueued earlier) guarantees data is ready.
                    cpu_ptr = self._rx_f16_cpu_buf.ctypes.data_as(c_void_p)
                    self.cuda.memcpy_async(cpu_ptr, c_void_p(address), self._rx_buffer_size, 2, self._rx_stream)
                    self.cuda.stream_synchronize(self._rx_stream)
                    numpy.copyto(
                        self._rx_f32_cpu_buf,
                        self._rx_f16_cpu_buf.reshape(self._rx_height, self._rx_width, self._rx_num_comps),
                        casting="same_kind",
                    )
                    _ib.copy_numpy_array(self._rx_f32_cpu_buf)
            else:
                _ib.copy_cuda_memory(
                    address,
                    self._rx_buffer_size,
                    self._rx_cached_shape,
                    stream=int(self._rx_stream.value),
                )

            if _diag:
                _copy_ms = (time.perf_counter() - _t_copy) * 1000.0
                self._log(
                    f"[DIAG] import_frame #{self._diag_frames_since_reinit}: "
                    f"slot={read_slot} write_idx={write_idx} addr=0x{address:x} "
                    f"stream_wait={_event_ms:.2f}ms copyCUDAMemory={_copy_ms:.2f}ms",
                    force=True,
                )

            self.frame_count += 1
            self._rx_last_write_idx = write_idx

            # Debug logging (97 = prime, avoids aliasing with slot counts 2,4,5)
            if self.verbose_performance and self.frame_count % 97 == 0:
                self._log(f"Frame {self.frame_count}: read_slot={read_slot}, write_idx={write_idx}")

            return True

        except (RuntimeError, OSError) as e:
            self._log(f"Import failed: {e}", force=True)

            traceback.print_exc()
            return False

    def update_receiver_resolution(self, import_buffer: TOP) -> bool:
        """Update ImportBuffer resolution from outside the cook cycle.

        Safe to call from Execute DAT when modoutsidecook is enabled on the Script TOP (TD 2025+).
        When modoutsidecook is NOT available, this is a no-op (resolution handled in onCook).

        Args:
            import_buffer: The ImportBuffer Script TOP operator

        Returns:
            True if resolution was updated, False if no update needed or not applicable
        """
        if not self._rx_needs_resolution_update:
            return False

        try:
            RealTOPHandle(import_buffer).set_resolution(self._rx_width, self._rx_height)
            self._rx_needs_resolution_update = False
            self._log(
                f"Set ImportBuffer resolution to {self._rx_width}x{self._rx_height} (from Execute DAT)",
                force=True,
            )
            return True
        except (AttributeError, RuntimeError) as e:
            self._log(f"Could not set ImportBuffer resolution: {e}", force=True)
            return False

    def initialize_receiver(self) -> bool:
        """Initialize receiver: open SharedMemory, read handles, open IPC handles.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        # Numslots is always disabled in Receiver mode (sender controls slot count)
        self._host.set_param_enabled("Numslots", False)

        _t0 = time.perf_counter()
        try:
            self.cuda = get_cuda_runtime(device=self.device)
            self._log(f"Loaded CUDA runtime on device {self.cuda.get_device()}", force=True)

            # Open SharedMemory (sender must have created it)
            try:
                self.shm_handle = SharedMemory(name=self.shm_name)
            except FileNotFoundError:
                self._log(f"SharedMemory '{self.shm_name}' not found. Sender not ready?")
                return False

            # Validate protocol magic number (new in this version)
            try:
                magic = struct.unpack(
                    "<I",
                    bytes(self.shm_handle.buf[MAGIC_OFFSET : MAGIC_OFFSET + MAGIC_SIZE]),
                )[0]
                if magic != PROTOCOL_MAGIC:
                    self._log(
                        f"Protocol magic mismatch: expected 0x{PROTOCOL_MAGIC:08X}, got 0x{magic:08X}. "
                        "Sender using incompatible protocol version.",
                        force=True,
                    )
                    self.shm_handle.close()
                    self.shm_handle = None
                    return False
            except (struct.error, ValueError, IndexError):
                self._log(
                    "Cannot read protocol magic. Sender may be using old protocol version.",
                    force=True,
                )
                self.shm_handle.close()
                self.shm_handle = None
                return False

            # Read header
            self._rx_ipc_version = struct.unpack(
                "<Q",
                bytes(self.shm_handle.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE]),
            )[0]
            self._rx_num_slots = struct.unpack(
                "<I",
                bytes(self.shm_handle.buf[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + NUM_SLOTS_SIZE]),
            )[0]

            if self._rx_num_slots == 0 or self._rx_num_slots > 10:
                self._log(
                    f"Invalid num_slots: {self._rx_num_slots}. Protocol error.",
                    force=True,
                )
                self.shm_handle.close()
                self.shm_handle = None
                return False

            # Sync UI parameter to show sender's slot count (informational only).
            # Do NOT set self.num_slots — that's the sender-specific working value.
            # Receiver always uses self._rx_num_slots for its own arrays.
            self._host.set_param_value("Numslots", self._rx_num_slots)

            # Cache receiver shutdown offset once — avoids per-frame arithmetic in import_frame()
            self._rx_shutdown_offset = SHM_HEADER_SIZE + (self._rx_num_slots * SLOT_SIZE)

            # Read extended metadata (if available)
            shutdown_offset = self._rx_shutdown_offset
            metadata_offset = shutdown_offset + SHUTDOWN_FLAG_SIZE

            # Check if SharedMemory is large enough for metadata
            if len(self.shm_handle.buf) >= metadata_offset + METADATA_SIZE:
                self._rx_width = struct.unpack(
                    "<I",
                    bytes(self.shm_handle.buf[metadata_offset : metadata_offset + 4]),
                )[0]
                self._rx_height = struct.unpack(
                    "<I",
                    bytes(self.shm_handle.buf[metadata_offset + 4 : metadata_offset + 8]),
                )[0]
                self._rx_num_comps = struct.unpack(
                    "<I",
                    bytes(self.shm_handle.buf[metadata_offset + 8 : metadata_offset + 12]),
                )[0]
                self._rx_format_kind, self._rx_bits_per_comp, self._rx_flags = _ST_BBH.unpack(
                    bytes(self.shm_handle.buf[metadata_offset + 12 : metadata_offset + 16])
                )
                self._rx_buffer_size = struct.unpack(
                    "<I",
                    bytes(self.shm_handle.buf[metadata_offset + 16 : metadata_offset + 20]),
                )[0]
                self._log(
                    f"Read metadata: {self._rx_width}x{self._rx_height}x{self._rx_num_comps}, "
                    f"kind={self._rx_format_kind} bits={self._rx_bits_per_comp} flags=0x{self._rx_flags:04x}, "
                    f"buf_size={self._rx_buffer_size}",
                    force=True,
                )
                # Strict size invariant: data_size must exactly equal W*H*C*(bits/8).
                # Any mismatch means the sender's encoding is inconsistent — refuse to init.
                expected_size = self._rx_width * self._rx_height * self._rx_num_comps * (self._rx_bits_per_comp // 8)
                if self._rx_bits_per_comp == 0 or self._rx_buffer_size != expected_size:
                    self._log(
                        f"Metadata size invariant failed: W*H*C*(bits/8)={expected_size} "
                        f"but buf_size={self._rx_buffer_size}. Sender/receiver protocol mismatch.",
                        force=True,
                    )
                    self.shm_handle.close()
                    self.shm_handle = None
                    return False
            else:
                self._log("No extended metadata in SharedMemory (legacy sender)", force=True)
                self.shm_handle.close()
                self.shm_handle = None
                return False  # Cannot proceed without knowing dimensions

            # Validate metadata
            if self._rx_width == 0 or self._rx_height == 0 or self._rx_buffer_size == 0:
                self._log(
                    "Metadata contains zeros - sender may not have written frame yet",
                    force=True,
                )
                self.shm_handle.close()
                self.shm_handle = None
                return False

            # Check for shutdown signal BEFORE opening IPC handles.
            # A stale SharedMemory (producer exited cleanly) will have shutdown_flag=1
            # and its IPC handles reference freed GPU memory → cudaIpcOpenMemHandle error 201.
            # Mirrors CUDAIPCImporter._initialize() pattern.
            try:
                if self.shm_handle.buf[shutdown_offset] == 1:
                    self._log(
                        "Shutdown flag is set — producer has exited. "
                        "SharedMemory contains stale IPC handles. Will retry.",
                        force=True,
                    )
                    self.shm_handle.close()
                    self.shm_handle = None
                    return False
            except (OSError, BufferError, IndexError) as e:
                self._log(f"Could not read shutdown flag: {e}", force=True)
                self.shm_handle.close()
                self.shm_handle = None
                return False

            # Log write_idx for diagnostics (0 = no frames sent yet, handles still valid)
            try:
                write_idx_diag = struct.unpack_from("<I", self.shm_handle.buf, WRITE_IDX_OFFSET)[0]
                self._log(f"Producer write_idx={write_idx_diag} (0 = no frames sent yet)", force=True)
            except (struct.error, ValueError):
                pass

            # Initialize arrays
            self._rx_dev_ptrs = [None] * self._rx_num_slots
            self._rx_ipc_handles = [None] * self._rx_num_slots
            self._rx_ipc_events = [None] * self._rx_num_slots

            # Create dedicated non-blocking stream for receiver IPC operations
            # MUST happen before ipc_open_mem_handle to establish CUDA context
            # Reuse existing stream on re-init to avoid leaks on reconnection cycles
            if not hasattr(self, "_rx_stream") or self._rx_stream is None:
                self._rx_stream = self.cuda.create_stream_with_priority(flags=0x01)
                self._log(
                    f"Created receiver stream: 0x{int(self._rx_stream.value):016x}",
                    force=True,
                )
            else:
                self._log(
                    f"Reusing receiver stream: 0x{int(self._rx_stream.value):016x}",
                    force=True,
                )

            # Open all IPC handles (per slot)
            for slot in range(self._rx_num_slots):
                base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

                # Read + open memory handle
                mem_handle_bytes = bytes(self.shm_handle.buf[base_offset : base_offset + 64])

                # Fix #1: Validate handle is non-zero before opening.
                # All-zero bytes mean sender wrote metadata but hasn't written IPC handles yet
                # (race condition: metadata and handles are two separate writes).
                if not any(mem_handle_bytes):
                    self._log(
                        f"Slot {slot}: IPC mem handle is all zeros - "
                        "sender hasn't written handles yet. Will retry with backoff.",
                        force=True,
                    )
                    self._cleanup_partial_receiver(slot)
                    return False

                self._rx_ipc_handles[slot] = cudaIpcMemHandle_t.from_buffer_copy(mem_handle_bytes)

                # Fix #2: Wrap ipc_open_mem_handle in try/except with diagnostic logging.
                # Error 201 (INVALID_CONTEXT) means the GPU memory was freed by the sender
                # (process exited/crashed) or the handle references a different CUDA device.
                try:
                    self._rx_dev_ptrs[slot] = self.cuda.ipc_open_mem_handle(self._rx_ipc_handles[slot], flags=1)
                except RuntimeError as e:
                    self._log(
                        f"Slot {slot}: cudaIpcOpenMemHandle failed: {e}. "
                        "Possible causes: sender process exited, GPU memory freed, "
                        "or CUDA device mismatch. Will retry with backoff.",
                        force=True,
                    )
                    self._cleanup_partial_receiver(slot)
                    return False

                # Read + open event handle
                event_handle_bytes = bytes(self.shm_handle.buf[base_offset + 64 : base_offset + 128])
                if any(event_handle_bytes):
                    try:
                        ipc_event_handle = cudaIpcEventHandle_t.from_buffer_copy(event_handle_bytes)
                        self._rx_ipc_events[slot] = self.cuda.ipc_open_event_handle(ipc_event_handle)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Failed to open IPC event for slot {slot}: {e}")
                        self._rx_ipc_events[slot] = None

                self._log(
                    f"Opened slot {slot}: GPU at 0x{self._rx_dev_ptrs[slot].value:016x}, "
                    f"event={'YES' if self._rx_ipc_events[slot] else 'NO'}",
                    force=True,
                )

            # Flag that Script TOP resolution needs to be updated (will be done outside cook cycle)
            self._rx_needs_resolution_update = True

            # Cache CUDAMemoryShape to avoid per-frame object creation
            if numpy is None:
                import numpy as np_module
            else:
                np_module = numpy

            # Decode numpy dtype directly from (format_kind, bits_per_comp, flags).
            # float16 path uses float32 as the CUDAMemoryShape dtype because copyCUDAMemory
            # doesn't accept float16 — the actual conversion happens via D2H + copyNumpyArray.
            _is_float16 = (
                self._rx_format_kind == FORMAT_KIND_FLOAT
                and self._rx_bits_per_comp == 16
                and not (self._rx_flags & FLAGS_BFLOAT16)
            )
            if self._rx_format_kind == FORMAT_KIND_UNSIGNED and self._rx_bits_per_comp == 8:
                np_dtype = np_module.uint8
            elif self._rx_format_kind == FORMAT_KIND_UNSIGNED and self._rx_bits_per_comp == 16:
                np_dtype = np_module.uint16
            elif _is_float16:
                np_dtype = np_module.float32  # shape dtype for copyCUDAMemory; real f16 handled via D2H
            else:
                np_dtype = np_module.float32  # float32/float64 — TD's copyCUDAMemory expects float32 shape

            # float16: allocate CPU buffers for D2H conversion (copyCUDAMemory doesn't support float16)
            if _is_float16:
                n_elems = self._rx_width * self._rx_height * self._rx_num_comps
                f16_bytes = n_elems * 2
                # Use pinned (page-locked) memory for DMA-capable D2H transfer via cudaMemcpyAsync.
                # cudaMemcpyAsync on pageable memory falls back to synchronous behavior per CUDA spec.
                self._rx_f16_pinned_ptr = None
                try:
                    import ctypes as _ctypes

                    self._rx_f16_pinned_ptr = self.cuda.malloc_host(f16_bytes)
                    _buf = (_ctypes.c_ubyte * f16_bytes).from_address(self._rx_f16_pinned_ptr.value)
                    self._rx_f16_cpu_buf = np_module.frombuffer(_buf, dtype=np_module.float16)
                    self._log("float16 receiver: allocated pinned CPU buffer for D2H (async path)", force=True)
                except (RuntimeError, OSError) as _e:
                    # Fall back to pageable if pinned allocation fails (e.g. low memory)
                    self._rx_f16_pinned_ptr = None
                    self._rx_f16_cpu_buf = np_module.empty(n_elems, dtype=np_module.float16)
                    self._log(f"float16 receiver: pinned alloc failed ({_e}), using pageable buffer", force=True)
                # float32 output buffer stays pageable — it is the target for copyNumpyArray, not DMA
                self._rx_f32_cpu_buf = np_module.empty(
                    (self._rx_height, self._rx_width, self._rx_num_comps), dtype=np_module.float32
                )

                # GPU-side float32 staging buffer for CuPy conversion path (avoids per-frame allocation).
                # When CuPy is available, float16→float32 happens on GPU with zero PCIe traffic.
                # Lazy import: CuPy is heavy; defer until first receiver init to avoid TD startup penalty.
                global CUPY_AVAILABLE, cp
                if cp is None:
                    try:
                        import cupy as cp  # noqa: PLC0415

                        CUPY_AVAILABLE = True
                    except ImportError:
                        CUPY_AVAILABLE = False

                if CUPY_AVAILABLE:
                    try:
                        self._rx_cupy_f32_buf = cp.empty(
                            (self._rx_height, self._rx_width, self._rx_num_comps), dtype=cp.float32
                        )
                        self._log(
                            "float16 receiver: CuPy GPU float32 buffer allocated (GPU-side conversion path)", force=True
                        )
                        # Pre-create per-slot float16 views to avoid per-frame UnownedMemory allocation.
                        self._rx_cupy_f16_views = []
                        for _i in range(self._rx_num_slots):
                            _ptr = self._rx_dev_ptrs[_i].value
                            _mem = cp.cuda.UnownedMemory(_ptr, self._rx_buffer_size, owner=self)
                            _memptr = cp.cuda.MemoryPointer(_mem, 0)
                            self._rx_cupy_f16_views.append(
                                cp.ndarray(
                                    (self._rx_height, self._rx_width, self._rx_num_comps),
                                    dtype=cp.float16,
                                    memptr=_memptr,
                                )
                            )
                    except Exception as _e:
                        self._rx_cupy_f32_buf = None
                        self._rx_cupy_f16_views = []
                        self._log(
                            f"float16 receiver: CuPy GPU buffer alloc failed ({_e}), CPU fallback active", force=True
                        )

            self._rx_cached_shape = CUDAMemoryShape()
            self._rx_cached_shape.width = self._rx_width
            self._rx_cached_shape.height = self._rx_height
            self._rx_cached_shape.numComps = self._rx_num_comps
            self._rx_cached_shape.dataType = np_dtype

            self._initialized = True
            self._diag_frames_since_reinit = 0  # Reset so import_frame logs the next 5 calls
            _init_ms = (time.perf_counter() - _t0) * 1000.0
            self._log(
                f"Receiver initialized: {self._rx_num_slots} slots, "
                f"{self._rx_width}x{self._rx_height}x{self._rx_num_comps} "
                f"(init took {_init_ms:.1f} ms)",
                force=True,
            )
            return True

        except (OSError, RuntimeError, ValueError) as e:
            self._log(f"Receiver initialization failed: {e}", force=True)

            traceback.print_exc()
            return False

    def _cleanup_partial_receiver(self, failed_slot: int) -> None:
        """Cleanup partially-opened receiver resources when initialization fails mid-slot.

        Called when `initialize_receiver()` fails partway through slot iteration.
        Closes IPC handles already opened for slots 0..failed_slot-1 to prevent
        GPU resource leaks across backoff retries.

        Args:
            failed_slot: The slot index that failed (0-based). Cleans up slots 0..failed_slot-1.
        """
        for i in range(failed_slot):
            if self._rx_dev_ptrs[i] is not None:
                try:
                    self.cuda.ipc_close_mem_handle(self._rx_dev_ptrs[i])
                    self._log(f"Cleaned up partial slot {i} mem handle")
                except (RuntimeError, OSError):
                    pass
                self._rx_dev_ptrs[i] = None
            if self._rx_ipc_events[i] is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self.cuda.destroy_event(self._rx_ipc_events[i])
                self._rx_ipc_events[i] = None

        # Close SharedMemory so next retry re-opens fresh (avoids reading stale content)
        if self.shm_handle is not None:
            with contextlib.suppress(OSError, BufferError):
                self.shm_handle.close()
            self.shm_handle = None

    def cleanup(self) -> None:
        """Cleanup Receiver CUDA IPC resources."""
        # Guard against double-cleanup (matches cleanup_sender() pattern)
        if not self._initialized and self.shm_handle is None:
            return

        _cleanup_t0 = time.perf_counter()

        # Toggle ImportBuffer Script TOP bypass to force TD to release its CUDA↔D3D11
        # interop registration BEFORE we tear down the underlying CUDA resources.
        # Without this, on WDDM the IPC device pointers remain bound to a stale interop
        # mapping inside TD's Script TOP, and the driver-side cleanup of those pointers
        # (cudaIpcCloseMemHandle / cudaStreamDestroy) blocks for several seconds while
        # WDDM tries to drain pending work — exceeding the 2s TDR threshold.
        _bypass_ms = 0.0
        # ImportBuffer bypass toggle disabled for bisect testing (TR1).
        # Was toggling bypass=True/False on the Script TOP before tearing down CUDA
        # resources; suspected to cause re-activation instability on WDDM.

        # Close all IPC memory handles
        _close_ms = 0.0
        if hasattr(self, "_rx_dev_ptrs") and self.cuda:
            _close_t0 = time.perf_counter()
            for slot, dev_ptr in enumerate(self._rx_dev_ptrs):
                if dev_ptr:
                    _slot_t0 = time.perf_counter()
                    try:
                        self.cuda.ipc_close_mem_handle(dev_ptr)
                        _slot_ms = (time.perf_counter() - _slot_t0) * 1000.0
                        self._log(f"Closed IPC handle for slot {slot} ({_slot_ms:.1f} ms)")
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error closing IPC handle for slot {slot}: {e}", force=True)
            _close_ms = (time.perf_counter() - _close_t0) * 1000.0

        # Destroy all IPC events (fix per NVIDIA simpleIPC: even opened handles consume resources)
        _events_ms = 0.0
        if hasattr(self, "_rx_ipc_events") and self.cuda:
            _events_t0 = time.perf_counter()
            for slot, event in enumerate(self._rx_ipc_events):
                if event:
                    _slot_t0 = time.perf_counter()
                    try:
                        self.cuda.destroy_event(event)
                        _slot_ms = (time.perf_counter() - _slot_t0) * 1000.0
                        self._log(f"Destroyed IPC event for slot {slot} ({_slot_ms:.1f} ms)")
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error destroying event for slot {slot}: {e}", force=True)
            _events_ms = (time.perf_counter() - _events_t0) * 1000.0

        # Destroy receiver stream
        _stream_ms = 0.0
        if hasattr(self, "_rx_stream") and self._rx_stream and self.cuda:
            _stream_t0 = time.perf_counter()
            try:
                self.cuda.destroy_stream(self._rx_stream)
                _stream_ms = (time.perf_counter() - _stream_t0) * 1000.0
                self._log(f"Destroyed receiver stream ({_stream_ms:.1f} ms)", force=True)
            except (RuntimeError, OSError) as e:
                self._log(f"Error destroying receiver stream: {e}", force=True)

        # Close SharedMemory
        if self.shm_handle:
            try:
                self.shm_handle.close()
            except (OSError, BufferError) as e:
                self._log(f"Error closing SharedMemory: {e}", force=True)

        self.shm_handle = None
        self._rx_dev_ptrs = []
        self._rx_ipc_handles = []
        self._rx_ipc_events = []
        self._rx_num_slots = 0
        self._rx_stream = None  # Prevent double-free
        # Free pinned float16 D2H buffer if allocated
        if hasattr(self, "_rx_f16_pinned_ptr") and self._rx_f16_pinned_ptr is not None:
            try:
                self.cuda.free_host(self._rx_f16_pinned_ptr)
            except (RuntimeError, OSError) as e:
                self._log(f"free_host skipped (context gone): {e}")
            self._rx_f16_pinned_ptr = None
        self._rx_f16_cpu_buf = None
        self._rx_f32_cpu_buf = None
        self._rx_cupy_f32_buf = None  # CuPy memory pool handles GPU free on GC
        self._rx_cupy_f16_views = []
        self._initialized = False
        self._rx_connect_attempts = 0
        self._rx_frames_since_last_retry = 0
        _cleanup_ms = (time.perf_counter() - _cleanup_t0) * 1000.0
        self._log(
            f"Receiver cleanup complete (total {_cleanup_ms:.1f} ms, "
            f"bypass {_bypass_ms:.1f} ms, ipc_close {_close_ms:.1f} ms, "
            f"events {_events_ms:.1f} ms, stream {_stream_ms:.1f} ms)",
            force=True,
        )
