"""
CUDA IPC Extension for TouchDesigner - Dual-Mode Sender/Receiver
Supports both exporting (Sender) and importing (Receiver) GPU textures via CUDA IPC

Usage in TouchDesigner:
    Sender: ext.CUDAIPCExtension.export_frame(top_op)
    Receiver: ext.CUDAIPCExtension.import_frame(import_buffer)

Architecture:
    Sender: TD GPU -> cudaMemory() -> Persistent Buffer -> IPC Handle -> SharedMemory
    Receiver: SharedMemory -> IPC Handle -> Opened GPU Buffer -> scriptTOP.copyCUDAMemory()
"""

from __future__ import annotations

import contextlib
import os
import struct
import threading
import time
import traceback
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory

try:
    import numpy
except ImportError:
    numpy = None  # Will be imported at runtime in TD

# Defer CuPy import to initialize_receiver() — heavy import imposes TD startup penalty
CUPY_AVAILABLE: bool = False
cp = None

# Import types with fallbacks
try:
    from td import COMP, TOP, CUDAMemoryShape
except ImportError:
    from typing import Any as COMP
    from typing import Any as TOP

    CUDAMemoryShape = None  # Will be accessed as global in TD runtime

from CUDAIPCWrapper import (  # noqa: E402
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
    get_cuda_runtime,
)
from NVMLObserver import NVML_AVAILABLE, NVMLObserver  # noqa: E402

# Protocol layout constants (named offsets, not hardcoded literals)
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

# Data type codes for extended protocol
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_UINT8 = 2
DTYPE_UINT16 = 3

# Pre-compiled struct objects for hot-path SHM reads/writes (~50-100ns saved per call vs format-string lookup)
_ST_U32 = struct.Struct("<I")  # uint32 LE (write_idx, num_slots, metadata fields)
_ST_U64 = struct.Struct("<Q")  # uint64 LE (version)
_ST_F64 = struct.Struct("<d")  # float64 LE (timestamp)

# Pixel format substrings that TD 2025 (CUDA 12.8) rejects in cudaMemory().
# uint8 / uint16 (fixed) and float32 are supported. float16 variants are not.
_CUDA_UNSUPPORTED_PIXEL_FORMATS = {"16-bit float", "16float"}
_FMT_TRANSFORM_NAME = "dtype_converter"  # Permanent Transform TOP in network (before ExportBuffer)
_EXPORT_BUFFER_NAME = "ExportBuffer"  # Null TOP downstream of dtype_converter; cudaMemory() source

# C3: CPU release-fence — ensures shutdown_flag write is visible before write_idx publish.
_fence_lock = threading.Lock()


def _release_fence() -> None:
    with _fence_lock:
        pass


class CUDAIPCExtension:
    """TouchDesigner extension for dual-mode CUDA IPC texture sharing.

    Modes:
    - Sender: Export GPU textures to SharedMemory via IPC handles
    - Receiver: Import GPU textures from SharedMemory into Script TOP

    Sender Responsibilities:
    - Allocate persistent GPU buffer (once at startup)
    - Copy TOP texture to buffer each frame
    - Export IPC handle + metadata (once at startup)
    - Write to SharedMemory for consumer process

    Receiver Responsibilities:
    - Open SharedMemory and read IPC handles
    - Open CUDA IPC memory handles to access sender's GPU buffers
    - Per frame: wait on GPU event, call scriptTOP.copyCUDAMemory()

    Performance:
    - Sender per-frame: ~10-20μs within TD CUDA context; ~177-219μs in standalone Python (WDDM + stream_synchronize)
    - Receiver per-frame: < 5μs (event sync enqueue + copyCUDAMemory call)
    - Zero CPU memory copies (GPU-direct)
    """

    def __init__(self, ownerComp: COMP) -> None:
        """Initialize CUDA IPC extension (Sender or Receiver mode).

        Args:
            ownerComp: The component that owns this extension
        """
        self.ownerComp = ownerComp

        # Determine mode from parameter
        try:
            self._mode = str(ownerComp.par.Mode.eval())  # 'Sender' or 'Receiver'
        except AttributeError:
            self._mode = "Sender"  # Default to sender for backward compat

        # CUDA runtime API
        self.cuda = None
        self._initialized = False

        # Ring buffer configuration - read from parameter or default to 3
        try:
            self.num_slots = int(ownerComp.par.Numslots.eval())
        except (AttributeError, ValueError):
            self.num_slots = 3

        # CUDA device index - read from parameter or default to 0.
        # IPC handles are device-scoped; sender and receiver must use the same device.
        try:
            self.device = int(ownerComp.par.Cudadevice.eval())
        except (AttributeError, ValueError):
            self.device = 0

        # GPU buffer state (arrays for ring buffer)
        self.dev_ptrs = [None] * self.num_slots  # List of GPU buffer pointers
        self.buffer_size = 0  # Aligned allocation size (for cudaMalloc)
        self.data_size = 0  # Actual data size (for cudaMemcpy)
        self.width = 0
        self.height = 0
        self.channels = 4  # RGBA

        # IPC handles (arrays for ring buffer - created once, reused)
        self.ipc_handles = [None] * self.num_slots  # List of IPC mem handles

        # IPC events for GPU-side synchronization (per-slot)
        self.ipc_events = [None] * self.num_slots  # List of IPC events
        self.ipc_event_handles = [None] * self.num_slots  # List of IPC event handles

        # Deferred cleanup (cudaFree blocks on IPC memory until receiver closes handles)
        self._pending_free_ptrs = []  # GPU pointers queued for deferred free
        self._pending_free_events = []  # Events queued for deferred destroy
        self._deferred_free_at_frame = 0  # Frame at which to execute deferred free

        # Ring buffer write index (atomic counter)
        self.write_idx = 0  # Increments every frame, wraps via modulo

        # SharedMemory for IPC handle transfer
        self.shm_handle = None
        # Read SharedMemory name from dedicated parameter or use fallback
        try:
            self.shm_name = ownerComp.par.Ipcmemname.eval()
        except AttributeError:
            # Fallback to default name if parameter doesn't exist
            self.shm_name = "cudalink_output_ipc"

        # Cached SHM byte offsets — computed once in initialize() to avoid per-frame arithmetic
        self._shutdown_offset = 0  # SHM_HEADER_SIZE + num_slots * SLOT_SIZE
        self._ts_offset = 0  # _shutdown_offset + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

        # Frame tracking
        self.frame_count = 0

        # CRITICAL: Keep reference to CUDAMemory to prevent GC
        self.cuda_mem_ref = None

        # Conditional synchronization (CPU fallback)
        self.sync_interval = 10  # Sync every N frames (reduces GPU sync overhead)

        # CUDALINK_EXPORT_SYNC=0 (default): correctness is guaranteed by the receiver's
        # cudaStreamWaitEvent(ipc_events[slot]) — no CPU block needed. Set to "1" to restore
        # the pre-2026-04-23 blocking sync. Measured cost of "1": ~295 µs/frame dead CPU wait
        # (A/B/C diagnostic, SESSION_LOG 2026-04-23; Handbook p3/pg56 confirmed the stream is
        # idle at frame-end). Mirrors Python lib default "0".
        self._export_sync: bool = os.environ.get("CUDALINK_EXPORT_SYNC", "0") == "1"
        # CUDALINK_EXPORT_PROFILE=1: enables fine-grained per-region sub-timers in export_frame.
        # Zero overhead when unset (single predicted-false branch per region).
        self._export_profile: bool = os.environ.get("CUDALINK_EXPORT_PROFILE", "0") == "1"
        # CUDALINK_EXPORT_FLUSH_PROBE=1: calls cudaStreamQuery after record_event when
        # _export_sync=False. Per CUDA Handbook p3/pg56, this forces WDDM-deferred commands
        # to submit without blocking — used to confirm the WDDM batching hypothesis.
        self._export_flush_probe: bool = os.environ.get("CUDALINK_EXPORT_FLUSH_PROBE", "0") == "1"
        self._nvml_observer: NVMLObserver | None = None

        # Performance metrics
        self.total_memcpy_time = 0.0
        self.total_record_event_time = 0.0
        self.total_export_time = 0.0
        self.total_cuda_memory_time = 0.0  # Time spent in cudaMemory() call
        # Profile accumulators (active only when CUDALINK_EXPORT_PROFILE=1)
        self.total_pre_interop_us: float = 0.0  # frame_start → just before cudaMemory()
        self.total_post_interop_us: float = 0.0  # cudaMemory() done → just before memcpy_async
        self.total_sync_us: float = 0.0  # cudaStreamSynchronize (when _export_sync)
        self.total_sticky_check_us: float = 0.0  # cudaPeekAtLastError per frame
        self.total_flush_probe_us: float = 0.0  # cudaStreamQuery WDDM flush probe
        self.total_shm_publish_us: float = 0.0  # write_idx + _release_fence + SHM pack
        self.total_unaccounted_us: float = 0.0  # frame_time − Σ all sub-timers

        # Verbosity control - read from local Debug parameter
        try:
            self.verbose_performance = bool(ownerComp.par.Debug.eval())
        except AttributeError:
            self.verbose_performance = False
        if self._export_profile:
            self.verbose_performance = True  # profile mode requires verbose timing path

        # Apply showCustomOnly from Hidebuiltin parameter on load
        with contextlib.suppress(AttributeError):
            self.ownerComp.showCustomOnly = bool(ownerComp.par.Hidebuiltin.eval())

        # Receiver-specific state (only used when mode='Receiver')
        self._rx_dev_ptrs = [None] * self.num_slots  # Opened IPC mem pointers
        self._rx_ipc_handles = [None] * self.num_slots  # IPC mem handles read from SHM
        self._rx_ipc_events = [None] * self.num_slots  # Opened IPC events
        self._rx_ipc_version = 0
        self._rx_num_slots = 0
        self._rx_shutdown_offset = 0  # Cached: SHM_HEADER_SIZE + _rx_num_slots * SLOT_SIZE
        self._rx_width = 0
        self._rx_height = 0
        self._rx_num_comps = 0
        self._rx_dtype_code = 0
        self._rx_buffer_size = 0
        self._rx_last_write_idx = 0

        # Receiver retry state
        self._rx_connect_attempts = 0
        self._rx_max_connect_attempts = 20
        self._rx_backoff_intervals = [1, 2, 4, 8, 16, 32, 64, 120]  # frames (exponential)
        self._rx_retry_interval_frames = 1  # Will be updated dynamically after failed attempts
        self._rx_frames_since_last_retry = 0  # Start at 0; first increment yields 1 >= 1 → immediate connect
        self._rx_needs_resolution_update = False  # Flag to defer Script TOP resolution setup
        self._rx_f16_cpu_buf = None  # float16 CPU buffer for D2H conversion (float16 IPC only)
        self._rx_f32_cpu_buf = None  # float32 CPU buffer after conversion (float16 IPC only)
        self._rx_f16_pinned_ptr = None  # cudaMallocHost pointer backing _rx_f16_cpu_buf (or None if pageable)
        self._rx_cupy_f32_buf = None  # float32 CuPy GPU buffer for GPU-side f16→f32 (CuPy path, float16 IPC only)
        self._rx_cupy_f16_views: list = []  # per-slot float16 CuPy views (pre-allocated, avoids per-frame alloc)

        # Format conversion state — True when dtype_converter is set to rgba32float
        self._fmt_conv_active = False
        # Cached dtype_converter Transform TOP reference (stable; refreshed on initialize())
        self._fmt_transform: object = None
        # Cached ExportBuffer TOP reference — eliminates per-frame ownerComp.op() lookup (same pattern as _fmt_transform)
        self._export_buffer: object = None
        # Format conversion result cache — avoid .lower() + set scan when pixel format unchanged
        self._last_pixel_fmt: str = ""
        self._last_fmt_needs_conv: bool = False

        # Lazy attributes — pre-initialized here to avoid per-frame hasattr()/getattr() fallbacks
        self.ipc_stream = None  # Created on first export_frame() or initialize()
        self._last_cuda_mem_err = ""  # Suppresses duplicate cudaMemory() error logs

        # Dtype change detection — pre-initialized to eliminate per-frame hasattr() in _has_dtype_changed()
        self._detected_numpy_dtype: object = None  # Set each frame from cuda_mem.shape.dataType
        self._last_numpy_dtype: object = None  # Set in _write_metadata_to_shm()

        # Cached parameter reference — eliminates 3-deep ownerComp.par.Active chain per frame
        try:
            self._active_par = ownerComp.par.Active
        except AttributeError:
            self._active_par = None  # Component has no Active parameter

        self._log(f"Extension initialized on {ownerComp} [Mode: {self._mode}]", force=True)

        # Disable Numslots in Receiver mode — sender controls slot count via SharedMemory
        if self._mode == "Receiver":
            with contextlib.suppress(AttributeError):
                self.ownerComp.par.Numslots.enable = False

    @property
    def mode(self) -> str:
        """Current operating mode: 'Sender' or 'Receiver'."""
        return self._mode

    def _log(self, msg: str, force: bool = False) -> None:
        """Log message with optional verbosity control.

        Args:
            msg: Message to log
            force: If True, always log regardless of verbosity setting
        """
        prefix = f"[CUDAIPCExtension:{self._mode}]"
        if force or self.verbose_performance:
            print(f"{prefix} {msg}")

    def _needs_format_conversion(self, top_op: TOP) -> bool:
        """Return True if the TOP's pixel format is unsupported by cudaMemory() in TD 2025.

        TD 2025 (CUDA 12.8) rejects float16 formats from cudaMemory().
        uint8, uint16 (fixed), and float32 are supported.
        """
        pixel_fmt = str(getattr(top_op, "pixelFormat", ""))
        if pixel_fmt == self._last_pixel_fmt:
            return self._last_fmt_needs_conv
        self._last_pixel_fmt = pixel_fmt
        pixel_lower = pixel_fmt.lower()
        self._last_fmt_needs_conv = any(u in pixel_lower for u in _CUDA_UNSUPPORTED_PIXEL_FORMATS)
        return self._last_fmt_needs_conv

    def switch_mode(self, new_mode: str) -> None:
        """Switch between Sender and Receiver modes.

        Args:
            new_mode: 'Sender' or 'Receiver'
        """
        if new_mode == self._mode:
            return

        self._log(f"Switching mode: {self._mode} -> {new_mode}", force=True)

        # Cleanup current mode
        self.cleanup()

        # Reset state
        self._mode = new_mode
        self._initialized = False
        self.frame_count = 0

        # When switching to Sender: re-read num_slots from UI (receiver may have updated it)
        # and resize sender arrays to match. cleanup() → _cleanup_sender() already reset
        # them to [], so initialize() would resize anyway, but doing it here ensures
        # export_frame()'s is_ready() check sees the correct array size immediately.
        if new_mode == "Sender":
            with contextlib.suppress(AttributeError, ValueError):
                self.num_slots = int(self.ownerComp.par.Numslots.eval())
            self.dev_ptrs = [None] * self.num_slots
            self.ipc_handles = [None] * self.num_slots
            self.ipc_events = [None] * self.num_slots
            self.ipc_event_handles = [None] * self.num_slots

        # Enable/disable Numslots based on new mode:
        # - Sender: editable (but only when not active — initialize() will disable it on activation)
        # - Receiver: always disabled (sender controls slot count via SharedMemory)
        with contextlib.suppress(AttributeError):
            self.ownerComp.par.Numslots.enable = new_mode == "Sender"

        self._log(f"Mode switched to {new_mode}. Will initialize on next frame.", force=True)

    def initialize(self, width: int, height: int, channels: int = 4, buffer_size: int | None = None) -> bool:
        """Initialize CUDA IPC resources.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            channels: Number of channels (default: 4 for RGBA)
            buffer_size: Actual buffer size in bytes (optional, auto-calculated if None)

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            self._log("Already initialized")
            return True

        # Lock Numslots while active — changing slot count at runtime causes array size mismatch
        with contextlib.suppress(AttributeError):
            self.ownerComp.par.Numslots.enable = False

        try:
            # Load CUDA runtime bound to the configured device
            self.cuda = get_cuda_runtime(device=self.device)
            self._log(f"Loaded CUDA runtime on device {self.cuda.get_device()}", force=True)

            # Create high-priority dedicated non-blocking stream for IPC operations.
            # Reuse existing stream on re-init to avoid leaks.
            if self.ipc_stream is None:
                self.ipc_stream = self.cuda.create_stream_with_priority(flags=0x01)
                self._log(
                    f"Created IPC stream (high-priority): 0x{int(self.ipc_stream.value):016x}",
                    force=True,
                )
            else:
                self._log(
                    f"Reusing IPC stream: 0x{int(self.ipc_stream.value):016x}",
                    force=True,
                )

            # Store dimensions
            self.width = width
            self.height = height
            self.channels = channels
            # Use provided buffer_size (from cuda_mem.size) or calculate
            raw_size = buffer_size if buffer_size is not None else width * height * channels * 4
            # Round up to 2MiB alignment (NVIDIA requirement: prevents unintended information disclosure)
            alignment = 2 * 1024 * 1024  # 2 MiB
            self.buffer_size = ((raw_size + alignment - 1) // alignment) * alignment
            self.data_size = raw_size  # Store actual data size for memcpy and comparisons

            # Defensive array resize: num_slots may have changed between cleanup and init
            # (e.g. handle_numslots_change() sets num_slots after cleanup resets arrays)
            if len(self.dev_ptrs) != self.num_slots:
                self.dev_ptrs = [None] * self.num_slots
                self.ipc_handles = [None] * self.num_slots
                self.ipc_events = [None] * self.num_slots
                self.ipc_event_handles = [None] * self.num_slots

            # Allocate ring buffer slots
            for slot in range(self.num_slots):
                # Allocate persistent GPU buffer for this slot
                self.dev_ptrs[slot] = self.cuda.malloc(self.buffer_size)
                self._log(
                    f"Allocated GPU buffer slot {slot}: "
                    f"{self.buffer_size / 1024 / 1024:.1f} MB at 0x{self.dev_ptrs[slot].value:016x}",
                    force=True,
                )

                # Create IPC handle for this buffer (ONCE - reuse for all frames)
                self.ipc_handles[slot] = self.cuda.ipc_get_mem_handle(self.dev_ptrs[slot])
                self._log(f"Created IPC handle for slot {slot} (64 bytes)")

                # Create IPC event for GPU-side synchronization (per-slot)
                self.ipc_events[slot] = self.cuda.create_ipc_event()
                self.ipc_event_handles[slot] = self.cuda.ipc_get_event_handle(self.ipc_events[slot])
                self._log(f"Created IPC event for slot {slot} (64 bytes)")

            self._log(f"Created {self.num_slots} IPC buffer slots with events", force=True)

            # Create SharedMemory for IPC handle transfer
            # Size: header + slots + shutdown flag + metadata + timestamp (for extended protocol)
            shm_size = (
                SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
            )

            try:
                # Try to open existing SharedMemory first
                self.shm_handle = SharedMemory(name=self.shm_name)
                self._log(f"Opened existing SharedMemory: {self.shm_name}", force=True)
            except FileNotFoundError:
                # Create new SharedMemory if doesn't exist
                self.shm_handle = SharedMemory(name=self.shm_name, create=True, size=shm_size)
                self._log(
                    f"Created new SharedMemory: {self.shm_name} ({shm_size} bytes)",
                    force=True,
                )

            # Write IPC handle to SharedMemory (ONCE - Python process reads at startup)
            self._write_handle_to_shm()

            # Write texture metadata to extended protocol region
            self._write_metadata_to_shm()

            # Cache SHM offsets: avoid recomputing these on every export_frame() call
            self._shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
            self._ts_offset = self._shutdown_offset + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

            # Cache dtype_converter TOP reference — eliminates per-frame ownerComp.op() lookup
            self._fmt_transform = self.ownerComp.op(_FMT_TRANSFORM_NAME)
            # Cache ExportBuffer TOP reference — eliminates per-frame ownerComp.op() lookup (same pattern)
            self._export_buffer = self.ownerComp.op(_EXPORT_BUFFER_NAME)

            # Create GPU timing events (only when Debug is ON for benchmarking)
            if self.verbose_performance:
                self._timing_start = self.cuda.create_timing_event()
                self._timing_end = self.cuda.create_timing_event()
                self._log("Created GPU timing events for benchmarking", force=False)
            else:
                self._timing_start = None
                self._timing_end = None

            self._initialized = True
            self._log("Initialization complete - ready for zero-copy GPU transfer", force=True)

            if NVML_AVAILABLE and os.environ.get("CUDALINK_NVML", "0") == "1":
                obs = NVMLObserver(device=self.device, enabled=True)
                if obs.start():
                    self._nvml_observer = obs
                    self._log(f"NVMLObserver attached on device {self.device}", force=True)

            return True

        except (OSError, RuntimeError, ValueError) as e:
            self._log(f"Initialization failed: {e}", force=True)
            traceback.print_exc()
            return False

    def _write_handle_to_shm(self) -> None:
        """Write magic + version + num_slots + write_idx + all IPC handles to SharedMemory.

        Layout (20 + NUM_SLOTS*192 + 1 bytes):
        [0-3]     magic (4B) - protocol validation "CIPC"
        [4-11]    version (8B)
        [12-15]   num_slots (4B)
        [16-19]   write_idx (4B)

        For each slot (128 bytes per slot):
        [20+slot*128 : 84+slot*128]   mem_handle (64B)
        [84+slot*128 : 148+slot*128]  event_handle (64B)

        [20+NUM_SLOTS*128]  shutdown flag (1B)
        """
        if self.shm_handle is None or not all(self.ipc_handles):
            return

        # Write protocol magic number (new in this version)
        self.shm_handle.buf[MAGIC_OFFSET : MAGIC_OFFSET + MAGIC_SIZE] = struct.pack("<I", PROTOCOL_MAGIC)

        # Read current version (if exists) and increment
        try:
            current_version = struct.unpack(
                "<Q",
                bytes(self.shm_handle.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE]),
            )[0]
        except (struct.error, ValueError, IndexError):
            current_version = 0
        new_version = current_version + 1

        # Write header
        self.shm_handle.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE] = struct.pack("<Q", new_version)
        self.shm_handle.buf[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + NUM_SLOTS_SIZE] = struct.pack("<I", self.num_slots)
        self.shm_handle.buf[WRITE_IDX_OFFSET : WRITE_IDX_OFFSET + WRITE_IDX_SIZE] = struct.pack(
            "<I", 0
        )  # write_idx=0 initially

        # Write handles for each slot
        for slot in range(self.num_slots):
            base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

            # Write memory handle (64 bytes)
            mem_handle_bytes = bytes(self.ipc_handles[slot].internal)
            self.shm_handle.buf[base_offset : base_offset + 64] = mem_handle_bytes

            # Write event handle (64 bytes) if available
            if self.ipc_event_handles[slot]:
                event_handle_bytes = bytes(self.ipc_event_handles[slot].reserved)
                self.shm_handle.buf[base_offset + 64 : base_offset + 128] = event_handle_bytes
                self._log(f"Wrote slot {slot} handles: mem={len(mem_handle_bytes)}B, event={len(event_handle_bytes)}B")
            else:
                self._log(f"Wrote slot {slot} mem handle: {len(mem_handle_bytes)}B")

        # Clear shutdown flag — matches CUDAIPCExporter._write_handles_to_shm() on the Python side.
        # Without this, a stale shutdown_flag=1 from a previous session (or a race where another
        # sender initialised after this one) would block the receiver indefinitely.
        shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
        self.shm_handle.buf[shutdown_offset] = 0

        self._log(
            f"Wrote all IPC handles v{new_version} to SharedMemory ({SHM_HEADER_SIZE + self.num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE} bytes total)",
            force=True,
        )

    def _write_metadata_to_shm(self) -> None:
        """Write texture metadata to the extended protocol region after shutdown flag.

        Extended protocol layout (appended after existing 593 bytes for 3 slots):
        [shutdown_offset+1 : +4]  width (uint32 LE)
        [shutdown_offset+5 : +4]  height (uint32 LE)
        [shutdown_offset+9 : +4]  num_comps (uint32 LE)
        [shutdown_offset+13 : +4] dtype_code (uint32 LE)  # 0=float32, 1=float16, 2=uint8
        [shutdown_offset+17 : +4] buffer_size (uint32 LE)
        """
        if self.shm_handle is None or self.data_size == 0:
            return

        # Calculate metadata offset (immediately after shutdown flag)
        shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
        metadata_offset = shutdown_offset + SHUTDOWN_FLAG_SIZE

        # Determine dtype_code from CUDAMemoryShape.dataType (preferred) or byte-ratio (fallback)
        detected_dtype = self._detected_numpy_dtype
        if detected_dtype is not None:
            try:
                import numpy as _np

                _dtype_to_code = {
                    _np.dtype("float32"): DTYPE_FLOAT32,
                    _np.dtype("float16"): DTYPE_FLOAT16,
                    _np.dtype("uint8"): DTYPE_UINT8,
                    _np.dtype("uint16"): DTYPE_UINT16,
                }
                dtype_code = _dtype_to_code.get(detected_dtype, DTYPE_FLOAT32)
            except Exception:  # noqa: BLE001
                detected_dtype = None  # Fall through to byte-ratio below

        if detected_dtype is None:
            # Fallback: byte-ratio inference (cannot distinguish float16 from uint16)
            if self.width > 0 and self.height > 0 and self.channels > 0:
                bytes_per_pixel = self.data_size / (self.width * self.height)
                bytes_per_comp = bytes_per_pixel / self.channels
            else:
                bytes_per_comp = 4  # Default to float32
            if bytes_per_comp <= 1:
                dtype_code = DTYPE_UINT8
            elif bytes_per_comp <= 2:
                # dtype_converter promotes float16 → float32 upstream, so any
                # remaining 2-byte format at this point is uint16 (fixed-point).
                dtype_code = DTYPE_UINT16
            else:
                dtype_code = DTYPE_FLOAT32

        # Write metadata fields
        self.shm_handle.buf[metadata_offset : metadata_offset + 4] = struct.pack("<I", self.width)
        self.shm_handle.buf[metadata_offset + 4 : metadata_offset + 8] = struct.pack("<I", self.height)
        self.shm_handle.buf[metadata_offset + 8 : metadata_offset + 12] = struct.pack("<I", self.channels)
        self.shm_handle.buf[metadata_offset + 12 : metadata_offset + 16] = struct.pack("<I", dtype_code)
        self.shm_handle.buf[metadata_offset + 16 : metadata_offset + 20] = struct.pack("<I", self.data_size)

        # Track last written dtype for change detection
        self._last_numpy_dtype = self._detected_numpy_dtype

        self._log(
            f"Wrote metadata: {self.width}x{self.height}x{self.channels}, dtype={dtype_code}, size={self.data_size}B"
        )

    def _has_dtype_changed(self) -> bool:
        """Check if detected numpy dtype differs from last written metadata.

        Both attributes are pre-initialized to None in __init__ and set as numpy.dtype
        objects (from cuda_mem.shape.dataType / _write_metadata_to_shm), so direct
        comparison is safe — no per-frame np.dtype() construction needed.
        """
        if self._detected_numpy_dtype is None or self._last_numpy_dtype is None:
            return False  # Not yet detected or not yet written
        return self._detected_numpy_dtype != self._last_numpy_dtype

    def _bump_version(self) -> None:
        """Increment SharedMemory version counter to signal consumers to re-read metadata."""
        if self.shm_handle is None:
            return
        try:
            current_version = struct.unpack_from("<Q", self.shm_handle.buf, VERSION_OFFSET)[0]
        except (struct.error, ValueError, IndexError):
            current_version = 0
        new_version = current_version + 1
        self.shm_handle.buf[VERSION_OFFSET : VERSION_OFFSET + VERSION_SIZE] = struct.pack("<Q", new_version)
        self._log(f"Version bumped to {new_version} (metadata-only change)")

    def export_frame(self, top_op: TOP | None = None) -> bool:
        """Export the ExportBuffer TOP texture via CUDA IPC.

        Resolves ExportBuffer internally (downstream of dtype_converter) so the
        correct post-conversion frame is always exported regardless of what the
        caller previously passed.

        Args:
            top_op: Deprecated. Accepted for backwards compatibility but ignored.
                ExportBuffer is always resolved from ownerComp internally.

        Returns:
            True if export successful, False otherwise
        """
        top_op = self._export_buffer
        if top_op is None or not getattr(top_op, "valid", True):
            self._export_buffer = None  # invalidate stale cache
            # Lazy lookup: op may have been added after initialize() (e.g. dynamic network edits)
            top_op = self.ownerComp.op(_EXPORT_BUFFER_NAME)
            if top_op is None:
                self._log(f"'{_EXPORT_BUFFER_NAME}' not found in component", force=True)
                return False
            self._export_buffer = top_op  # cache for subsequent frames

        # Check if Active parameter is enabled (use cached par ref to avoid 3-deep chain per frame)
        if self._active_par is not None and not bool(self._active_par.eval()):
            return False

        # Start frame timer (only if verbose)
        if self.verbose_performance:
            frame_start = time.perf_counter()
            if self._export_profile:
                _t_pre = frame_start
                # initialize per-frame profile locals so unaccounted calc is always defined
                _this_pre = _this_post = _this_sync = _this_sticky = _this_fp = _this_shm = 0.0
                # record_event_time is only set in the ipc_events path; init here for the fallback
                record_event_time = 0.0

        try:
            # Ensure CUDA runtime and stream exist BEFORE first cudaMemory() call.
            # Always use a non-blocking stream (never None/default stream) for TD 2025 compat.
            if self.cuda is None:
                self.cuda = get_cuda_runtime(device=self.device)
            if self.ipc_stream is None:
                self.ipc_stream = self.cuda.create_stream_with_priority(flags=0x01)
                self._log(
                    f"Created IPC stream (pre-init): 0x{int(self.ipc_stream.value):016x}",
                    force=True,
                )

            # TD 2025 rejects float16 pixel formats from cudaMemory().
            # dtype_converter Transform TOP sits before ExportBuffer — toggle its format param.
            # Check source TOP (upstream of converter), not ExportBuffer (downstream).
            # Use cached reference (set in initialize()) to avoid per-frame ownerComp.op() lookup.
            fmt_transform = self._fmt_transform
            if fmt_transform is not None and not getattr(fmt_transform, "valid", True):
                self._fmt_transform = None
                fmt_transform = None
            if fmt_transform is None:
                # Lazy lookup: initialize() may not have run yet (pre-init export_frame() path).
                # Without this, float16 sources are permanently stuck — cudaMemory() fails before
                # auto-init at line 719 is reached. Same pattern as _export_buffer lazy fallback.
                fmt_transform = self.ownerComp.op(_FMT_TRANSFORM_NAME)
                if fmt_transform is not None:
                    self._fmt_transform = fmt_transform
            if fmt_transform is not None:
                source_top = fmt_transform.inputs[0] if fmt_transform.inputs else top_op
                if self._needs_format_conversion(source_top):
                    if not self._fmt_conv_active:
                        fmt_transform.par.format = "rgba32float"
                        self._fmt_conv_active = True
                        self._log(
                            f"Pixel format '{getattr(source_top, 'pixelFormat', '?')}' unsupported "
                            f"by cudaMemory() — dtype_converter set to rgba32float, skipping frame",
                            force=True,
                        )
                        return False  # dtype_converter cooks next frame
                else:
                    if self._fmt_conv_active:
                        fmt_transform.par.format = "useinput"
                        self._fmt_conv_active = False
                        self._log(
                            "Source format CUDA-compatible — dtype_converter set to useinput",
                            force=True,
                        )
                        return False  # format reverts next cook

            # Time cudaMemory() call (OpenGL→CUDA interop)
            if self.verbose_performance:
                if self._export_profile:
                    _this_pre = (time.perf_counter() - _t_pre) * 1_000_000
                    self.total_pre_interop_us += _this_pre
                cuda_mem_start = time.perf_counter()

            # Get TOP's CUDA memory — always pass a valid stream (never None)
            try:
                cuda_mem = top_op.cudaMemory(
                    stream=int(self.ipc_stream.value),
                )
            except Exception as cuda_err:
                pixel_fmt = getattr(top_op, "pixelFormat", "unknown")
                err_msg = f"cudaMemory() failed (pixelFormat={pixel_fmt}): {cuda_err}"
                if err_msg != self._last_cuda_mem_err:
                    self._log(err_msg, force=True)
                    self._last_cuda_mem_err = err_msg
                return False

            if self.verbose_performance:
                cuda_mem_time = (time.perf_counter() - cuda_mem_start) * 1_000_000  # microseconds
                self.total_cuda_memory_time += cuda_mem_time
                if self._export_profile:
                    _t_post = time.perf_counter()

            # Reset error suppression on success
            if self._last_cuda_mem_err:
                self._log("cudaMemory() recovered.", force=True)
                self._last_cuda_mem_err = ""

            if cuda_mem is None:
                self._log(f"Failed to get CUDA memory from {top_op}", force=True)
                return False

            # CRITICAL: Keep reference to prevent garbage collection
            self.cuda_mem_ref = cuda_mem

            # Get dimensions from cuda_mem.shape (cache reference to avoid repeated lookups)
            shape = cuda_mem.shape
            actual_width = shape.width
            actual_height = shape.height
            actual_channels = shape.numComps
            actual_size = cuda_mem.size
            # CUDAMemoryShape.dataType returns numpy.dtype — more reliable than byte-ratio inference
            try:
                self._detected_numpy_dtype = shape.dataType
            except AttributeError:
                self._detected_numpy_dtype = None

            # Check if we need to (re)initialize
            if not self._initialized or actual_size != self.data_size:
                if self._initialized:
                    self._log(
                        f"Resolution changed: {self.width}x{self.height}x{self.channels} -> {actual_width}x{actual_height}x{actual_channels}",
                        force=True,
                    )
                    # Queue old resources for deferred free (cudaFree blocks on IPC memory)
                    self._pending_free_ptrs.extend([p for p in self.dev_ptrs if p])
                    self._pending_free_events.extend([e for e in self.ipc_events if e])
                    self.dev_ptrs = [None] * self.num_slots
                    self.ipc_events = [None] * self.num_slots
                    self.ipc_handles = [None] * self.num_slots
                    self.ipc_event_handles = [None] * self.num_slots
                    self._initialized = False
                    # Schedule deferred free after 30 frames (receiver needs time to close handles)
                    self._deferred_free_at_frame = self.frame_count + 30

                if not self.initialize(actual_width, actual_height, actual_channels, actual_size):
                    return False
                # Metadata already written by initialize()

            elif (
                actual_width != self.width
                or actual_height != self.height
                or actual_channels != self.channels
                or self._has_dtype_changed()
            ):
                # Metadata-only update: buffer size unchanged so GPU handles stay valid.
                # Rewrite the 20-byte metadata region and bump version to signal consumers.
                self.width = actual_width
                self.height = actual_height
                self.channels = actual_channels
                self._write_metadata_to_shm()
                self._bump_version()
                self._log(
                    "Metadata changed (dtype/dimensions) without size change — updated in-place",
                    force=True,
                )

            # Calculate current slot for ring buffer rotation
            slot = self.write_idx % self.num_slots

            # Time cudaMemcpyAsync D2D (non-blocking) - only if verbose
            if self.verbose_performance:
                if self._export_profile:
                    _this_post = (time.perf_counter() - _t_post) * 1_000_000
                    self.total_post_interop_us += _this_post
                memcpy_start = time.perf_counter()
                # Record GPU timing start event (actual GPU time measurement)
                if self._timing_start:
                    self.cuda.record_event(self._timing_start, stream=self.ipc_stream)

            # Copy TOP texture to this slot's persistent buffer (async on IPC stream)
            self.cuda.memcpy_async(
                dst=self.dev_ptrs[slot],
                src=c_void_p(cuda_mem.ptr),
                count=self.data_size,
                kind=3,  # cudaMemcpyDeviceToDevice
                stream=self.ipc_stream,
            )

            if self.verbose_performance:
                # Record GPU timing end event (actual GPU time measurement)
                if self._timing_end:
                    self.cuda.record_event(self._timing_end, stream=self.ipc_stream)
                memcpy_time = (
                    time.perf_counter() - memcpy_start
                ) * 1_000_000  # microseconds (enqueue time only, copy is async)
                self.total_memcpy_time += memcpy_time

            # GPU-side synchronization with CUDA IPC Events
            if self.ipc_events[slot]:
                if self.verbose_performance:
                    record_start = time.perf_counter()

                # Record event for this slot after async memcpy (stream-ordered)
                self.cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)

                if self.verbose_performance:
                    record_event_time = (time.perf_counter() - record_start) * 1_000_000
                    self.total_record_event_time += record_event_time

                # CUDALINK_EXPORT_SYNC=1: CPU-blocks on ipc_stream after record_event.
                # Default is now "0" (receiver cudaStreamWaitEvent guarantees correctness).
                # Enable for regression testing or if downstream consumers rely on CPU-timing.
                if self._export_sync:
                    if self.verbose_performance and self._export_profile:
                        _t_sync = time.perf_counter()
                    self.cuda.stream_synchronize(self.ipc_stream)
                    if self.verbose_performance and self._export_profile:
                        _this_sync = (time.perf_counter() - _t_sync) * 1_000_000
                        self.total_sync_us += _this_sync

                if self.verbose_performance and self._export_profile:
                    _t_sticky = time.perf_counter()
                self.cuda.check_sticky_error("export_frame")
                if self.verbose_performance and self._export_profile:
                    _this_sticky = (time.perf_counter() - _t_sticky) * 1_000_000
                    self.total_sticky_check_us += _this_sticky

                # WDDM deferred-submission probe: forces pending GPU work to submit without
                # blocking. Per CUDA Handbook p3/pg56, WDDM buffers commands until a flush;
                # cudaStreamQuery triggers that flush. Only active when EXPORT_FLUSH_PROBE=1
                # and EXPORT_SYNC=0 (if sync is on, the stream is already flushed above).
                if self._export_flush_probe and not self._export_sync:
                    if self.verbose_performance and self._export_profile:
                        _t_fp = time.perf_counter()
                    self.cuda.stream_query(self.ipc_stream)
                    if self.verbose_performance and self._export_profile:
                        _this_fp = (time.perf_counter() - _t_fp) * 1_000_000
                        self.total_flush_probe_us += _this_fp

                # Always write producer timestamp (enables consumer latency measurement)
                _ST_F64.pack_into(self.shm_handle.buf, self._ts_offset, time.perf_counter())
            else:
                # FALLBACK: Conditional CPU synchronization
                if self.frame_count % self.sync_interval == 0:
                    self.cuda.synchronize()

            # Publish write_idx LAST: clear shutdown_flag first so the consumer
            # (which reads shutdown_flag before write_idx) always sees flag=0
            # when it detects a new frame (atomicity improvement).
            if self.verbose_performance and self._export_profile:
                _t_shm = time.perf_counter()
            self.write_idx += 1
            self.shm_handle.buf[self._shutdown_offset] = 0
            _release_fence()  # C3: release barrier — shutdown_flag visible before write_idx
            _ST_U32.pack_into(self.shm_handle.buf, WRITE_IDX_OFFSET, self.write_idx)  # publish last
            if self.verbose_performance and self._export_profile:
                _this_shm = (time.perf_counter() - _t_shm) * 1_000_000
                self.total_shm_publish_us += _this_shm

            # Frame tracking
            self.frame_count += 1

            # Calculate total frame time (only if verbose)
            if self.verbose_performance:
                frame_time = (time.perf_counter() - frame_start) * 1_000_000
                self.total_export_time += frame_time
                if self._export_profile:
                    _this_accounted = (
                        _this_pre
                        + cuda_mem_time
                        + _this_post
                        + memcpy_time
                        + record_event_time
                        + _this_sync
                        + _this_sticky
                        + _this_fp
                        + _this_shm
                    )
                    self.total_unaccounted_us += frame_time - _this_accounted

            # Detailed first-frame diagnostic (one-time, not affected by 100-frame interval)
            if self.verbose_performance and self.frame_count == 1:
                self._log(
                    f"FIRST FRAME: cudaMemory={cuda_mem_time:.1f}us, "
                    f"memcpy={memcpy_time:.1f}us, total={frame_time:.1f}us, "
                    f"res={actual_width}x{actual_height}, size={actual_size / (1024 * 1024):.1f}MB",
                    force=True,
                )

            # Log performance metrics every 97 frames (prime — avoids aliasing with slot counts 2,4,5)
            if self.verbose_performance and self.frame_count % 97 == 0:
                avg_memcpy = self.total_memcpy_time / self.frame_count
                avg_record = self.total_record_event_time / self.frame_count if all(self.ipc_events) else 0
                avg_total = self.total_export_time / self.frame_count
                avg_cuda_mem = self.total_cuda_memory_time / self.frame_count
                sync_mode = (
                    f"GPU-Events[{self.num_slots}]" if all(self.ipc_events) else f"CPU-Sync(1/{self.sync_interval})"
                )

                log_msg = (
                    f"Frame {self.frame_count}: slot {slot}, "
                    f"avg cudaMemory={avg_cuda_mem:.1f}us, "
                    f"avg memcpy={avg_memcpy:.1f}us, record={avg_record:.1f}us, "
                    f"total={avg_total:.1f}us, mode={sync_mode}"
                )

                # Add GPU elapsed time if timing events available
                if self._timing_start and self._timing_end:
                    try:
                        # Wait for timing events to complete before reading (prevents error 600)
                        self.cuda.wait_event(self._timing_end)
                        gpu_memcpy_ms = self.cuda.event_elapsed_time(self._timing_start, self._timing_end)
                        log_msg += f", GPU memcpy={gpu_memcpy_ms * 1000:.1f}us (actual GPU time)"
                    except RuntimeError as e:
                        # Rare: event wait/query failed
                        log_msg += f", GPU timing: {e}"

                if self._nvml_observer is not None:
                    snap = self._nvml_observer.snapshot()
                    if snap.get("nvml_available"):
                        log_msg += (
                            f" | [NVML] gpu={snap.get('gpu_util_pct', '?')}%"
                            f" mem={snap.get('mem_bw_util_pct', '?')}%"
                            f" sm={snap.get('sm_clock_mhz', '?')}MHz"
                            f" pcie_tx={snap.get('pcie_tx_kbps', '?')}kbps"
                            f" pcie_rx={snap.get('pcie_rx_kbps', '?')}kbps"
                            f" temp={snap.get('temp_c', '?')}C"
                            f" power={snap.get('power_w', '?')}W"
                        )
                        reasons = snap.get("throttle_reasons") or []
                        if reasons:
                            log_msg += f" throttle={','.join(reasons)}"

                if self._export_profile:
                    avg_pre = self.total_pre_interop_us / self.frame_count
                    avg_post = self.total_post_interop_us / self.frame_count
                    avg_sync = self.total_sync_us / self.frame_count
                    avg_sticky = self.total_sticky_check_us / self.frame_count
                    avg_fp = self.total_flush_probe_us / self.frame_count
                    avg_shm = self.total_shm_publish_us / self.frame_count
                    avg_unacc = self.total_unaccounted_us / self.frame_count
                    log_msg += (
                        f" | [PROFILE] pre={avg_pre:.1f}us"
                        f" interop={avg_cuda_mem:.1f}us"
                        f" post={avg_post:.1f}us"
                        f" memcpy={avg_memcpy:.1f}us"
                        f" record={avg_record:.1f}us"
                        f" sync={avg_sync:.1f}us"
                        f" sticky={avg_sticky:.1f}us"
                        f" flush_probe={avg_fp:.1f}us"
                        f" shm={avg_shm:.1f}us"
                        f" unacc={avg_unacc:.1f}us"
                    )

                self._log(log_msg, force=False)

            return True

        except (OSError, RuntimeError, AttributeError) as e:
            self._log(f"Export failed: {e}", force=True)

            traceback.print_exc()
            return False

    def _check_deferred_cleanup(self) -> None:
        """Execute deferred GPU cleanup if scheduled and enough frames have passed.

        Lightweight check meant to be called from onFrameStart for minimal overhead.
        """
        if self._pending_free_ptrs and self.frame_count >= self._deferred_free_at_frame:
            self._deferred_free()

    def _deferred_free(self) -> None:
        """Free GPU resources queued from export_frame() when deferred frame threshold is reached.

        Called via _check_deferred_cleanup() after receiver has had time to close IPC handles.
        """
        if self.cuda is None:
            return

        freed_count = 0
        for ptr in self._pending_free_ptrs:
            try:
                self.cuda.free(ptr)
                freed_count += 1
            except (RuntimeError, OSError) as e:
                self._log(f"Deferred free failed: {e}")
        self._pending_free_ptrs.clear()

        for event in self._pending_free_events:
            try:
                self.cuda.destroy_event(event)
            except (RuntimeError, OSError) as e:
                self._log(f"Deferred event destroy failed: {e}")
        self._pending_free_events.clear()

        if freed_count > 0:
            self._log(
                f"Deferred cleanup complete: freed {freed_count} GPU buffers",
                force=True,
            )

    def cleanup(self) -> None:
        """Cleanup all resources for the current mode."""
        if self._mode == "Sender":
            self._cleanup_sender()
            # Re-enable Numslots when sender deactivates (allow editing again)
            with contextlib.suppress(AttributeError):
                self.ownerComp.par.Numslots.enable = True
        else:
            self.cleanup_receiver()

    def __delTD__(self) -> None:
        """TouchDesigner extension cleanup method (called on re-initialization).

        TD calls __delTD__ (not Python's __del__) when the extension is destroyed.
        This provides a safety net for cleanup if onExit() isn't called.
        """
        self.cleanup()

    def _is_cuda_context_valid(self) -> bool:
        """Check if CUDA context is still valid (TD may destroy it before __delTD__)."""
        if self.cuda is None:
            return False
        try:
            self.cuda.cudart.cudaGetLastError()
            return True
        except (OSError, RuntimeError):
            return False

    def _cleanup_sender(self) -> None:
        """Cleanup Sender CUDA IPC resources (all ring buffer slots).

        CRITICAL ORDER: Signal shutdown FIRST, then free GPU resources.
        cudaFree() blocks until all processes close IPC handles.
        """
        # Skip if already cleaned up (prevents double-cleanup from Active toggle + __delTD__)
        if not self._initialized and self.shm_handle is None:
            return

        if self._nvml_observer is not None:
            self._nvml_observer.stop()
            self._nvml_observer = None

        cuda_valid = self._is_cuda_context_valid()
        if not cuda_valid:
            self._log("CUDA context already destroyed — skipping GPU cleanup", force=True)

        # Signal shutdown to consumer (before closing SharedMemory)
        if self.shm_handle and self.shm_handle.buf is not None:
            try:
                shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
                self.shm_handle.buf[shutdown_offset] = 1
                self._log("Shutdown signal sent to consumer", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Warning: Could not write shutdown signal: {e}", force=True)

        # Zero out IPC handle bytes so any reader sees invalid handles.
        # On Windows, unlink() is a no-op (SharedMemory uses CreateFileMapping kernel
        # objects), so the SharedMemory may persist with stale non-zero handles that
        # pass the all-zero validation check. Zeroing them prevents error 201 when a
        # new Receiver reads before the SHM is destroyed or overwritten by a new producer.
        if self.shm_handle and self.shm_handle.buf is not None:
            try:
                for slot in range(self.num_slots):
                    base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)
                    self.shm_handle.buf[base_offset : base_offset + SLOT_SIZE] = b"\x00" * SLOT_SIZE
                self._log("Zeroed IPC handle bytes in SharedMemory", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Warning: Could not zero IPC handles: {e}", force=True)

        # Destroy IPC events (sender-side resources, safe to destroy)
        if cuda_valid and hasattr(self, "ipc_events") and self.cuda:
            for slot, event in enumerate(self.ipc_events):
                if event:
                    try:
                        self.cuda.destroy_event(event)
                        self._log(f"Destroyed IPC event slot {slot}", force=True)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error destroying event slot {slot}: {e}", force=True)

        # Destroy GPU timing events (benchmarking resources)
        if cuda_valid and self.cuda:
            if hasattr(self, "_timing_start") and self._timing_start:
                try:
                    self.cuda.destroy_event(self._timing_start)
                    self._log("Destroyed GPU timing start event", force=False)
                except (RuntimeError, OSError) as e:
                    self._log(f"Error destroying timing start event: {e}", force=True)
                finally:
                    self._timing_start = None
            if hasattr(self, "_timing_end") and self._timing_end:
                try:
                    self.cuda.destroy_event(self._timing_end)
                    self._log("Destroyed GPU timing end event", force=False)
                except (RuntimeError, OSError) as e:
                    self._log(f"Error destroying timing end event: {e}", force=True)
                finally:
                    self._timing_end = None

        # Destroy dedicated IPC stream (set to None to prevent double-free)
        if cuda_valid and hasattr(self, "ipc_stream") and self.ipc_stream and self.cuda:
            try:
                self.cuda.destroy_stream(self.ipc_stream)
                self._log("Destroyed IPC stream", force=True)
                self.ipc_stream = None
            except (RuntimeError, OSError) as e:
                self._log(f"Error destroying IPC stream: {e}", force=True)

        # Close SharedMemory (but don't unlink yet)
        if self.shm_handle:
            try:
                self.shm_handle.close()
                self._log("Closed SharedMemory", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Error closing SharedMemory: {e}", force=True)

        # Grace period for receiver to close IPC handles
        if cuda_valid:
            time.sleep(0.1)  # 100ms for receiver to detect shutdown and close handles

        # Free GPU buffers (now safe, receiver has closed IPC handles)
        if cuda_valid and hasattr(self, "dev_ptrs") and self.cuda:
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr:
                    try:
                        self.cuda.free(dev_ptr)
                        self._log(f"Freed GPU buffer slot {slot}", force=True)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error freeing GPU buffer slot {slot}: {e}", force=True)

        # Free any pending deferred resources
        if cuda_valid and hasattr(self, "_pending_free_ptrs"):
            self._deferred_free()

        # Reset dtype_converter Transform TOP to pass-through on cleanup
        fmt_transform = self.ownerComp.op(_FMT_TRANSFORM_NAME)
        if fmt_transform is not None:
            fmt_transform.par.format = "useinput"
        self._fmt_conv_active = False

        # Unlink SharedMemory (sender is owner and should clean up)
        if hasattr(self, "shm_name"):
            try:
                try:
                    shm_temp = SharedMemory(name=self.shm_name)
                    shm_temp.close()
                    shm_temp.unlink()
                    self._log("Unlinked SharedMemory", force=True)
                except FileNotFoundError:
                    pass  # Already unlinked
            except (OSError, RuntimeError, AttributeError) as e:
                self._log(f"Warning: Could not unlink SharedMemory: {e}", force=True)

        # Reset state to prevent double-free on re-entry.
        # Use empty lists — initialize() will resize to current self.num_slots.
        self.dev_ptrs = []
        self.ipc_events = []
        self.ipc_handles = []
        self.ipc_event_handles = []
        self.ipc_stream = None
        self.shm_handle = None
        self._fmt_conv_active = False
        self._export_buffer = None
        self._fmt_transform = None

        # Reset per-session counters so averages are accurate after reinit
        # and slot selection starts from 0 (matching SharedMemory write_idx=0 written on init).
        self.write_idx = 0
        self.frame_count = 0
        self.total_memcpy_time = 0.0
        self.total_record_event_time = 0.0
        self.total_export_time = 0.0
        self.total_cuda_memory_time = 0.0

        self._initialized = False
        self._log("Sender cleanup complete", force=True)

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
        # Check Active parameter (use cached par ref to avoid 3-deep chain per frame)
        if self._active_par is not None and not bool(self._active_par.eval()):
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

            # Copy CUDA memory into ImportBuffer texture using cached shape
            address = self._rx_dev_ptrs[read_slot].value

            if self._rx_dtype_code == DTYPE_FLOAT16:
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

                    import_buffer.copyCUDAMemory(
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
                    import_buffer.copyNumpyArray(self._rx_f32_cpu_buf)
            else:
                import_buffer.copyCUDAMemory(
                    address,
                    self._rx_buffer_size,
                    self._rx_cached_shape,
                    stream=int(self._rx_stream.value),
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
            import_buffer.par.outputresolution = 9  # Custom Resolution
            import_buffer.par.resolutionw = self._rx_width
            import_buffer.par.resolutionh = self._rx_height
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
        with contextlib.suppress(AttributeError):
            self.ownerComp.par.Numslots.enable = False

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
            with contextlib.suppress(AttributeError):
                self.ownerComp.par.Numslots = self._rx_num_slots

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
                self._rx_dtype_code = struct.unpack(
                    "<I",
                    bytes(self.shm_handle.buf[metadata_offset + 12 : metadata_offset + 16]),
                )[0]
                self._rx_buffer_size = struct.unpack(
                    "<I",
                    bytes(self.shm_handle.buf[metadata_offset + 16 : metadata_offset + 20]),
                )[0]
                self._log(
                    f"Read metadata: {self._rx_width}x{self._rx_height}x{self._rx_num_comps}, "
                    f"dtype={self._rx_dtype_code}, buf_size={self._rx_buffer_size}",
                    force=True,
                )
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

            dtype_map = {
                DTYPE_FLOAT32: np_module.float32,
                # float16 not supported by copyCUDAMemory — handled via D2H+copyNumpyArray path
                DTYPE_FLOAT16: np_module.float32,  # shape dtype = float32 (unused for float16 path)
                DTYPE_UINT8: np_module.uint8,
                DTYPE_UINT16: np_module.uint16,
            }
            np_dtype = dtype_map.get(self._rx_dtype_code, np_module.float32)

            # float16: allocate CPU buffers for D2H conversion (copyCUDAMemory doesn't support float16)
            if self._rx_dtype_code == DTYPE_FLOAT16:
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
            self._log(
                f"Receiver initialized: {self._rx_num_slots} slots, "
                f"{self._rx_width}x{self._rx_height}x{self._rx_num_comps}",
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

    def cleanup_receiver(self) -> None:
        """Cleanup Receiver CUDA IPC resources."""
        # Guard against double-cleanup (matches cleanup_sender() pattern)
        if not self._initialized and self.shm_handle is None:
            return

        # Close all IPC memory handles
        if hasattr(self, "_rx_dev_ptrs") and self.cuda:
            for slot, dev_ptr in enumerate(self._rx_dev_ptrs):
                if dev_ptr:
                    try:
                        self.cuda.ipc_close_mem_handle(dev_ptr)
                        self._log(f"Closed IPC handle for slot {slot}")
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error closing IPC handle for slot {slot}: {e}", force=True)

        # Destroy all IPC events (fix per NVIDIA simpleIPC: even opened handles consume resources)
        if hasattr(self, "_rx_ipc_events") and self.cuda:
            for slot, event in enumerate(self._rx_ipc_events):
                if event:
                    try:
                        self.cuda.destroy_event(event)
                        self._log(f"Destroyed IPC event for slot {slot}")
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error destroying event for slot {slot}: {e}", force=True)

        # Destroy receiver stream
        if hasattr(self, "_rx_stream") and self._rx_stream and self.cuda:
            try:
                self.cuda.destroy_stream(self._rx_stream)
                self._log("Destroyed receiver stream", force=True)
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
        self._log("Receiver cleanup complete", force=True)

    def is_ready(self) -> bool:
        """Check if extension is ready (mode-aware).

        Returns:
            True if initialized and ready, False otherwise
        """
        if self._mode == "Sender":
            return self._initialized and all(ptr is not None for ptr in self.dev_ptrs)
        else:
            # len() check required: all([]) returns True, which would incorrectly
            # report ready after cleanup_receiver() resets to empty list
            return (
                self._initialized and len(self._rx_dev_ptrs) > 0 and all(ptr is not None for ptr in self._rx_dev_ptrs)
            )

    def get_stats(self) -> dict[str, object]:
        """Get extension statistics (mode-aware).

        Returns:
            Dictionary with extension stats
        """
        base = {
            "mode": self._mode,
            "initialized": self._initialized,
            "frame_count": self.frame_count,
            "shm_name": self.shm_name,
            "num_slots": self.num_slots,
        }
        if self._mode == "Sender":
            base.update(
                {
                    "buffer_size_mb": self.buffer_size / 1024 / 1024 if self.buffer_size > 0 else 0,
                    "resolution": f"{self.width}x{self.height}x{self.channels}" if self.width > 0 else "N/A",
                    "write_idx": self.write_idx,
                    "dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self.dev_ptrs],
                }
            )
        else:
            base.update(
                {
                    "rx_resolution": f"{self._rx_width}x{self._rx_height}x{self._rx_num_comps}"
                    if self._rx_width > 0
                    else "N/A",
                    "rx_buffer_size_mb": self._rx_buffer_size / 1024 / 1024 if self._rx_buffer_size > 0 else 0,
                    "rx_last_write_idx": self._rx_last_write_idx,
                    "rx_dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self._rx_dev_ptrs],
                }
            )
        return base
