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

import struct
import time
import traceback
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

try:
    import numpy
except ImportError:
    numpy = None  # Will be imported at runtime in TD

# Import types with fallbacks
try:
    from td import COMP, TOP, CUDAMemoryShape
except ImportError:
    from typing import Any as COMP
    from typing import Any as TOP

    CUDAMemoryShape = None  # Will be accessed as global in TD runtime

import contextlib

from CUDAIPCWrapper import (
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
    get_cuda_runtime,
)

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
SLOT_SIZE = 192  # 128B mem_handle + 64B event_handle
SHUTDOWN_FLAG_SIZE = 1
METADATA_SIZE = 20  # 4B width + 4B height + 4B num_comps + 4B dtype_code + 4B buffer_size
TIMESTAMP_SIZE = 8  # 8B float64 producer timestamp (for latency measurement)
# TIMESTAMP_OFFSET calculated at runtime: SHM_HEADER_SIZE + (num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

# Data type codes for extended protocol
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_UINT8 = 2


class CUDAIPCExtension:
    """
    TouchDesigner extension for dual-mode CUDA IPC texture sharing.

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
    - Sender per-frame: < 2μs (D2D copy + event record)
    - Receiver per-frame: < 1μs (event sync + copyCUDAMemory call)
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

        # Frame tracking
        self.frame_count = 0

        # CRITICAL: Keep reference to CUDAMemory to prevent GC
        self.cuda_mem_ref = None

        # Conditional synchronization (CPU fallback)
        self.sync_interval = 10  # Sync every N frames (reduces GPU sync overhead)

        # Performance metrics
        self.total_memcpy_time = 0.0
        self.total_record_event_time = 0.0
        self.total_export_time = 0.0
        self.total_cuda_memory_time = 0.0  # Time spent in cudaMemory() call

        # Verbosity control - read from local Debug parameter
        try:
            self.verbose_performance = bool(ownerComp.par.Debug.eval())
        except AttributeError:
            self.verbose_performance = False

        # Apply showCustomOnly from Hidebuiltin parameter on load
        try:
            self.ownerComp.showCustomOnly = bool(ownerComp.par.Hidebuiltin.eval())
        except AttributeError:
            pass

        # Receiver-specific state (only used when mode='Receiver')
        self._rx_dev_ptrs = [None] * self.num_slots  # Opened IPC mem pointers
        self._rx_ipc_handles = [None] * self.num_slots  # IPC mem handles read from SHM
        self._rx_ipc_events = [None] * self.num_slots  # Opened IPC events
        self._rx_ipc_version = 0
        self._rx_num_slots = 0
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

    def initialize(self, width: int, height: int, channels: int = 4, buffer_size: Optional[int] = None) -> bool:
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
            # Load CUDA runtime
            self.cuda = get_cuda_runtime()
            self._log("Loaded CUDA runtime", force=True)

            # Create dedicated non-blocking stream for IPC operations (fixes TD cudaMemory() perf warning)
            # Reuse existing stream on re-init to avoid leaks
            if not hasattr(self, "ipc_stream") or self.ipc_stream is None:
                self.ipc_stream = self.cuda.create_stream(flags=0x01)  # cudaStreamNonBlocking
                self._log(
                    f"Created IPC stream: 0x{int(self.ipc_stream.value):016x}",
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
                self._log(f"Created IPC handle for slot {slot} (128 bytes)")

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

        For each slot (192 bytes per slot):
        [20+slot*192 : 148+slot*192]   mem_handle (128B)
        [148+slot*192 : 212+slot*192]  event_handle (64B)

        [20+NUM_SLOTS*192]  shutdown flag (1B)
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

            # Write memory handle (128 bytes)
            mem_handle_bytes = bytes(self.ipc_handles[slot].internal)
            self.shm_handle.buf[base_offset : base_offset + 128] = mem_handle_bytes

            # Write event handle (64 bytes) if available
            if self.ipc_event_handles[slot]:
                event_handle_bytes = bytes(self.ipc_event_handles[slot].reserved)
                self.shm_handle.buf[base_offset + 128 : base_offset + 192] = event_handle_bytes
                self._log(f"Wrote slot {slot} handles: mem={len(mem_handle_bytes)}B, event={len(event_handle_bytes)}B")
            else:
                self._log(f"Wrote slot {slot} mem handle: {len(mem_handle_bytes)}B")

        self._log(
            f"Wrote all IPC handles v{new_version} to SharedMemory ({16 + self.num_slots * 192 + 1} bytes total)",
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

        # Determine dtype_code from data_size inference
        if self.width > 0 and self.height > 0 and self.channels > 0:
            bytes_per_pixel = self.data_size / (self.width * self.height)
            bytes_per_comp = bytes_per_pixel / self.channels
        else:
            bytes_per_comp = 4  # Default to float32

        if bytes_per_comp <= 1:
            dtype_code = DTYPE_UINT8
        elif bytes_per_comp <= 2:
            dtype_code = DTYPE_FLOAT16
        else:
            dtype_code = DTYPE_FLOAT32

        # Write metadata fields
        self.shm_handle.buf[metadata_offset : metadata_offset + 4] = struct.pack("<I", self.width)
        self.shm_handle.buf[metadata_offset + 4 : metadata_offset + 8] = struct.pack("<I", self.height)
        self.shm_handle.buf[metadata_offset + 8 : metadata_offset + 12] = struct.pack("<I", self.channels)
        self.shm_handle.buf[metadata_offset + 12 : metadata_offset + 16] = struct.pack("<I", dtype_code)
        self.shm_handle.buf[metadata_offset + 16 : metadata_offset + 20] = struct.pack("<I", self.data_size)

        self._log(
            f"Wrote metadata: {self.width}x{self.height}x{self.channels}, dtype={dtype_code}, size={self.data_size}B"
        )

    def export_frame(self, top_op: TOP) -> bool:
        """Export TOP texture via CUDA IPC.

        Args:
            top_op: TouchDesigner TOP operator to export

        Returns:
            True if export successful, False otherwise
        """
        # Check if Active parameter is enabled
        try:
            if not bool(self.ownerComp.par.Active.eval()):
                return False
        except AttributeError:
            pass  # No Active parameter, proceed with export

        # Start frame timer (only if verbose)
        if self.verbose_performance:
            frame_start = time.perf_counter()

        try:
            # Time cudaMemory() call (OpenGL→CUDA interop, suspected 4K bottleneck)
            if self.verbose_performance:
                cuda_mem_start = time.perf_counter()

            # Get TOP's CUDA memory (pass stream for proper synchronization per TD docs)
            cuda_mem = top_op.cudaMemory(
                stream=int(self.ipc_stream.value) if self._initialized else None,
            )

            if self.verbose_performance:
                cuda_mem_time = (time.perf_counter() - cuda_mem_start) * 1_000_000  # microseconds
                self.total_cuda_memory_time += cuda_mem_time

            if cuda_mem is None:
                self._log(f"Failed to get CUDA memory from {top_op}", force=True)
                return False

            # CRITICAL: Keep reference to prevent garbage collection
            self.cuda_mem_ref = cuda_mem

            # Get dimensions from cuda_mem.shape
            actual_width = cuda_mem.shape.width
            actual_height = cuda_mem.shape.height
            actual_channels = cuda_mem.shape.numComps
            actual_size = cuda_mem.size

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

            # Calculate current slot for ring buffer rotation
            slot = self.write_idx % self.num_slots

            # Time cudaMemcpyAsync D2D (non-blocking) - only if verbose
            if self.verbose_performance:
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

                # Always write producer timestamp (enables consumer latency measurement)
                timestamp_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE
                struct.pack_into("<d", self.shm_handle.buf, timestamp_offset, time.perf_counter())
            else:
                # FALLBACK: Conditional CPU synchronization
                if self.frame_count % self.sync_interval == 0:
                    self.cuda.synchronize()

            # Update write_idx in SharedMemory (must happen for both paths)
            self.write_idx += 1
            struct.pack_into("<I", self.shm_handle.buf, WRITE_IDX_OFFSET, self.write_idx)

            # Frame tracking
            self.frame_count += 1

            # Calculate total frame time (only if verbose)
            if self.verbose_performance:
                frame_time = (time.perf_counter() - frame_start) * 1_000_000
                self.total_export_time += frame_time

            # Detailed first-frame diagnostic (one-time, not affected by 100-frame interval)
            if self.verbose_performance and self.frame_count == 1:
                self._log(
                    f"FIRST FRAME: cudaMemory={cuda_mem_time:.1f}us, "
                    f"memcpy={memcpy_time:.1f}us, total={frame_time:.1f}us, "
                    f"res={actual_width}x{actual_height}, size={actual_size / (1024 * 1024):.1f}MB",
                    force=True,
                )

            # Log performance metrics every 100 frames (if verbose)
            if self.verbose_performance and self.frame_count % 100 == 0:
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

        cuda_valid = self._is_cuda_context_valid()
        if not cuda_valid:
            self._log("CUDA context already destroyed — skipping GPU cleanup", force=True)

        # STEP 1: Signal shutdown to consumer (before closing SharedMemory)
        if self.shm_handle and self.shm_handle.buf is not None:
            try:
                shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
                self.shm_handle.buf[shutdown_offset] = 1
                self._log("Shutdown signal sent to consumer", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Warning: Could not write shutdown signal: {e}", force=True)

        # STEP 1b: Zero out IPC handle bytes so any reader sees invalid handles.
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

        # STEP 2: Destroy IPC events (sender-side resources, safe to destroy)
        if cuda_valid and hasattr(self, "ipc_events") and self.cuda:
            for slot, event in enumerate(self.ipc_events):
                if event:
                    try:
                        self.cuda.destroy_event(event)
                        self._log(f"Destroyed IPC event slot {slot}", force=True)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error destroying event slot {slot}: {e}", force=True)

        # STEP 2b: Destroy GPU timing events (benchmarking resources)
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

        # STEP 3: Destroy dedicated IPC stream (set to None to prevent double-free)
        if cuda_valid and hasattr(self, "ipc_stream") and self.ipc_stream and self.cuda:
            try:
                self.cuda.destroy_stream(self.ipc_stream)
                self._log("Destroyed IPC stream", force=True)
                self.ipc_stream = None
            except (RuntimeError, OSError) as e:
                self._log(f"Error destroying IPC stream: {e}", force=True)

        # STEP 4: Close SharedMemory (but don't unlink yet)
        if self.shm_handle:
            try:
                self.shm_handle.close()
                self._log("Closed SharedMemory", force=True)
            except (OSError, BufferError) as e:
                self._log(f"Error closing SharedMemory: {e}", force=True)

        # STEP 5: Grace period for receiver to close IPC handles
        if cuda_valid:
            time.sleep(0.1)  # 100ms for receiver to detect shutdown and close handles

        # STEP 6: Free GPU buffers (now safe, receiver has closed IPC handles)
        if cuda_valid and hasattr(self, "dev_ptrs") and self.cuda:
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr:
                    try:
                        self.cuda.free(dev_ptr)
                        self._log(f"Freed GPU buffer slot {slot}", force=True)
                    except (RuntimeError, OSError) as e:
                        self._log(f"Error freeing GPU buffer slot {slot}: {e}", force=True)

        # STEP 7: Free any pending deferred resources
        if cuda_valid and hasattr(self, "_pending_free_ptrs"):
            self._deferred_free()

        # STEP 8: Unlink SharedMemory (sender is owner and should clean up)
        if hasattr(self, "shm_name"):
            try:
                from multiprocessing.shared_memory import SharedMemory

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

        self._initialized = False
        self._log("Sender cleanup complete", force=True)

    def import_frame(self, import_buffer: "TOP") -> bool:
        """Import frame from CUDA IPC into ImportBuffer (Script TOP).

        Can be called from:
        - Inside ImportBuffer's onCook callback (TD 2023+ compatibility)
        - Execute DAT onFrameStart with modoutsidecook enabled (TD 2025+)

        Args:
            import_buffer: The ImportBuffer Script TOP operator

        Returns:
            True if import successful, False otherwise.
        """
        # Check Active parameter
        try:
            if not bool(self.ownerComp.par.Active.eval()):
                return False
        except AttributeError:
            pass

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
            shutdown_offset = SHM_HEADER_SIZE + (self._rx_num_slots * SLOT_SIZE)
            if self.shm_handle.buf[shutdown_offset] == 1:
                self._log("Sender shutdown detected. Cleaning up.", force=True)
                self.cleanup_receiver()
                return False

            # Check for version change (sender re-initialized)
            current_version = struct.unpack_from("<Q", self.shm_handle.buf, VERSION_OFFSET)[0]
            if current_version != self._rx_ipc_version:
                self._log(
                    f"Sender re-initialized (v{self._rx_ipc_version} -> v{current_version}). Reconnecting...",
                    force=True,
                )
                self.cleanup_receiver()
                return False  # Will reinitialize on next frame

            # Read write_idx and calculate read slot
            write_idx = struct.unpack_from("<I", self.shm_handle.buf, WRITE_IDX_OFFSET)[0]
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
                # Fallback: still use stream sync (not device sync)
                self.cuda.stream_synchronize(self._rx_stream)

            # Copy CUDA memory into ImportBuffer texture using cached shape
            address = self._rx_dev_ptrs[read_slot].value
            import_buffer.copyCUDAMemory(
                address,
                self._rx_buffer_size,
                self._rx_cached_shape,
                stream=int(self._rx_stream.value),
            )

            self.frame_count += 1
            self._rx_last_write_idx = write_idx

            # Debug logging
            if self.verbose_performance and self.frame_count % 100 == 0:
                self._log(f"Frame {self.frame_count}: read_slot={read_slot}, write_idx={write_idx}")

            return True

        except (RuntimeError, OSError) as e:
            self._log(f"Import failed: {e}", force=True)

            traceback.print_exc()
            return False

    def update_receiver_resolution(self, import_buffer: "TOP") -> bool:
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
            self.cuda = get_cuda_runtime()

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

            # Read extended metadata (if available)
            shutdown_offset = SHM_HEADER_SIZE + (self._rx_num_slots * SLOT_SIZE)
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
                self._rx_stream = self.cuda.create_stream(flags=0x01)  # cudaStreamNonBlocking
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
                mem_handle_bytes = bytes(self.shm_handle.buf[base_offset : base_offset + 128])

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
                event_handle_bytes = bytes(self.shm_handle.buf[base_offset + 128 : base_offset + 192])
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
                DTYPE_FLOAT16: np_module.float16,
                DTYPE_UINT8: np_module.uint8,
            }
            np_dtype = dtype_map.get(self._rx_dtype_code, np_module.float32)

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
