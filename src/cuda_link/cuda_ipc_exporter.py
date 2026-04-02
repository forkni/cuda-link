"""
CUDA IPC Exporter for Python Process
Exports GPU memory FROM Python TO TouchDesigner via CUDA IPC handles.

Enables the reverse direction: Python AI pipeline → TouchDesigner display.
The TD side receives frames using CUDAIPCExtension in "Receiver" mode.

Usage:
    from cuda_link import CUDAIPCExporter

    # Export AI-generated frames to TouchDesigner
    with CUDAIPCExporter(
        shm_name="ai_output_ipc",
        height=512, width=512, channels=4, dtype="uint8",
    ) as exporter:
        exporter.initialize()
        while running:
            # output_tensor: (H, W, 4) uint8 BGRA on GPU
            exporter.export_frame(
                gpu_ptr=output_tensor.data_ptr(),
                size=output_tensor.nelement() * output_tensor.element_size(),
            )

Architecture:
    Python GPU tensor --> cudaMemcpy D2D --> Persistent IPC Ring Buffer
                                                     |
                                      IPC Handle in SharedMemory (v0.5.0 protocol)
                                                     |
                        TouchDesigner (CUDAIPCExtension Receiver) opens handle once
                        --> import_frame(script_top) --> copyCUDAMemory() per frame

Performance:
    - Initialization: ~1ms (buffer alloc + handle export)
    - Per-frame: ~177µs at 512x512 @ 60 FPS (includes producer-side stream_synchronize)
    - Per-frame: ~219µs at 1080p @ 60 FPS (async D2D + stream_synchronize + protocol writes)

Compatibility:
    - Protocol: v0.5.0 (byte-identical to CUDAIPCExtension/CUDAIPCImporter)
    - TD side: CUDAIPCExtension in "Receiver" mode reads the SharedMemory
    - Platform: Windows only (CUDA IPC limitation)
"""

from __future__ import annotations

import logging
import struct
import time
import traceback
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory

from .cuda_ipc_wrapper import cudaIpcMemHandle_t, get_cuda_runtime  # noqa: F401

logger = logging.getLogger(__name__)

# Protocol layout constants (must match CUDAIPCExtension and CUDAIPCImporter)
PROTOCOL_MAGIC = 0x43495043  # "CIPC" - protocol validation magic number
SHM_HEADER_SIZE = 20  # 4B magic + 8B version + 4B num_slots + 4B write_idx
SLOT_SIZE = 128  # 64B mem_handle + 64B event_handle
SHUTDOWN_FLAG_SIZE = 1
METADATA_SIZE = 20  # 4B width + 4B height + 4B num_comps + 4B dtype_code + 4B buffer_size
TIMESTAMP_SIZE = 8  # 8B float64 producer timestamp

# Pre-compiled struct objects for hot-path SHM reads/writes (~50-100ns saved per call vs format-string lookup)
_ST_U32 = struct.Struct("<I")  # uint32 LE (write_idx, num_slots, metadata fields)
_ST_U64 = struct.Struct("<Q")  # uint64 LE (version)
_ST_F64 = struct.Struct("<d")  # float64 LE (timestamp)
_ST_U8 = struct.Struct("<B")  # uint8 (shutdown_flag, magic byte)

# Data type codes (must match CUDAIPCExtension and CUDAIPCImporter)
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_UINT8 = 2
DTYPE_UINT16 = 3

_DTYPE_CODE_MAP = {
    "float32": DTYPE_FLOAT32,
    "float16": DTYPE_FLOAT16,
    "uint8": DTYPE_UINT8,
    "uint16": DTYPE_UINT16,
}

_DTYPE_ITEMSIZE_MAP = {
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "uint16": 2,
}


class CUDAIPCExporter:
    """Python-side exporter for CUDA IPC GPU memory.

    Sends GPU frames FROM Python TO TouchDesigner via CUDA IPC.
    Pairs with CUDAIPCExtension in "Receiver" mode on the TD side.

    Responsibilities:
    - Allocate persistent GPU ring buffer (cudaMalloc, 2 MiB aligned)
    - Export IPC handles + metadata via SharedMemory (v0.5.0 protocol, once at startup)
    - Per-frame: accept raw GPU pointer, async D2D memcpy to ring slot, record IPC event
    - 7-step cleanup: shutdown signal → events → stream → SHM close → grace → free → unlink

    Performance:
    - Initialization: ~1ms (buffer alloc + handle export)
    - Per-frame overhead: ~177µs at 512x512, ~219µs at 1080p @ 60 FPS (async D2D + stream_synchronize)
    - Zero CPU memory copies (GPU-direct)
    """

    def __init__(
        self,
        shm_name: str,
        height: int,
        width: int,
        channels: int = 4,
        dtype: str = "uint8",
        num_slots: int = 2,
        debug: bool = False,
    ) -> None:
        """Initialize CUDA IPC exporter.

        Args:
            shm_name: SharedMemory name. Must match the TD Receiver's Ipcmemname parameter.
            height: Frame height in pixels.
            width: Frame width in pixels.
            channels: Number of channels (default: 4 for BGRA/RGBA).
            dtype: Data type string: "float32", "float16", or "uint8" (default: "uint8").
            num_slots: Ring buffer slot count (default: 2 for double-buffering). Range: 1-10.
            debug: Enable verbose per-frame performance logging.

        Raises:
            ValueError: If dtype is unsupported or num_slots is out of range.
        """
        if dtype not in _DTYPE_CODE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype!r}. Must be one of {list(_DTYPE_CODE_MAP)}")
        if not (0 < num_slots <= 10):
            raise ValueError(f"num_slots must be 1-10, got {num_slots}")

        self.shm_name = shm_name
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = dtype
        self.num_slots = num_slots
        self.debug = debug

        # Derived sizes
        itemsize = _DTYPE_ITEMSIZE_MAP[dtype]
        self.data_size = height * width * channels * itemsize  # Actual data bytes

        # CUDA state
        self.cuda = None
        self._initialized = False
        self.ipc_stream = None  # Dedicated non-blocking CUDA stream
        self.source_sync_event = None  # Cross-stream sync event (GPU-side, non-blocking CPU)

        # Ring buffer state (arrays sized by num_slots)
        self.dev_ptrs: list = [None] * num_slots  # GPU buffer pointers
        self.ipc_handles: list = [None] * num_slots  # IPC memory handles
        self.ipc_events: list = [None] * num_slots  # IPC events for GPU sync
        self.ipc_event_handles: list = [None] * num_slots  # Exportable event handles
        self.write_idx: int = 0  # Monotonic frame counter

        # SharedMemory
        self.shm_handle: SharedMemory | None = None
        self.buffer_size: int = self.data_size  # Will be 2MiB-aligned in initialize()

        # Performance tracking
        self.frame_count: int = 0
        self.total_memcpy_us: float = 0.0
        self.total_export_us: float = 0.0
        self.total_stream_wait_us: float = 0.0
        self.total_record_event_us: float = 0.0
        self.total_shm_write_us: float = 0.0

        # Cached SharedMemory offsets (computed once in initialize(), constant thereafter)
        self._ts_offset: int = 0
        self._shutdown_offset: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Allocate GPU ring buffer, create IPC handles, write to SharedMemory.

        Must be called before export_frame(). Safe to call multiple times
        (idempotent — returns True if already initialized).

        Returns:
            True if initialization succeeded, False on error.
        """
        if self._initialized:
            logger.debug("Already initialized")
            return True

        try:
            # Load CUDA runtime
            self.cuda = get_cuda_runtime()
            logger.info("Loaded CUDA runtime")

            # Create or reuse dedicated non-blocking IPC stream
            if self.ipc_stream is None:
                self.ipc_stream = self.cuda.create_stream(flags=0x01)  # cudaStreamNonBlocking
                logger.info("Created IPC stream: 0x%016x", int(self.ipc_stream.value))
            else:
                logger.debug("Reusing IPC stream: 0x%016x", int(self.ipc_stream.value))

            # Create cross-stream sync event for GPU-side producer ordering.
            # ipc_stream is non-blocking and does NOT implicitly sync with any other stream,
            # so callers MUST either call record_source_sync() or torch.cuda.synchronize()
            # before export_frame(). The event enables the GPU-side-only path.
            if self.source_sync_event is None:
                self.source_sync_event = self.cuda.create_sync_event()
                logger.info("Created cross-stream source sync event")

            # Apply 2 MiB alignment (NVIDIA requirement: prevents information disclosure)
            alignment = 2 * 1024 * 1024
            self.buffer_size = ((self.data_size + alignment - 1) // alignment) * alignment
            logger.info(
                "Buffer: %.1f KB data, %.1f KB aligned, %d slots",
                self.data_size / 1024,
                self.buffer_size / 1024,
                self.num_slots,
            )

            # PHASE 1: Allocate GPU ring buffer + create IPC handles
            for slot in range(self.num_slots):
                self.dev_ptrs[slot] = self.cuda.malloc(self.buffer_size)
                logger.info(
                    "Slot %d: allocated %.1f KB at 0x%016x",
                    slot,
                    self.buffer_size / 1024,
                    self.dev_ptrs[slot].value,
                )

                # Memory handle (once at startup — reused every frame)
                self.ipc_handles[slot] = self.cuda.ipc_get_mem_handle(self.dev_ptrs[slot])
                logger.debug("Slot %d: created IPC mem handle (64 bytes)", slot)

                # Event handle for GPU-side synchronization
                self.ipc_events[slot] = self.cuda.create_ipc_event()
                self.ipc_event_handles[slot] = self.cuda.ipc_get_event_handle(self.ipc_events[slot])
                logger.debug("Slot %d: created IPC event (64 bytes)", slot)

            logger.info("Created %d IPC buffer slots with GPU-side sync", self.num_slots)

            # PHASE 2: Create SharedMemory
            shm_size = (
                SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
            )
            try:
                self.shm_handle = SharedMemory(name=self.shm_name)
                logger.info("Opened existing SharedMemory: %s", self.shm_name)
            except FileNotFoundError:
                self.shm_handle = SharedMemory(name=self.shm_name, create=True, size=shm_size)
                logger.info("Created SharedMemory: %s (%d bytes)", self.shm_name, shm_size)

            # PHASE 3: Write protocol header + IPC handles + metadata
            self._write_handles_to_shm()
            self._write_metadata_to_shm()

            # Cache constant SharedMemory offsets so export_frame() avoids per-frame arithmetic
            self._ts_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

            self._initialized = True
            logger.info("Initialization complete — ready for zero-copy GPU transfer")
            return True

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Initialization failed: %s", e)
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Protocol write helpers
    # ------------------------------------------------------------------

    def _write_handles_to_shm(self) -> None:
        """Write v0.5.0 protocol header + IPC handles to SharedMemory.

        Layout:
            [0-3]    magic (uint32 LE) = 0x43495043
            [4-11]   version (uint64 LE) — incremented on each init
            [12-15]  num_slots (uint32 LE)
            [16-19]  write_idx (uint32 LE) — initialized to 0

        Per slot (128 bytes):
            [base..+63]   cudaIpcMemHandle_t (64 bytes)
            [base+64..+64] cudaIpcEventHandle_t (64 bytes)

        Footer:
            shutdown_flag (1 byte) = 0
        """
        if self.shm_handle is None or not all(self.ipc_handles):
            return

        # Increment version (detect re-initialization from consumer side)
        try:
            current_version = struct.unpack_from("<Q", self.shm_handle.buf, 4)[0]
        except (struct.error, ValueError, IndexError):
            current_version = 0
        new_version = current_version + 1

        # Write header
        struct.pack_into("<I", self.shm_handle.buf, 0, PROTOCOL_MAGIC)
        struct.pack_into("<Q", self.shm_handle.buf, 4, new_version)
        struct.pack_into("<I", self.shm_handle.buf, 12, self.num_slots)
        struct.pack_into("<I", self.shm_handle.buf, 16, 0)  # write_idx = 0 initially

        # Write per-slot handles
        for slot in range(self.num_slots):
            base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)

            mem_handle_bytes = bytes(self.ipc_handles[slot].internal)
            self.shm_handle.buf[base_offset : base_offset + 64] = mem_handle_bytes

            if self.ipc_event_handles[slot]:
                event_handle_bytes = bytes(self.ipc_event_handles[slot].reserved)
                self.shm_handle.buf[base_offset + 64 : base_offset + 128] = event_handle_bytes

        # Initialize shutdown flag to 0 and cache its offset for export_frame() reassertion
        self._shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
        struct.pack_into("<B", self.shm_handle.buf, self._shutdown_offset, 0)

        logger.info("Wrote IPC handles v%d to SharedMemory", new_version)

    def _write_metadata_to_shm(self) -> None:
        """Write texture metadata to the extended protocol region.

        Layout (20 bytes after shutdown flag):
            +0   width (uint32)
            +4   height (uint32)
            +8   num_comps (uint32)
            +12  dtype_code (uint32)  — 0=float32, 1=float16, 2=uint8
            +16  data_size (uint32)   — actual bytes (before alignment)
        """
        if self.shm_handle is None or self.data_size == 0:
            return

        shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
        metadata_offset = shutdown_offset + SHUTDOWN_FLAG_SIZE

        dtype_code = _DTYPE_CODE_MAP[self.dtype]

        struct.pack_into("<I", self.shm_handle.buf, metadata_offset, self.width)
        struct.pack_into("<I", self.shm_handle.buf, metadata_offset + 4, self.height)
        struct.pack_into("<I", self.shm_handle.buf, metadata_offset + 8, self.channels)
        struct.pack_into("<I", self.shm_handle.buf, metadata_offset + 12, dtype_code)
        struct.pack_into("<I", self.shm_handle.buf, metadata_offset + 16, self.data_size)

        logger.debug(
            "Wrote metadata: %dx%dx%d, dtype=%s(%d), data_size=%dB",
            self.width,
            self.height,
            self.channels,
            self.dtype,
            dtype_code,
            self.data_size,
        )

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def record_source_sync(self, producer_stream_handle: int) -> None:
        """Record sync event on the producer's CUDA stream (GPU-side, non-blocking CPU).

        Call this AFTER your GPU kernel writes to the source buffer, BEFORE export_frame().
        export_frame() will make ipc_stream wait for this event GPU-side before the D2D
        memcpy, ensuring source data is ready without blocking the CPU.

        This replaces ``torch.cuda.synchronize()`` and saves ~0.2-0.5ms per frame.

        ``ipc_stream`` is created with cudaStreamNonBlocking and does NOT implicitly
        synchronize with any other stream (including the legacy default stream). Without
        this call, export_frame() has no ordering guarantee with the caller's GPU work.

        Args:
            producer_stream_handle: Raw CUDA stream integer. Examples:
                - PyTorch:  ``torch.cuda.current_stream().cuda_stream``
                - CuPy:     ``cupy.cuda.get_current_stream().ptr``
                - Raw CUDA: the ``cudaStream_t`` cast to int

        If this method is never called, ``source_sync_event`` stays in its initial
        unrecorded state and stream_wait_event() in export_frame() is a benign no-op
        (backward compatible). The caller is then responsible for their own sync.
        """
        if self.source_sync_event is not None and self.cuda is not None:
            from .cuda_ipc_wrapper import CUDAStream_t

            self.cuda.record_event(self.source_sync_event, CUDAStream_t(producer_stream_handle))

    def export_frame(self, gpu_ptr: int, size: int) -> bool:
        """Export one frame from GPU memory via IPC ring buffer.

        Call this every frame after your GPU kernel produces output.
        For correct cross-stream ordering without CPU blocking, call
        record_source_sync(stream_handle) immediately before this.
        The TD Receiver will pick it up within ~16ms (1 frame at 60 FPS).

        Args:
            gpu_ptr: Source GPU pointer (from tensor.data_ptr()).
            size: Buffer size in bytes (from tensor.nelement() * tensor.element_size()).

        Returns:
            True if export succeeded, False on error.
        """
        if not self._initialized:
            logger.warning("Not initialized — call initialize() first")
            return False

        if self.debug:
            frame_start = time.perf_counter()

        try:
            slot = self.write_idx % self.num_slots

            # GPU-side wait: ipc_stream waits for source data to be ready (non-blocking CPU).
            # Only active if record_source_sync() was called; otherwise this is a no-op.
            if self.debug:
                _t = time.perf_counter()
            if self.source_sync_event is not None:
                self.cuda.stream_wait_event(self.ipc_stream, self.source_sync_event, 0)
            if self.debug:
                self.total_stream_wait_us += (time.perf_counter() - _t) * 1_000_000

            # Async D2D copy to this slot's persistent IPC buffer
            if self.debug:
                memcpy_start = time.perf_counter()
            self.cuda.memcpy_async(
                dst=self.dev_ptrs[slot],
                src=c_void_p(gpu_ptr),
                count=self.data_size,
                kind=3,  # cudaMemcpyDeviceToDevice
                stream=self.ipc_stream,
            )
            if self.debug:
                self.total_memcpy_us += (time.perf_counter() - memcpy_start) * 1_000_000

            # Record IPC event (stream-ordered, signals consumer that data is ready)
            if self.debug:
                _t = time.perf_counter()
            if self.ipc_events[slot]:
                self.cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)
            if self.debug:
                self.total_record_event_us += (time.perf_counter() - _t) * 1_000_000

            # Synchronize ipc_stream to ensure the D2D copy + event record have EXECUTED on
            # the GPU before publishing write_idx. Without this, the consumer's query_event()
            # may return False (event not yet signaled) even though write_idx is visible, because
            # the GPU executes stream operations asynchronously. Blocking here guarantees the
            # IPC event is pre-signaled when the consumer reads write_idx, so query_event()
            # returns True on the first call (no polling delay). Cost: ~D2D GPU time per frame
            # (~13us at 512x512, ~100us at 1080p) — acceptable for the ordering guarantee.
            self.cuda.stream_synchronize(self.ipc_stream)

            # Write timestamp, clear shutdown_flag, then publish write_idx LAST.
            # Ordering matters: the consumer reads shutdown_flag BEFORE write_idx, so
            # clearing it before incrementing write_idx ensures the consumer always sees
            # shutdown_flag=0 when it detects a new frame (atomicity improvement).
            if self.debug:
                _t = time.perf_counter()
            self.write_idx += 1
            _ST_F64.pack_into(self.shm_handle.buf, self._ts_offset, time.perf_counter())
            self.shm_handle.buf[self._shutdown_offset] = 0
            _ST_U32.pack_into(self.shm_handle.buf, 16, self.write_idx)  # publish last
            if self.debug:
                self.total_shm_write_us += (time.perf_counter() - _t) * 1_000_000

            self.frame_count += 1

            if self.debug:
                frame_time = (time.perf_counter() - frame_start) * 1_000_000
                self.total_export_us += frame_time

                if self.frame_count % 97 == 0:
                    n = self.frame_count
                    logger.debug(
                        "Frame %d: slot=%d | stream_wait=%.1fus memcpy=%.1fus "
                        "record_event=%.1fus shm_write=%.1fus | total=%.1fus",
                        n,
                        slot,
                        self.total_stream_wait_us / n,
                        self.total_memcpy_us / n,
                        self.total_record_event_us / n,
                        self.total_shm_write_us / n,
                        self.total_export_us / n,
                    )

            return True

        except (OSError, RuntimeError) as e:
            logger.error("Export failed: %s", e)
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _is_cuda_context_valid(self) -> bool:
        """Check if CUDA context is still valid.

        The CUDA context may be destroyed before cleanup() is called
        (e.g., when the Python process is terminating). Checking this
        prevents spurious CUDA errors during cleanup.
        """
        if self.cuda is None:
            return False
        try:
            self.cuda.cudart.cudaGetLastError()
            return True
        except (OSError, RuntimeError):
            return False

    def cleanup(self) -> None:
        """Cleanup all CUDA IPC resources.

        7-step shutdown sequence (order is critical):
        1. Signal shutdown to consumer via SharedMemory flag
        2. Destroy IPC events (sender-side resources, safe to destroy)
        3. Destroy IPC stream
        4. Close SharedMemory (don't unlink yet — consumer may still read flag)
        5. Grace period (100ms) for consumer to detect shutdown and close handles
        6. Free GPU buffers (cudaFree blocks until consumer closes IPC handles)
        7. Unlink SharedMemory (producer owns it and is responsible for cleanup)
        """
        # Double-cleanup guard: skip if already cleaned up
        if not self._initialized and self.shm_handle is None:
            return

        cuda_valid = self._is_cuda_context_valid()
        if not cuda_valid:
            logger.warning("CUDA context already destroyed — skipping GPU cleanup")

        # STEP 1: Signal shutdown to consumer
        if self.shm_handle:
            try:
                shutdown_offset = SHM_HEADER_SIZE + (self.num_slots * SLOT_SIZE)
                struct.pack_into("<B", self.shm_handle.buf, shutdown_offset, 1)
                logger.info("Shutdown signal sent to consumer")
            except (OSError, BufferError) as e:
                logger.warning("Could not write shutdown signal: %s", e)

        # STEP 1b: Zero out IPC handle bytes so any reader sees invalid handles.
        # On Windows, unlink() is a no-op (SharedMemory uses CreateFileMapping kernel
        # objects), so the SharedMemory may persist with stale non-zero handles that
        # pass the all-zero validation check and trigger error 201 on open.
        if self.shm_handle:
            try:
                for slot in range(self.num_slots):
                    base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)
                    self.shm_handle.buf[base_offset : base_offset + SLOT_SIZE] = b"\x00" * SLOT_SIZE
                logger.debug("Zeroed IPC handle bytes in SharedMemory")
            except (OSError, BufferError) as e:
                logger.warning("Could not zero IPC handles: %s", e)

        # STEP 2: Destroy IPC events + cross-stream sync event
        if cuda_valid and self.cuda and self.ipc_events:
            for slot, event in enumerate(self.ipc_events):
                if event:
                    try:
                        self.cuda.destroy_event(event)
                        logger.debug("Destroyed IPC event slot %d", slot)
                    except (RuntimeError, OSError) as e:
                        logger.error("Error destroying event slot %d: %s", slot, e)

        if cuda_valid and self.cuda and self.source_sync_event:
            try:
                self.cuda.destroy_event(self.source_sync_event)
                logger.debug("Destroyed cross-stream sync event")
            except (RuntimeError, OSError) as e:
                logger.error("Error destroying sync event: %s", e)
            self.source_sync_event = None

        # STEP 3: Destroy IPC stream
        if cuda_valid and self.cuda and self.ipc_stream:
            try:
                self.cuda.destroy_stream(self.ipc_stream)
                logger.info("Destroyed IPC stream")
                self.ipc_stream = None
            except (RuntimeError, OSError) as e:
                logger.error("Error destroying IPC stream: %s", e)

        # STEP 4: Close SharedMemory (don't unlink yet)
        if self.shm_handle:
            try:
                self.shm_handle.close()
                logger.debug("Closed SharedMemory")
            except (OSError, BufferError) as e:
                logger.error("Error closing SharedMemory: %s", e)
            self.shm_handle = None

        # STEP 5: Grace period for consumer to close IPC handles
        if cuda_valid:
            time.sleep(0.1)  # 100ms for consumer to detect shutdown and close handles

        # STEP 6: Free GPU buffers (safe now that consumer has closed handles)
        if cuda_valid and self.cuda and self.dev_ptrs:
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr:
                    try:
                        self.cuda.free(dev_ptr)
                        logger.debug("Freed GPU buffer slot %d", slot)
                    except (RuntimeError, OSError) as e:
                        logger.error("Error freeing GPU buffer slot %d: %s", slot, e)

        # STEP 7: Unlink SharedMemory (producer is owner and responsible for cleanup)
        try:
            shm_temp = SharedMemory(name=self.shm_name)
            shm_temp.close()
            shm_temp.unlink()
            logger.info("Unlinked SharedMemory")
        except FileNotFoundError:
            pass  # Already unlinked
        except (OSError, RuntimeError) as e:
            logger.warning("Could not unlink SharedMemory: %s", e)

        # Reset all state to prevent double-free on re-entry
        self.dev_ptrs = [None] * self.num_slots
        self.ipc_events = [None] * self.num_slots
        self.ipc_handles = [None] * self.num_slots
        self.ipc_event_handles = [None] * self.num_slots
        self._initialized = False

        logger.info("Cleanup complete")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> CUDAIPCExporter:
        """Enter context manager — returns self for use in 'with' statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager — cleanup resources regardless of exception."""
        self.cleanup()
        return None  # Don't suppress exceptions

    def __del__(self) -> None:
        """Destructor — cleanup on garbage collection if not already done."""
        if getattr(self, "_initialized", False):
            self.cleanup()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Check if exporter is ready to export frames.

        Returns:
            True if initialized with all GPU buffers allocated.
        """
        return self._initialized and all(ptr is not None for ptr in self.dev_ptrs)

    def get_stats(self) -> dict:
        """Get exporter statistics for monitoring.

        Returns:
            Dictionary with current exporter state and performance metrics.
        """
        avg_memcpy = self.total_memcpy_us / self.frame_count if self.frame_count > 0 else 0.0
        avg_total = self.total_export_us / self.frame_count if self.frame_count > 0 else 0.0
        return {
            "initialized": self._initialized,
            "shm_name": self.shm_name,
            "resolution": f"{self.width}x{self.height}x{self.channels}",
            "dtype": self.dtype,
            "num_slots": self.num_slots,
            "data_size_kb": self.data_size / 1024,
            "buffer_size_mb": self.buffer_size / (1024 * 1024),
            "frame_count": self.frame_count,
            "write_idx": self.write_idx,
            "avg_memcpy_us": avg_memcpy,
            "avg_total_us": avg_total,
            "dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self.dev_ptrs],
        }
