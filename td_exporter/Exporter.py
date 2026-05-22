"""
Exporter — deep module for zero-copy GPU frame export via CUDA IPC.

Public surface:
  Exporter.open(spec, *, policy, cuda) -> Exporter
  exporter.export(frame) -> FrameOutcome
  exporter.close() -> None
  exporter.record_source_sync(producer_stream_handle) -> None  (GPU-side ordering)

Context manager: ``with Exporter.open(...) as exp: ...``

See _exporter_port.py for FrameSpec, ExportPolicy, GpuFrame, FrameOutcome, CudaPort.
See _cuda_adapters.py for CTypesCudaAdapter (production) and FakeCudaAdapter (tests).
"""

from __future__ import annotations

import contextlib
import logging
import struct
import threading
import time
import traceback
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import NVTXShim as _nvtx
from CudaAdapters import CTypesCudaAdapter
from ExporterPort import (
    CudaPort,
    ExportPolicy,
    FrameOutcome,
    FrameSpec,
    GpuFrame,
)
from FrameProfile import FrameProfile
from ActivationBarrier import CheckerBarrier
from CUDARuntimeTypes import (
    CUDART_GRAPHS_MIN_VERSION,
    CUDAGraph_t,
    CUDAGraphExec_t,
    CUDAGraphNode_t,
    CUDAStream_t,
)
from SHMProtocol import (
    _ST_U32,
    MAGIC_OFFSET,
    METADATA_SIZE,
    NUM_SLOTS_OFFSET,
    PROTOCOL_MAGIC,
    SHM_HEADER_SIZE,
    SHUTDOWN_FLAG_SIZE,
    SLOT_SIZE,
    TIMESTAMP_SIZE,
    WRITE_IDX_OFFSET,
    DtypeCodec,
    Metadata,
    SHMLayout,
    bump_version,
    publish_frame,
)

if TYPE_CHECKING:
    from .nvml_observer import NVMLObserver

logger = logging.getLogger(__name__)

_NVTX_EXPORTER_SLOT_NAMES: tuple[str, ...] = _nvtx.slot_names("cudalink.exporter.slot")

_EXPORTER_PROFILE_REGIONS: tuple[str, ...] = (
    "stream_wait",
    "memcpy",
    "record_event",
    "shm_write",
    "export",
    "sync",
    "sticky_check",
    "flush_probe",
    "ptr_cache_miss",
)

_DTYPE_ITEMSIZE_MAP = {
    "float32": 4,
    "float16": 2,
    "uint8": 1,
    "uint16": 2,
}
_DTYPE_TO_KIND_BITS = {k: DtypeCodec.encode(k) for k in ("float32", "float16", "uint8", "uint16")}


def _read_hws_mode() -> str:
    try:
        import winreg  # noqa: PLC0415

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers")
        value, _ = winreg.QueryValueEx(key, "HwSchMode")
        winreg.CloseKey(key)
        return str(value)
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class Exporter:
    """Deep module: zero-copy GPU frame export over a CUDA IPC ring buffer.

    Responsibilities:
    - Allocate a persistent GPU ring buffer (cudaMalloc, 2 MiB aligned).
    - Export IPC handles + metadata via SharedMemory (v0.5.0 protocol, once at open).
    - Per-frame: accept a GpuFrame, async D2D memcpy to ring slot, record IPC event.
    - 7-step close: shutdown signal → events → stream → SHM close → grace → free → unlink.

    Construction: always via Exporter.open() — the classmethod is the only valid
    constructor because ``__init__`` alone leaves the instance uninitialized.
    """

    def __init__(self, spec: FrameSpec, policy: ExportPolicy, cuda: CudaPort) -> None:
        if spec.dtype not in _DTYPE_TO_KIND_BITS:
            raise ValueError(f"Unsupported dtype: {spec.dtype!r}. Must be one of {list(_DTYPE_TO_KIND_BITS)}")
        if not (0 < spec.num_slots <= 10):
            raise ValueError(f"num_slots must be 1-10, got {spec.num_slots}")

        self._spec = spec
        self._policy = policy
        self._cuda = cuda

        self._initialized = False
        self._closed = False

        # GPU graphs state
        self._graphs_disabled: bool = False
        self._source_sync_recorded: bool = False
        self._graph_execs: list[CUDAGraphExec_t | None] = [None] * spec.num_slots
        self._graph_templates: list[CUDAGraph_t | None] = [None] * spec.num_slots
        self._graph_memcpy_nodes: list[CUDAGraphNode_t | None] = [None] * spec.num_slots

        # CUDA handles (set during _initialize)
        self.ipc_stream = None
        self.source_sync_event = None
        self.dev_ptrs: list = [None] * spec.num_slots
        self.ipc_handles: list = [None] * spec.num_slots
        self.ipc_events: list = [None] * spec.num_slots
        self.ipc_event_handles: list = [None] * spec.num_slots
        self.write_idx: int = 0

        # SharedMemory
        self.shm_handle: SharedMemory | None = None
        itemsize = _DTYPE_ITEMSIZE_MAP[spec.dtype]
        self.data_size: int = spec.height * spec.width * spec.channels * itemsize
        self.buffer_size: int = self.data_size  # 2 MiB-aligned in _initialize

        # Profiling
        self.frame_count: int = 0
        self._profile: FrameProfile = FrameProfile(_EXPORTER_PROFILE_REGIONS)

        # Cached layout offsets (set by _write_handles_to_shm)
        self._layout: SHMLayout = SHMLayout(spec.num_slots)
        self._ts_offset: int = self._layout.timestamp_offset
        self._shutdown_offset: int = self._layout.shutdown_offset

        # Device-affinity cache
        self._source_sync_device_warned: bool = False
        self._ptr_device_cache: set[int] = set()

        # Activation barrier
        self._barrier = CheckerBarrier(enabled=policy.barrier_enabled, stale_ns=policy.barrier_stale_ns)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        spec: FrameSpec,
        *,
        policy: ExportPolicy | None = None,
        cuda: CudaPort | None = None,
        barrier: CheckerBarrier | None = None,
    ) -> Exporter:
        """Construct and fully initialize an Exporter. Raises on failure; leaves no half-state.

        Args:
            spec:    Frame geometry + SHM routing (FrameSpec).
            policy:  Behavioural knobs. None → ExportPolicy.from_env().
            cuda:    CudaPort adapter. None → CTypesCudaAdapter.for_device(spec.device).
                     Pass a FakeCudaAdapter() for unit tests.
            barrier: CheckerBarrier to use. None → construct from policy.
                     Pass CheckerBarrier(shm=FakeShmAdapter(...)) for unit tests.

        Returns:
            A ready-to-use Exporter.

        Raises:
            ValueError: dtype or num_slots invalid, or device mismatch.
            RuntimeError: CUDA or SHM allocation failed.
            OSError: SHM create/open failed.
        """
        if policy is None:
            policy = ExportPolicy.from_env()
        if cuda is None:
            cuda = CTypesCudaAdapter.for_device(spec.device)

        exp = cls(spec, policy, cuda)
        if barrier is not None:
            exp._barrier.close()
            exp._barrier = barrier
        try:
            exp._initialize()
        except Exception:
            exp._do_cleanup(cuda_valid=False)
            raise
        return exp

    # ------------------------------------------------------------------
    # Initialization (called only from open())
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        actual_device = self._cuda.get_device()
        if actual_device != self._spec.device:
            raise RuntimeError(
                f"Device mismatch: requested device {self._spec.device} but CUDA context "
                f"is bound to device {actual_device}. Ensure no other code calls "
                "cudaSetDevice() with a different index before Exporter.open()."
            )
        logger.info("Loaded CUDA runtime on device %d", actual_device)

        hws_mode = _read_hws_mode()
        logger.info("WDDM HwSchMode: %s (0=software, 2=hardware/GPU-P, unknown=non-Windows)", hws_mode)
        with _nvtx.annotate(f"cudalink.startup.hws_mode={hws_mode}", "cyan"):
            pass

        if self.ipc_stream is None:
            if self._policy.high_priority_stream:
                self.ipc_stream = self._cuda.create_stream_with_priority(flags=0x01)
                logger.info("Created IPC stream (high-priority): %s", self.ipc_stream)
            else:
                self.ipc_stream = self._cuda.create_stream(flags=0x01)
                logger.info("Created IPC stream (normal-priority): %s", self.ipc_stream)
        else:
            logger.debug("Reusing IPC stream: %s", self.ipc_stream)

        if self.source_sync_event is None:
            self.source_sync_event = self._cuda.create_sync_event()
            logger.info("Created cross-stream source sync event")

        alignment = 2 * 1024 * 1024
        self.buffer_size = ((self.data_size + alignment - 1) // alignment) * alignment
        logger.info(
            "Buffer: %.1f KB data, %.1f KB aligned, %d slots",
            self.data_size / 1024,
            self.buffer_size / 1024,
            self._spec.num_slots,
        )

        # Re-assert primary context before allocating IPC buffers.  TD's cook thread may
        # have entered TD's interop context (via top.cudaMemory / cudaGraphicsMap*) between
        # CUDARuntimeAPI.__init__ and here; cudaMalloc in that context mints IPC handles
        # that are bound to the interop context and cannot be opened by a second process.
        self._cuda.set_device(self._spec.device)

        for slot in range(self._spec.num_slots):
            self.dev_ptrs[slot] = self._cuda.malloc(self.buffer_size)
            logger.info(
                "Slot %d: allocated %.1f KB at 0x%016x", slot, self.buffer_size / 1024, self.dev_ptrs[slot].value
            )
            self.ipc_handles[slot] = self._cuda.ipc_get_mem_handle(self.dev_ptrs[slot])
            logger.debug("Slot %d: created IPC mem handle (64 bytes)", slot)
            self.ipc_events[slot] = self._cuda.create_ipc_event()
            self.ipc_event_handles[slot] = self._cuda.ipc_get_event_handle(self.ipc_events[slot])
            logger.debug("Slot %d: created IPC event (64 bytes)", slot)

        logger.info("Created %d IPC buffer slots with GPU-side sync", self._spec.num_slots)

        shm_size = (
            SHM_HEADER_SIZE + (self._spec.num_slots * SLOT_SIZE) + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE
        )
        try:
            self.shm_handle = SharedMemory(name=self._spec.shm_name)
            logger.info("Opened existing SharedMemory: %s", self._spec.shm_name)
        except FileNotFoundError:
            self.shm_handle = SharedMemory(name=self._spec.shm_name, create=True, size=shm_size)
            logger.info("Created SharedMemory: %s (%d bytes)", self._spec.shm_name, shm_size)

        self._write_handles_to_shm()
        self._write_metadata_to_shm()
        self._ts_offset = self._layout.timestamp_offset
        self._initialized = True

        if self._policy.use_graphs:
            try:
                rt_version = self._cuda.get_runtime_version()
            except (RuntimeError, OSError) as exc:
                rt_version = 0
                logger.warning("cudaRuntimeGetVersion failed (%s) — disabling graphs", exc)
            if rt_version >= CUDART_GRAPHS_MIN_VERSION:
                self._build_export_graphs()
            else:
                logger.warning(
                    "ExportPolicy.use_graphs=True ignored: cudart %d < %d "
                    "(cudaGraphInstantiateWithFlags requires 11.4+). "
                    "Falling back to legacy stream path.",
                    rt_version,
                    CUDART_GRAPHS_MIN_VERSION,
                )
                self._graphs_disabled = True

        logger.info("Initialization complete — ready for zero-copy GPU transfer")

    # ------------------------------------------------------------------
    # Protocol write helpers
    # ------------------------------------------------------------------

    def _write_handles_to_shm(self) -> None:
        if self.shm_handle is None or not all(self.ipc_handles):
            return
        _ST_U32.pack_into(self.shm_handle.buf, MAGIC_OFFSET, PROTOCOL_MAGIC)
        new_version = bump_version(self.shm_handle.buf)
        _ST_U32.pack_into(self.shm_handle.buf, NUM_SLOTS_OFFSET, self._spec.num_slots)
        _ST_U32.pack_into(self.shm_handle.buf, WRITE_IDX_OFFSET, 0)
        for slot in range(self._spec.num_slots):
            base_offset = SHM_HEADER_SIZE + (slot * SLOT_SIZE)
            _mem_bytes = bytes(self.ipc_handles[slot].internal)
            print(f"[SENDER-HEX] slot{slot} mem handle: {_mem_bytes[:16].hex()}...", flush=True)
            self.shm_handle.buf[base_offset : base_offset + 64] = _mem_bytes
            if self.ipc_event_handles[slot]:
                self.shm_handle.buf[base_offset + 64 : base_offset + 128] = bytes(self.ipc_event_handles[slot].reserved)
        self._layout = SHMLayout(self._spec.num_slots)
        self._shutdown_offset = self._layout.shutdown_offset
        self.shm_handle.buf[self._shutdown_offset] = 0
        logger.info("Wrote IPC handles v%d to SharedMemory", new_version)

    def _write_metadata_to_shm(self) -> None:
        if self.shm_handle is None or self.data_size == 0:
            return
        kind, bits, flags = DtypeCodec.encode(self._spec.dtype)
        Metadata(
            width=self._spec.width,
            height=self._spec.height,
            num_comps=self._spec.channels,
            format_kind=kind,
            bits_per_comp=bits,
            flags=flags,
            data_size=self.data_size,
        ).pack_into(self.shm_handle.buf, self._layout)

    # ------------------------------------------------------------------
    # CUDA Graph helpers
    # ------------------------------------------------------------------

    def _build_export_graphs(self) -> None:
        placeholder_src = self.dev_ptrs[0]
        for slot in range(self._spec.num_slots):
            capture_started = False
            try:
                self._cuda.stream_begin_capture(self.ipc_stream, mode=2)
                capture_started = True
                self._cuda.memcpy_async(
                    dst=self.dev_ptrs[slot],
                    src=placeholder_src,
                    count=self.data_size,
                    kind=3,
                    stream=self.ipc_stream,
                )
                template_graph = self._cuda.stream_end_capture(self.ipc_stream)
                capture_started = False
                nodes = self._cuda.graph_get_nodes(template_graph)
                if len(nodes) != 1:
                    self._cuda.graph_destroy(template_graph)
                    raise RuntimeError(f"Unexpected graph node count {len(nodes)} (expected 1: MemcpyNode).")
                graph_exec = self._cuda.graph_instantiate(template_graph)
                self._graph_execs[slot] = graph_exec
                self._graph_templates[slot] = template_graph
                self._graph_memcpy_nodes[slot] = nodes[0]
                logger.debug("Built export graph for slot %d (1-node: Memcpy)", slot)
            except (RuntimeError, OSError) as exc:
                if capture_started:
                    with contextlib.suppress(RuntimeError, OSError):
                        abandoned = self._cuda.stream_end_capture(self.ipc_stream)
                        self._cuda.graph_destroy(abandoned)
                logger.warning(
                    "CUDA Graph build failed for slot %d (%s) — "
                    "disabling graphs, falling back to legacy stream path. "
                    "Set ExportPolicy(use_graphs=False) to suppress.",
                    slot,
                    exc,
                )
                self._graphs_disabled = True
                self._destroy_export_graphs()
                return
        logger.info("CUDA export graphs built for %d slots (use_graphs=True)", self._spec.num_slots)

    def _destroy_export_graphs(self) -> None:
        for slot, graph_exec in enumerate(self._graph_execs):
            if graph_exec is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self._cuda.graph_exec_destroy(graph_exec)
                self._graph_execs[slot] = None
        for slot, template in enumerate(getattr(self, "_graph_templates", [])):
            if template is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    self._cuda.graph_destroy(template)
                self._graph_templates[slot] = None
        self._graph_memcpy_nodes = [None] * self._spec.num_slots

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_source_sync(self, producer_stream_handle: int) -> None:
        """Record the cross-stream sync event on the producer's CUDA stream.

        Call this AFTER your GPU kernel writes to the source buffer and BEFORE export().
        The Exporter's IPC stream will then GPU-wait on this event before the D2D memcpy,
        preserving ordering without blocking the CPU.

        Equivalent to passing ``producer_stream=producer_stream_handle`` in GpuFrame —
        this method is provided for callers who prefer the two-step pattern.

        Args:
            producer_stream_handle: Raw stream handle as int.
                PyTorch: ``torch.cuda.current_stream().cuda_stream``
                CuPy:    ``cupy.cuda.get_current_stream().ptr``
        """
        if self.source_sync_event is not None:
            if not self._source_sync_device_warned:
                current_device = self._cuda.get_device()
                if current_device != self._spec.device:
                    msg = (
                        f"record_source_sync: current CUDA device ({current_device}) "
                        f"does not match exporter device ({self._spec.device}). "
                        "Call cudaSetDevice(device) before creating your producer stream. "
                        "Set ExportPolicy(strict_device=True) to raise instead of warn."
                    )
                    if self._policy.strict_device:
                        raise ValueError(msg)
                    logger.error(msg)
                    self._source_sync_device_warned = True
            self._cuda.record_event(self.source_sync_event, CUDAStream_t(producer_stream_handle))
            self._source_sync_recorded = True

    def export(self, frame: GpuFrame) -> FrameOutcome:
        """Export one frame via CUDA IPC ring buffer.

        Non-blocking on the steady path. Returns FrameOutcome rather than raising
        on backpressure.

        Args:
            frame: GpuFrame(ptr, size, producer_stream=None).
                   ptr must be a device-resident pointer; size must equal the
                   FrameSpec geometry (height * width * channels * itemsize).
                   producer_stream, if given, triggers a GPU-side ordering event
                   before the D2D memcpy (replaces record_source_sync()).

        Returns:
            PUBLISHED           — frame delivered to the ring buffer.
            SKIPPED_BARRIER     — activation barrier blocked this frame.
            FAILED              — unrecoverable error; caller should call close().
        """
        if not self._initialized:
            logger.warning("Exporter not initialized")
            return FrameOutcome.FAILED
        if frame.size != self.data_size:
            logger.error("Size mismatch: expected %d, got %d", self.data_size, frame.size)
            return FrameOutcome.FAILED

        # Activation-barrier check
        if self._barrier.evaluate().should_skip:
            if self.shm_handle is not None and self._shutdown_offset:
                with contextlib.suppress(OSError, BufferError):
                    self.shm_handle.buf[self._shutdown_offset] = 0
            return FrameOutcome.SKIPPED_BARRIER

        profile = self._policy.export_profile
        _cuda = self._cuda
        frame_start = 0.0
        _t = 0.0
        _t_sync = 0.0
        _t_sticky = 0.0
        _t_fp = 0.0
        if profile:
            frame_start = time.perf_counter()

        _nvtx.push_range(_NVTX_EXPORTER_SLOT_NAMES[self.write_idx % self._spec.num_slots], "green")
        try:
            slot = self.write_idx % self._spec.num_slots

            # Handle producer_stream from GpuFrame (equivalent to record_source_sync)
            if frame.producer_stream is not None:
                self.record_source_sync(frame.producer_stream)

            # Device-affinity pointer check (first appearance; capped cache)
            gpu_ptr_int = frame.ptr if isinstance(frame.ptr, int) else int(frame.ptr)
            if gpu_ptr_int not in self._ptr_device_cache:
                self._profile.record("ptr_cache_miss", 1.0)
                attrs = _cuda.pointer_get_attributes(gpu_ptr_int)
                if attrs.type not in (2, 3):
                    msg = (
                        f"export: gpu_ptr 0x{gpu_ptr_int:016x} is not device/managed memory "
                        f"(type={attrs.type}). Pass a GPU-resident pointer."
                    )
                    if self._policy.strict_device:
                        raise ValueError(msg)
                    logger.error(msg)
                elif attrs.device != self._spec.device:
                    msg = (
                        f"export: gpu_ptr 0x{gpu_ptr_int:016x} belongs to device "
                        f"{attrs.device}, but exporter is bound to device {self._spec.device}."
                    )
                    if self._policy.strict_device:
                        raise ValueError(msg)
                    logger.error(msg)
                if len(self._ptr_device_cache) < 64:
                    self._ptr_device_cache.add(gpu_ptr_int)

            # --- Graph or legacy copy path ---
            if self._policy.use_graphs and not self._graphs_disabled:
                if profile:
                    _t = time.perf_counter()
                try:
                    _cuda.graph_exec_memcpy_node_set_params_1d(
                        self._graph_execs[slot],
                        self._graph_memcpy_nodes[slot],
                        dst=self.dev_ptrs[slot],
                        src=c_void_p(gpu_ptr_int),
                        count=self.data_size,
                        kind=3,
                    )
                    if self._source_sync_recorded and self.source_sync_event is not None:
                        _cuda.stream_wait_event(self.ipc_stream, self.source_sync_event, 0)
                    _cuda.graph_launch(self._graph_execs[slot], self.ipc_stream)
                    if self.ipc_events[slot]:
                        _cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)
                except (RuntimeError, OSError) as _graph_err:
                    logger.warning("Graph launch failed (%s) — disabling graphs, retrying via legacy path", _graph_err)
                    self._graphs_disabled = True
                    goto_legacy = True
                else:
                    goto_legacy = False
                if profile:
                    self._profile.record("memcpy", (time.perf_counter() - _t) * 1_000_000)
            else:
                goto_legacy = True

            if goto_legacy:
                if profile:
                    _t = time.perf_counter()
                if self.source_sync_event is not None:
                    _cuda.stream_wait_event(self.ipc_stream, self.source_sync_event, 0)
                if profile:
                    self._profile.record("stream_wait", (time.perf_counter() - _t) * 1_000_000)
                    _t = time.perf_counter()
                with _nvtx.verbose_range("cudalink.exporter.memcpy", "green"):
                    _cuda.memcpy_async(
                        dst=self.dev_ptrs[slot],
                        src=c_void_p(gpu_ptr_int),
                        count=self.data_size,
                        kind=3,
                        stream=self.ipc_stream,
                    )
                if profile:
                    self._profile.record("memcpy", (time.perf_counter() - _t) * 1_000_000)
                    _t = time.perf_counter()
                with _nvtx.verbose_range("cudalink.exporter.record_event", "green"):
                    if self.ipc_events[slot]:
                        _cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)
                if profile:
                    self._profile.record("record_event", (time.perf_counter() - _t) * 1_000_000)

            if self._policy.export_sync:
                if profile:
                    _t_sync = time.perf_counter()
                _cuda.stream_synchronize(self.ipc_stream)
                if profile:
                    self._profile.record("sync", (time.perf_counter() - _t_sync) * 1_000_000)

            if profile:
                _t_sticky = time.perf_counter()
            _cuda.check_sticky_error("export")
            if profile:
                self._profile.record("sticky_check", (time.perf_counter() - _t_sticky) * 1_000_000)

            if self._policy.flush_probe and not self._policy.export_sync:
                if profile:
                    _t_fp = time.perf_counter()
                with _nvtx.verbose_range("cudalink.exporter.flush_probe", "green"):
                    _cuda.stream_query(self.ipc_stream)
                if profile:
                    self._profile.record("flush_probe", (time.perf_counter() - _t_fp) * 1_000_000)

            if profile:
                _t = time.perf_counter()
            with _nvtx.verbose_range("cudalink.exporter.shm_write", "green"):
                self.write_idx += 1
                publish_frame(self.shm_handle.buf, self._layout, self.write_idx, time.perf_counter())
            if profile:
                self._profile.record("shm_write", (time.perf_counter() - _t) * 1_000_000)

            self.frame_count += 1

            if profile:
                frame_time = (time.perf_counter() - frame_start) * 1_000_000
                self._profile.record("export", frame_time)
                if self.frame_count % 97 == 0:
                    n = self.frame_count
                    avg_wait = self._profile.avg("stream_wait", n)
                    avg_memcpy = self._profile.avg("memcpy", n)
                    avg_record = self._profile.avg("record_event", n)
                    avg_sync = self._profile.avg("sync", n)
                    avg_sticky = self._profile.avg("sticky_check", n)
                    avg_fp = self._profile.avg("flush_probe", n)
                    avg_shm = self._profile.avg("shm_write", n)
                    avg_total = self._profile.avg("export", n)
                    avg_unacc = avg_total - (
                        avg_wait + avg_memcpy + avg_record + avg_sync + avg_sticky + avg_fp + avg_shm
                    )
                    logger.debug(
                        "Frame %d [PROFILE] memcpy=%.1fus record=%.1fus sync=%.1fus"
                        " sticky=%.1fus flush_probe=%.1fus shm=%.1fus unacc=%.1fus total=%.1fus",
                        n,
                        avg_memcpy,
                        avg_record,
                        avg_sync,
                        avg_sticky,
                        avg_fp,
                        avg_shm,
                        avg_unacc,
                        avg_total,
                    )

            return FrameOutcome.PUBLISHED

        except (OSError, RuntimeError) as e:
            logger.error("Export failed: %s", e)
            traceback.print_exc()
            return FrameOutcome.FAILED
        finally:
            _nvtx.pop_range()

    def close(self) -> None:
        """Shutdown the exporter and release all resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        cuda_valid = self._is_cuda_context_valid()
        self._do_cleanup(cuda_valid=cuda_valid)

    def _is_cuda_context_valid(self) -> bool:
        try:
            return self._cuda.peek_last_error() == 0
        except (OSError, RuntimeError, AttributeError):
            return False

    def _do_cleanup(self, *, cuda_valid: bool) -> None:
        """7-step shutdown. Called from close() and open() failure handler."""
        if not self._initialized and self.shm_handle is None:
            # Nothing was allocated — clean up any barrier SHM only
            self._barrier.close()
            return

        if not cuda_valid:
            logger.warning("CUDA context already destroyed — skipping GPU cleanup")

        # STEP 1: Signal shutdown + zero IPC handles in SHM
        if self.shm_handle:
            try:
                shutdown_offset = SHM_HEADER_SIZE + (self._spec.num_slots * SLOT_SIZE)
                struct.pack_into("<B", self.shm_handle.buf, shutdown_offset, 1)
                logger.info("Shutdown signal sent to consumer")
            except (OSError, BufferError) as e:
                logger.warning("Could not write shutdown signal: %s", e)
            try:
                for slot in range(self._spec.num_slots):
                    base = SHM_HEADER_SIZE + (slot * SLOT_SIZE)
                    self.shm_handle.buf[base : base + SLOT_SIZE] = b"\x00" * SLOT_SIZE
            except (OSError, BufferError) as e:
                logger.warning("Could not zero IPC handles: %s", e)

        # STEP 1c: Destroy graph execs
        if cuda_valid and self._policy.use_graphs:
            self._destroy_export_graphs()

        # STEP 2: Destroy IPC events + sync event
        if cuda_valid and self.ipc_events:
            for _slot, event in enumerate(self.ipc_events):
                if event:
                    with contextlib.suppress(RuntimeError, OSError):
                        self._cuda.destroy_event(event)
        if cuda_valid and self.source_sync_event:
            with contextlib.suppress(RuntimeError, OSError):
                self._cuda.destroy_event(self.source_sync_event)
            self.source_sync_event = None

        # STEP 3: Destroy IPC stream
        if cuda_valid and self.ipc_stream:
            with contextlib.suppress(RuntimeError, OSError):
                self._cuda.destroy_stream(self.ipc_stream)
            self.ipc_stream = None

        # STEP 4: Close SHM
        if self.shm_handle:
            with contextlib.suppress(OSError, BufferError):
                self.shm_handle.close()
            self.shm_handle = None

        # STEP 5: Grace period for consumer to close IPC handles
        if cuda_valid:
            time.sleep(0.1)

        # STEP 6: Free GPU buffers (deadline-bounded parallel cudaFree)
        if cuda_valid and self.dev_ptrs:
            free_threads: list[tuple[threading.Thread, int]] = []
            for slot, dev_ptr in enumerate(self.dev_ptrs):
                if dev_ptr:

                    def _free(ptr: c_void_p, s: int = slot) -> None:
                        with contextlib.suppress(RuntimeError, OSError):
                            self._cuda.free(ptr)
                            logger.debug("Freed GPU buffer slot %d", s)

                    t = threading.Thread(target=_free, args=(dev_ptr,), daemon=True)
                    t.start()
                    free_threads.append((t, slot))

            deadline = time.perf_counter() + 0.5
            for t, slot in free_threads:
                remaining = deadline - time.perf_counter()
                t.join(timeout=max(remaining, 0.0))
                if t.is_alive():
                    logger.warning("cudaFree slot %d timed out — receiver may not have closed the IPC handle.", slot)

        # STEP 7: Unlink SHM (producer owns it)
        try:
            shm_temp = SharedMemory(name=self._spec.shm_name)
            shm_temp.close()
            shm_temp.unlink()
            logger.info("Unlinked SharedMemory")
        except FileNotFoundError:
            pass
        except (OSError, RuntimeError) as e:
            logger.warning("Could not unlink SharedMemory: %s", e)

        self._barrier.close()

        # Reset state
        self.dev_ptrs = [None] * self._spec.num_slots
        self.ipc_events = [None] * self._spec.num_slots
        self.ipc_handles = [None] * self._spec.num_slots
        self.ipc_event_handles = [None] * self._spec.num_slots
        self._initialized = False
        logger.info("Exporter closed")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Exporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_initialized", False) and not getattr(self, "_closed", True):
            self.close()

    # ------------------------------------------------------------------
    # Status / telemetry
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True if open() succeeded and close() has not been called."""
        return self._initialized and not self._closed and all(ptr is not None for ptr in self.dev_ptrs)

    def attach_nvml_observer(self, observer: NVMLObserver) -> None:
        """Attach an NVMLObserver for GPU telemetry in get_stats()."""
        self._nvml_observer = observer

    def get_stats(self) -> dict:
        """Return a dict with current exporter state and frame-averaged metrics."""
        n = self.frame_count
        stats: dict = {
            "initialized": self._initialized,
            "closed": self._closed,
            "shm_name": self._spec.shm_name,
            "resolution": f"{self._spec.width}x{self._spec.height}x{self._spec.channels}",
            "dtype": self._spec.dtype,
            "num_slots": self._spec.num_slots,
            "data_size_kb": self.data_size / 1024,
            "buffer_size_mb": self.buffer_size / (1024 * 1024),
            "frame_count": n,
            "write_idx": self.write_idx,
            "avg_memcpy_us": self._profile.avg("memcpy", n),
            "avg_total_us": self._profile.avg("export", n),
            "dev_ptrs": [f"0x{ptr.value:016x}" if ptr else "NULL" for ptr in self.dev_ptrs],
        }
        observer = getattr(self, "_nvml_observer", None)
        if observer is not None:
            stats["nvml"] = observer.snapshot()
        return stats
