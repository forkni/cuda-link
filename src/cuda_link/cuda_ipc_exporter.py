"""
CUDA IPC Exporter — compatibility shim (v1.6.0).

``CUDAIPCExporter`` is deprecated in v1.6.0.  Use ``Exporter.open(FrameSpec(...))``
from ``cuda_link.exporter`` instead.  ``CUDAIPCExporter`` will be removed in v1.7.0.

Migration guide: ``docs/MIGRATION_v1.6.md``

Re-exports for backwards-compatible ``from cuda_link.cuda_ipc_exporter import ...``:
  Exporter, FrameSpec, ExportPolicy, GpuFrame, FrameOutcome
"""

from __future__ import annotations

import contextlib
import logging
import os
import struct
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

# New public API — re-exported for backwards-compat callers
from ._exporter_port import ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from .activation_barrier import bump_skip as _ab_bump
from .activation_barrier import open_or_create as _ab_open
from .activation_barrier import read_state as _ab_read
from .exporter import (
    _DTYPE_ITEMSIZE_MAP,
    _DTYPE_TO_KIND_BITS,
    _EXPORTER_PROFILE_REGIONS,  # noqa: F401 — re-exported for backwards-compat imports
    _NVTX_EXPORTER_SLOT_NAMES,  # noqa: F401 — re-exported for backwards-compat imports
    Exporter,
)
from .shm_protocol import (  # noqa: F401 — re-exported for backwards-compat imports
    METADATA_SIZE,
    NUM_SLOTS_OFFSET,
    PROTOCOL_MAGIC,
    SHM_HEADER_SIZE,
    SHUTDOWN_FLAG_SIZE,
    SLOT_SIZE,
    TIMESTAMP_SIZE,
    WRITE_IDX_OFFSET,
    SHMLayout,
)

logger = logging.getLogger(__name__)


@dataclass
class ProducerActivationBarrier:
    """Producer-side activation-barrier state.

    Kept here for backwards compatibility.  New code should use
    ``ExportPolicy.barrier_enabled`` / ``ExportPolicy.barrier_stale_ns`` and
    pass the policy to ``Exporter.open()``.
    """

    enabled: bool
    stale_ns: int
    shm: SharedMemory | None = None
    _skip_log_last_ns: int = field(init=False, default=0, repr=False)
    _stale_log_last_ns: int = field(init=False, default=0, repr=False)

    @classmethod
    def from_env(cls) -> ProducerActivationBarrier:
        return cls(
            enabled=os.getenv("CUDALINK_ACTIVATION_BARRIER", "1") != "0",
            stale_ns=int(os.getenv("CUDALINK_BARRIER_STALE_NS", str(5 * 1_000_000_000))),
        )

    def should_skip_publish(self) -> bool:
        """Hot path: True => caller skips this frame.

        Lazily opens the SHM segment on first call. Applies a stale-timeout so a
        Sender that crashes mid-init cannot block the producer indefinitely.
        """
        if self.shm is None:
            try:
                self.shm = _ab_open(create=False)
            except FileNotFoundError:
                return False
        try:
            active_count, last_change_ns, _ = _ab_read(self.shm)
        except (OSError, RuntimeError, struct.error):
            return False
        if active_count <= 0:
            return False
        now_ns = time.monotonic_ns()
        if now_ns - last_change_ns > self.stale_ns:
            if now_ns - self._stale_log_last_ns > 1_000_000_000:
                logger.warning(
                    "[ACTIVATION_BARRIER] stale barrier (count=%d, age=%.1fs) — ignoring",
                    active_count,
                    (now_ns - last_change_ns) / 1e9,
                )
                self._stale_log_last_ns = now_ns
            return False
        with contextlib.suppress(OSError, RuntimeError, struct.error):
            _ab_bump(self.shm)
        if now_ns - self._skip_log_last_ns > 1_000_000_000:
            logger.info("[ACTIVATION_BARRIER] skipping publish (active_count=%d)", active_count)
            self._skip_log_last_ns = now_ns
        return True

    def close(self) -> None:
        """Idempotent: close SHM handle if held."""
        if self.shm is not None:
            with contextlib.suppress(OSError, RuntimeError):
                self.shm.close()
            self.shm = None


class CUDAIPCExporter:
    """DEPRECATED in v1.6.0 — use ``Exporter.open(FrameSpec(...))`` instead.

    This class is a backwards-compatibility shim that delegates to the new ``Exporter``
    (``cuda_link.exporter``).  It will be removed in v1.7.0.

    Migration guide: ``docs/MIGRATION_v1.6.md``
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
        device: int = 0,
    ) -> None:
        warnings.warn(
            "CUDAIPCExporter is deprecated in v1.6.0. "
            "Use Exporter.open(FrameSpec(...)) instead — see docs/MIGRATION_v1.6.md",
            DeprecationWarning,
            stacklevel=2,
        )
        # Validate without CUDA (mirrors Exporter.__init__ validation)
        if dtype not in _DTYPE_TO_KIND_BITS:
            raise ValueError(f"Unsupported dtype: {dtype!r}. Must be one of {list(_DTYPE_TO_KIND_BITS)}")
        if not (0 < num_slots <= 10):
            raise ValueError(f"num_slots must be 1-10, got {num_slots}")

        self.shm_name = shm_name
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = dtype
        self.num_slots = num_slots
        self.debug = debug
        self.device = device

        itemsize = _DTYPE_ITEMSIZE_MAP[dtype]
        self.data_size: int = height * width * channels * itemsize

        # Read policy at construction time — mirrors old env-var reading in __init__
        self._policy = ExportPolicy.from_env()
        self._export_sync: bool = self._policy.export_sync
        self._export_flush_probe: bool = self._policy.flush_probe
        self._strict_device: bool = self._policy.strict_device
        self._barrier = ProducerActivationBarrier(
            enabled=self._policy.barrier_enabled,
            stale_ns=self._policy.barrier_stale_ns,
        )

        # Inner Exporter — created lazily on initialize()
        self._inner: Exporter | None = None

    def initialize(self) -> bool:
        """Allocate GPU ring buffer, create IPC handles, write to SharedMemory.

        Idempotent — returns True if already initialized.
        """
        if self._inner is not None:
            return self._inner.is_ready()
        try:
            self._inner = Exporter.open(
                FrameSpec(
                    shm_name=self.shm_name,
                    height=self.height,
                    width=self.width,
                    channels=self.channels,
                    dtype=self.dtype,
                    num_slots=self.num_slots,
                    device=self.device,
                ),
                policy=self._policy,
            )
            return self._inner.is_ready()
        except Exception as exc:
            logger.error("CUDAIPCExporter.initialize() failed: %s", exc)
            return False

    def export_frame(self, gpu_ptr: int, size: int) -> bool:
        """Export one frame. Returns True on success, False on FAILED outcome."""
        if self._inner is None:
            logger.warning("CUDAIPCExporter: export_frame() called before initialize()")
            return False
        outcome = self._inner.export(GpuFrame(ptr=gpu_ptr, size=size))
        return outcome != FrameOutcome.FAILED

    def record_source_sync(self, producer_stream_handle: int) -> None:
        if self._inner is not None:
            self._inner.record_source_sync(producer_stream_handle)

    def cleanup(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None
        # Also close barrier SHM if held by the shim
        self._barrier.close()

    def is_ready(self) -> bool:
        return self._inner is not None and self._inner.is_ready()

    def attach_nvml_observer(self, observer: object) -> None:
        if self._inner is not None:
            self._inner.attach_nvml_observer(observer)  # type: ignore[arg-type]

    def get_stats(self) -> dict:
        if self._inner is not None:
            return self._inner.get_stats()
        return {
            "initialized": False,
            "shm_name": self.shm_name,
            "resolution": f"{self.width}x{self.height}x{self.channels}",
            "dtype": self.dtype,
            "num_slots": self.num_slots,
            "data_size_kb": self.data_size / 1024,
            "buffer_size_mb": 0.0,
            "frame_count": 0,
            "write_idx": 0,
            "avg_memcpy_us": 0.0,
            "avg_total_us": 0.0,
            "dev_ptrs": [],
        }

    def __enter__(self) -> CUDAIPCExporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.cleanup()

    def __del__(self) -> None:
        if getattr(self, "_inner", None) is not None:
            self._inner.close()  # type: ignore[union-attr]
