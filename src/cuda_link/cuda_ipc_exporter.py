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
from .exporter import Exporter

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

    This class is a thin compatibility shim that delegates all work to the new
    ``Exporter`` (``cuda_link.exporter``).  It will be removed in v1.7.0.

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
        debug: bool = False,  # noqa: ARG002 — ignored; use ExportPolicy.export_profile
        device: int = 0,
    ) -> None:
        warnings.warn(
            "CUDAIPCExporter is deprecated in v1.6.0. "
            "Use Exporter.open(FrameSpec(...)) instead — see docs/MIGRATION_v1.6.md",
            DeprecationWarning,
            stacklevel=2,
        )
        self._inner = Exporter.open(
            FrameSpec(
                shm_name=shm_name,
                height=height,
                width=width,
                channels=channels,
                dtype=dtype,
                num_slots=num_slots,
                device=device,
            ),
            policy=ExportPolicy.from_env(),
        )
        # Expose original attributes for backwards-compat property reads
        self.shm_name = shm_name
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = dtype
        self.num_slots = num_slots
        self.device = device

    def initialize(self) -> bool:
        """No-op shim — ``Exporter.open()`` already initialises; returns is_ready()."""
        return self._inner.is_ready()

    def export_frame(self, gpu_ptr: int, size: int) -> bool:
        """Export one frame. Returns True on success, False on FAILED outcome."""
        outcome = self._inner.export(GpuFrame(ptr=gpu_ptr, size=size))
        return outcome != FrameOutcome.FAILED

    def record_source_sync(self, producer_stream_handle: int) -> None:
        self._inner.record_source_sync(producer_stream_handle)

    def cleanup(self) -> None:
        self._inner.close()

    def is_ready(self) -> bool:
        return self._inner.is_ready()

    def attach_nvml_observer(self, observer: object) -> None:
        self._inner.attach_nvml_observer(observer)  # type: ignore[arg-type]

    def get_stats(self) -> dict:
        return self._inner.get_stats()

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
        if hasattr(self, "_inner"):
            self._inner.close()
