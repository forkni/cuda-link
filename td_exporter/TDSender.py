"""
TDSender — Sender engine: thin TD-COMP adapter over the canonical Exporter.

Owns only the genuinely TD-specific concerns:
  - ExportBuffer TOP resolution and pixel-format rejection (TD 2025 interop quirks)
  - cuda_memory() → GpuFrame bridge (mapping TD's CUDAMemoryRef to GpuFrame)
  - Dynamic geometry / dtype change: close+reopen Exporter on resolution switch
  - HolderBarrier lifecycle (holder role: pauses the Python Exporter during TD init)
  - Host status side-effects (set_warning_status, clear_status)

All GPU ring-buffer allocation, IPC handle export, SHM writes, CUDA graph capture,
event handling, publish, and 7-step cleanup delegate to the canonical Exporter
(td_exporter/Exporter.py — auto-derived from src/cuda_link/exporter.py).

textDAT name: TDSender  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import contextlib
import os
import traceback
from typing import Any, Callable

from ActivationBarrier import HolderBarrier  # noqa: E402, I001
from Exporter import Exporter, ExportPolicy, FrameOutcome, FrameSpec, GpuFrame  # noqa: E402
from NVTXShim import pop_range as _nvtx_pop  # noqa: E402
from NVTXShim import push_range as _nvtx_push  # noqa: E402
from SHMProtocol import DtypeCodec  # noqa: E402
from TDConfig import TDSenderConfig  # noqa: E402
from TDHost import TDHost  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level constants (TD 2025 pixel-format rejection table)
# ---------------------------------------------------------------------------

_CUDA_UNSUPPORTED_PIXEL_FORMATS = {
    # Rejected outright by cudaMemory(): "Texture is invalid pixel format to be shared with CUDA."
    "16-bit float",
    "16float",
    # Bug A: rejected outright with "Source TOP has unsupported pixel format." (no fallback)
    "10-bit",
    "10bit",
    # Bug B: cudaMemory() "succeeds" but returns dataType=uint8/numComps=4 — raw byte layout,
    # NOT the 11:11:10 packed float semantic. Silent receiver corruption without conversion.
    "11-bit",
    "11bit",
}
_EXPORT_BUFFER_NAME = "ExportBuffer"

# Pre-built NVTX range name strings per slot — eliminates f-string allocation on every export_frame call.
_NVTX_SENDER_SLOT_NAMES: tuple[str, ...] = tuple(f"cudalink.sender.export_frame.slot{i}" for i in range(10))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _cm_dtype_to_str(cm_data_type: object) -> str:
    """Map CUDAMemoryRef.data_type (numpy dtype) to a DtypeCodec dtype string.

    Falls back to "uint8" for unknown or missing dtype objects.
    """
    try:
        name = cm_data_type.name  # e.g. "float32", "uint8", "float16"
        if name in DtypeCodec.supported():
            return name
    except (AttributeError, TypeError):
        pass
    return "uint8"


def _guess_dtype_from_buffer_size(width: int, height: int, channels: int, buffer_size: int | None) -> str:
    """Estimate dtype from bytes-per-pixel ratio when data_type is not yet known.

    Used at initialize() time before the first cuda_memory() call reveals the actual dtype.
    """
    if buffer_size is None or width <= 0 or height <= 0 or channels <= 0:
        return "uint8"
    bpp = round(buffer_size / (width * height * channels), 1)
    return {1.0: "uint8", 2.0: "float16", 4.0: "float32"}.get(bpp, "uint8")


# ---------------------------------------------------------------------------
# TDSenderEngine
# ---------------------------------------------------------------------------


class TDSenderEngine:
    """Thin TD-COMP adapter over the canonical Exporter.

    Constructed by CUDAIPCExtension and replaced (not mutated) on mode switches —
    guaranteeing zero state leak between Sender and Receiver modes.
    """

    def __init__(
        self,
        host: TDHost,
        config: TDSenderConfig,
        cuda: Any,  # ignored — Exporter.open(cuda=None) creates its own CTypesCUDAAdapter
        log_fn: Callable,
        num_slots: int,
        device: int,
        shm_name: str,
        verbose: bool,
    ) -> None:
        self._host = host
        self._config = config
        self._log_fn = log_fn
        self.num_slots = num_slots
        self.device = device
        self.shm_name = shm_name
        self.verbose_performance = verbose

        self._initialized: bool = False
        self._exporter: Exporter | None = None
        self._current_spec: FrameSpec | None = None
        self._policy: ExportPolicy | None = None

        # HolderBarrier — signals the Python Exporter (CheckerBarrier) to pause during TD init.
        # This is the HOLDER role; the Exporter's CheckerBarrier is disabled via barrier_enabled=False
        # in ExportPolicy so the canonical Exporter never creates a conflicting CheckerBarrier.
        self._barrier = HolderBarrier(
            enabled=config.activation_barrier,
            settle_frames=config.barrier_settle_frames,
        )

        # TD-specific per-frame state (not in canonical Exporter)
        self._warned_format: bool = False
        self._export_buffer: object = None
        self._last_pixel_fmt: str = ""
        self._last_fmt_needs_conv: bool = False

    # ------------------------------------------------------------------
    # Compatibility properties — delegate to Exporter when initialized,
    # else return empty-but-correctly-sized defaults.
    # Accessed by test_cuda_ipc_exporter.py and test_extension_characterization.py.
    # ------------------------------------------------------------------

    @property
    def dev_ptrs(self) -> list:
        return self._exporter.dev_ptrs if self._exporter is not None else [None] * self.num_slots

    @property
    def ipc_handles(self) -> list:
        return self._exporter.ipc_handles if self._exporter is not None else [None] * self.num_slots

    @property
    def shm_handle(self) -> object:
        return self._exporter.shm_handle if self._exporter is not None else None

    @property
    def write_idx(self) -> int:
        return self._exporter.write_idx if self._exporter is not None else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, force: bool = False) -> None:
        if force or self.verbose_performance:
            self._log_fn(msg)

    # ------------------------------------------------------------------
    # Public API (called by CUDAIPCExtension)
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True when Exporter is open and all ring-buffer slots are allocated."""
        return self._initialized and self._exporter is not None and self._exporter.is_ready()

    def get_stats(self) -> dict:
        """Sender statistics dict.

        Delegates to Exporter.get_stats() which already returns all required keys
        (resolution, buffer_size_mb, write_idx, dev_ptrs, frame_count, etc.).
        Adds 'mode' = 'Sender' for the CUDAIPCExtension facade.
        """
        if self._exporter is None:
            return {
                "mode": "Sender",
                "initialized": False,
                "frame_count": 0,
                "shm_name": self.shm_name,
                "num_slots": self.num_slots,
                "buffer_size_mb": 0,
                "resolution": "N/A",
                "write_idx": 0,
                "dev_ptrs": [],
            }
        stats = self._exporter.get_stats()
        stats["mode"] = "Sender"
        return stats

    def _check_deferred_cleanup(self) -> None:
        """No-op: Exporter.close() handles all cleanup immediately; no deferred queue.

        This method MUST exist — called by CUDAIPCExtension._check_deferred_cleanup()
        and by callbacks_template.py:onFrameStart every cook.
        """

    def _is_unsupported_format(self, top_op: object) -> bool:
        """Return True if TOP pixel format is unsupported by cudaMemory() in TD 2025.

        Empirical probe (verification/results/cuda_memory_probe_20260510_090919.json,
        TD 2025.32820): cudaMemory() rejects all 4 float16 variants and 10:10:10:2 fixed
        outright; 11:11:10 float "succeeds" but returns dataType=uint8/numComps=4 (raw
        byte layout, NOT semantic) — silent corruption. On True: sender skips the frame
        and emits a component warning; on False: warning is cleared.

        top_op may be a RealTOPHandle, FakeTOPHandle, or raw TD TOP (backward compat).
        """
        if hasattr(top_op, "pixel_format"):
            pixel_fmt = str(top_op.pixel_format)
        else:
            pixel_fmt = str(getattr(top_op, "pixelFormat", ""))
        if pixel_fmt == self._last_pixel_fmt:
            return self._last_fmt_needs_conv
        self._last_pixel_fmt = pixel_fmt
        pixel_lower = pixel_fmt.lower()
        self._last_fmt_needs_conv = any(u in pixel_lower for u in _CUDA_UNSUPPORTED_PIXEL_FORMATS)
        return self._last_fmt_needs_conv

    def initialize(self, width: int, height: int, channels: int = 4, buffer_size: int | None = None) -> bool:
        """Open the Exporter with geometry from TD parameters.

        Called by CUDAIPCExtension.initialize() when the COMP first activates or
        after a mode switch. Geometry (width/height/channels) is known at this point;
        dtype is guessed from buffer_size ratio and corrected on the first export_frame()
        call when cuda_memory() reveals the actual data_type.

        Args:
            width: Texture width in pixels.
            height: Texture height in pixels.
            channels: Number of channels (default 4 for RGBA).
            buffer_size: Actual buffer size in bytes; used to infer dtype when set.

        Returns:
            True if Exporter opened successfully, False otherwise.
        """
        if self._initialized:
            self._log("Already initialized")
            return True

        # Lock Numslots while active — changing slot count at runtime causes array-size mismatch.
        self._host.set_param_enabled("Numslots", False)

        try:
            # HolderBarrier: signal the Python Exporter (CheckerBarrier) to pause pushes
            # during this Sender's WDDM-saturating init burst.
            self._barrier.acquire(os.getpid(), log_fn=self._log)

            # Best-guess dtype — corrected on first export_frame() if wrong.
            dtype_guess = _guess_dtype_from_buffer_size(width, height, channels, buffer_size)

            spec = FrameSpec(
                shm_name=self.shm_name,
                height=height,
                width=width,
                channels=channels,
                dtype=dtype_guess,
                num_slots=self.num_slots,
                device=self.device,
            )
            policy = ExportPolicy(
                export_sync=self._config.export_sync,
                flush_probe=self._config.export_flush_probe,
                use_graphs=self._config.use_graphs,
                high_priority_stream=self._config.stream_high_prio,
                export_profile=self._config.export_profile,
                # HolderBarrier is managed by this adapter; disable Exporter's CheckerBarrier
                # so it does not create a conflicting parallel check on the same SHM segment.
                barrier_enabled=False,
            )

            # Exporter.open() with cuda=None creates CTypesCUDAAdapter.for_device(spec.device).
            self._exporter = Exporter.open(spec, policy=policy, cuda=None)
            self._current_spec = spec
            self._policy = policy

            # Cache ExportBuffer handle — eliminates per-frame ownerComp.op() lookup.
            self._export_buffer = self._host.find_top(_EXPORT_BUFFER_NAME)

            self._initialized = True
            self._barrier.arm_settle_countdown()
            self._log("Initialization complete — ready for zero-copy GPU transfer", force=True)
            return True

        except (OSError, RuntimeError, ValueError, Exception) as e:
            self._log(f"Initialization failed: {e}", force=True)
            self._host.set_error_status(f"Initialization failed: {e}")
            traceback.print_exc()
            return False

    def export_frame(self, top_op: object = None) -> bool:
        """Export the ExportBuffer TOP texture via CUDA IPC.

        top_op is deprecated and ignored; ExportBuffer is always resolved from
        ownerComp internally to guarantee the correct frame is exported.

        Returns:
            True if frame was published, False if skipped or failed.
        """
        # Resolve ExportBuffer TOP handle (cached; re-looked-up if invalid).
        top_op = self._export_buffer
        if top_op is None or not top_op.is_valid():
            self._export_buffer = None
            top_op = self._host.find_top(_EXPORT_BUFFER_NAME)
            if top_op is None:
                self._log(f"'{_EXPORT_BUFFER_NAME}' not found in component", force=True)
                return False
            self._export_buffer = top_op

        # is_active() gate — skip silently when Sender parameter is OFF.
        if not self._host.is_active():
            return False

        # Pixel-format rejection (TD 2025 CUDA interop quirks — see _is_unsupported_format).
        if self._is_unsupported_format(top_op):
            src_fmt = top_op.pixel_format if hasattr(top_op, "pixel_format") else getattr(top_op, "pixelFormat", "?")
            self._host.set_warning_status(f"unsupported pixel format {src_fmt!r}")
            if not self._warned_format:
                self._log(
                    f"Pixel format {src_fmt!r} unsupported by cudaMemory() — "
                    "transfer suspended; component tinted yellow",
                    force=True,
                )
                self._warned_format = True
            return False
        if self._warned_format:
            self._host.clear_status()
            self._log("Source pixel format now supported — transfer resumed", force=True)
            self._warned_format = False

        if self._exporter is None or not self._initialized:
            # Lazy init: probe geometry from the first available frame, then skip this
            # cook.  The Exporter opens here; the actual export starts on the next cook.
            try:
                cm_probe = top_op.cuda_memory()  # stream=None → default stream, safe for probe
            except Exception as e:
                self._log(f"Auto-init: cuda_memory() failed: {e}", force=True)
                return False
            self.initialize(cm_probe.width, cm_probe.height, cm_probe.channels, cm_probe.size)
            return False  # skip this frame; next cook exports normally

        _nvtx_push(_NVTX_SENDER_SLOT_NAMES[self._exporter.write_idx % self.num_slots], "green")
        try:
            # Bridge step 1: request TD texture memory on the Exporter's IPC stream so
            # that TD's CUDA work is enqueued on the same stream as the D2D copy — no
            # explicit event ordering needed.
            try:
                cm = top_op.cuda_memory(stream=int(self._exporter.ipc_stream.value))
            except Exception as cuda_err:
                self._log(f"cudaMemory() failed: {cuda_err}", force=True)
                return False

            # Dynamic geometry / dtype correction — close+reopen Exporter if needed.
            detected_dtype = _cm_dtype_to_str(cm.data_type)
            if (cm.height, cm.width, detected_dtype) != (
                self._current_spec.height,
                self._current_spec.width,
                self._current_spec.dtype,
            ):
                self._log(
                    f"Geometry/dtype change: "
                    f"{self._current_spec.width}x{self._current_spec.height} {self._current_spec.dtype}"
                    f" → {cm.width}x{cm.height} {detected_dtype}",
                    force=True,
                )
                with contextlib.suppress(Exception):
                    self._exporter.close()
                new_spec = FrameSpec(
                    shm_name=self.shm_name,
                    height=cm.height,
                    width=cm.width,
                    channels=self._current_spec.channels,
                    dtype=detected_dtype,
                    num_slots=self.num_slots,
                    device=self.device,
                )
                self._exporter = Exporter.open(new_spec, policy=self._policy, cuda=None)
                self._current_spec = new_spec
                # Re-fetch texture memory on the new Exporter's stream.
                try:
                    cm = top_op.cuda_memory(stream=int(self._exporter.ipc_stream.value))
                except Exception as cuda_err:
                    self._log(f"cudaMemory() after re-init failed: {cuda_err}", force=True)
                    return False

            # Bridge step 2: wrap raw GPU pointer into GpuFrame and hand to Exporter.
            frame = GpuFrame(ptr=cm.ptr, size=cm.size)
            outcome = self._exporter.export(frame)

            if outcome is FrameOutcome.PUBLISHED:
                self._barrier.tick_and_maybe_release(os.getpid(), log_fn=self._log)
                return True
            return False

        finally:
            _nvtx_pop()

    def cleanup(self) -> None:
        """Release all CUDA/SHM resources via Exporter.close().

        Idempotent — safe to call multiple times (e.g. Active toggle + __delTD__).
        """
        # Release activation barrier first (mid-settle cleanup path).
        self._barrier.force_release(os.getpid(), log_fn=self._log)
        self._barrier.close()

        if self._exporter is not None:
            with contextlib.suppress(Exception):
                self._exporter.close()
            self._exporter = None

        self._initialized = False
        self._export_buffer = None

        self._host.clear_status()
        self._host.set_param_enabled("Numslots", True)
