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
from SHMProtocol import DtypeCodec, FLAGS_MONO_ALPHA, read_version, set_version  # noqa: E402
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
# Map TD pixelFormatName values (op.pixelFormatName) → (dtype_str, num_channels).
# Used in export_frame to detect dtype changes that are invisible to cm.size/cm.data_type —
# specifically dtype-shrink transitions (e.g. float32→uint8) where TD keeps the old
# (larger) CUDA allocation unchanged, so neither cm.size nor cm.data_type reflects the change.
# pixelFormatName updates immediately on any format change (grow or shrink).
# Built from the TD Script TOP par.format menu names (Section 1 of pixel_format_probe.py).
# Float16 variants are included for completeness but are caught first by _is_unsupported_format.
_PIXEL_FMT_NAME_TO_DTYPE: dict[str, tuple[str, int]] = {
    # RGBA
    "rgba32float": ("float32", 4),
    "rgba16fixed": ("uint16", 4),
    "rgba8fixed": ("uint8", 4),
    # RG
    "rg32float": ("float32", 2),
    "rg16fixed": ("uint16", 2),
    "rg8fixed": ("uint8", 2),
    # Mono (1 component)
    "mono32float": ("float32", 1),
    "mono16fixed": ("uint16", 1),
    "mono8fixed": ("uint8", 1),
    # Alpha-only (1 component)
    "a32float": ("float32", 1),
    "a16fixed": ("uint16", 1),
    "a8fixed": ("uint8", 1),
    # Mono+Alpha (2 components)
    "monoalpha32float": ("float32", 2),
    "monoalpha16fixed": ("uint16", 2),
    "monoalpha8fixed": ("uint8", 2),
    # Float16 variants (unsupported by cudaMemory; present so the map is complete)
    "rgba16float": ("float16", 4),
    "rg16float": ("float16", 2),
    "mono16float": ("float16", 1),
    "a16float": ("float16", 1),
    "monoalpha16float": ("float16", 2),
    # "useinput", "rgb10a2fixed", "rgba11float" → NOT mapped (no reliable dtype/channel count)
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

    Used at initialize() time before the first cuda_memory() call reveals the actual dtype,
    and as a size-authoritative fallback when cm.data_type is stale or None.

    Maps to TD-shareable (non-rejected) dtypes only:
      1 byte/component → uint8   ("8-bit fixed")
      2 bytes/component → uint16  ("16-bit fixed") — NOT float16 (TD rejects float16)
      4 bytes/component → float32 ("32-bit float")
    """
    if buffer_size is None or width <= 0 or height <= 0 or channels <= 0:
        return "uint8"
    bpp = round(buffer_size / (width * height * channels), 1)
    return {1.0: "uint8", 2.0: "uint16", 4.0: "float32"}.get(bpp, "uint8")


def _resolve_frame_dtype(
    width: int,
    height: int,
    channels: int,
    cm_size: int | None,
    cm_data_type: object,
    fallback_dtype: str,
) -> str:
    """Return the authoritative dtype for an incoming frame.

    cm.size is the ground truth because it is exactly what Exporter.export validates.
    cm.data_type.name can be None or stale on the transition frame when TD switches
    texture format — so we cross-check name and size and fall back to the size-derived
    dtype when they disagree.

    Resolution order:
      1. If cm.data_type.name is in DtypeCodec.supported() AND its itemsize matches
         the real byte count → name is correct, return it.
      2. Else derive dtype from bytes-per-pixel via _guess_dtype_from_buffer_size and
         return that if it matches the size.
      3. Otherwise return fallback_dtype (typically the current spec's dtype).
    """
    name = _cm_dtype_to_str(cm_data_type)  # "uint8" on None/unknown
    px = width * height * channels
    if px > 0 and cm_size:
        # Path 1: explicit name is consistent with the byte count — trust it.
        if name in DtypeCodec.supported() and DtypeCodec.itemsize(name) * px == cm_size:
            return name
        # Path 2: name is stale/wrong — derive from bytes.
        guessed = _guess_dtype_from_buffer_size(width, height, channels, cm_size)
        if DtypeCodec.itemsize(guessed) * px == cm_size:
            return guessed
    return fallback_dtype


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
        self._closed: bool = False
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

        # Engine-held monotonic version counter — used to seed set_version() after
        # close()+open() so the receiver always sees a strictly-greater version and
        # triggers _refresh_on_version_change. Without this, Exporter.close() unlinks
        # the SHM, the fresh segment starts at version 1 again, and the receiver's
        # `version != last_version` check silently passes (1 != 1 → False).
        self._ipc_version: int = 0

        # TD-specific per-frame state (not in canonical Exporter)
        self._warned_format: bool = False
        self._warned_dtype_size: bool = False  # one-shot: unresolvable bytes-per-pixel
        self._export_buffer: object = None
        self._last_pixel_fmt: str = ""
        self._last_fmt_needs_conv: bool = False
        self._last_pixel_fmt_name: str = ""  # last seen pixelFormatName; change → override dtype

    # ------------------------------------------------------------------
    # Compatibility properties — delegate to Exporter when initialized,
    # else return empty-but-correctly-sized defaults.
    # Accessed by test_cuda_ipc_exporter.py and test_extension_characterization.py.
    # ------------------------------------------------------------------

    @property
    def dev_ptrs(self) -> list:
        if self._exporter is not None:
            return self._exporter.dev_ptrs
        # Pre-init: return correctly-sized null stubs so len() == num_slots.
        # Post-cleanup (_closed=True): return [] — no slots exist.
        return [] if self._closed else [None] * self.num_slots

    @property
    def ipc_handles(self) -> list:
        if self._exporter is not None:
            return self._exporter.ipc_handles
        return [] if self._closed else [None] * self.num_slots

    @property
    def frame_count(self) -> int:
        return self._exporter.frame_count if self._exporter is not None else 0

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

    def _arm_same_stream_ordering(self) -> None:
        """Declare same-stream producer ordering on the current Exporter's IPC stream.

        The TD sender fetches texture memory via ``cudaMemory(stream=ipc_stream.value)``
        (export_frame step 1), so TD enqueues its GL→CUDA interop work on the same stream
        as the subsequent D2D memcpy — FIFO-ordered by construction.  Recording the source
        sync event on the IPC stream itself is the canonical declaration of that invariant;
        the per-frame ``stream_wait_event`` on an already-completed event is a cheap
        CUDA no-op in steady state.

        Call once immediately after every ``Exporter.open(...)`` (both the initial open and
        any format-change reopen) before the first ``export_frame()`` on that instance.
        """
        self._exporter.record_source_sync(int(self._exporter.ipc_stream.value))

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

    def initialize(
        self, width: int, height: int, channels: int = 4, buffer_size: int | None = None, extra_flags: int = 0
    ) -> bool:
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
                extra_flags=extra_flags,
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

            self._arm_same_stream_ordering()

            # Seed the monotonic version counter from the freshly-opened SHM segment.
            # The receiver will cache this value (ipc_version) on first connect; any
            # subsequent reopen must produce a strictly-greater value so the receiver's
            # `version != last_version` check fires. See export_frame() for the bump.
            if self._exporter.shm_handle is not None:
                self._ipc_version = read_version(self._exporter.shm_handle.buf)

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
            # Use pixelFormatName as authoritative dtype/channel source at init time.
            # cm.size may still reflect a stale allocation from the previous session
            # (TD holds the old CUDA memory until ExportBuffer is explicitly re-cooked).
            # pixelFormatName updates immediately when the source format changes, so it
            # gives the correct dtype even before cm.size has caught up.
            _pf_init = str(getattr(top_op, "pixel_format_name", "") or "")
            _pf_init_mapped = (
                _PIXEL_FMT_NAME_TO_DTYPE.get(_pf_init) if _pf_init and _pf_init not in ("useinput",) else None
            )
            if _pf_init_mapped is not None:
                _init_dtype, _init_ch = _pf_init_mapped
                # Compute the buffer_size that should correspond to the declared format.
                # _guess_dtype_from_buffer_size will then derive the same dtype from it,
                # ensuring _current_spec uses the correct dtype from the start.
                _init_size = cm_probe.width * cm_probe.height * _init_ch * DtypeCodec.itemsize(_init_dtype)
                self._last_pixel_fmt_name = _pf_init  # seed cache — first export frame won't re-override
                self._log(
                    f"Auto-init dtype from pixelFormatName={_pf_init!r}: {_init_dtype}/{_init_ch}ch",
                    force=True,
                )
                _init_extra_flags = FLAGS_MONO_ALPHA if _pf_init.startswith("monoalpha") else 0
                self.initialize(cm_probe.width, cm_probe.height, _init_ch, _init_size, extra_flags=_init_extra_flags)
            else:
                self.initialize(cm_probe.width, cm_probe.height, cm_probe.channels, cm_probe.size)
            return False  # skip this frame; next cook exports normally

        _nvtx_push(_NVTX_SENDER_SLOT_NAMES[self._exporter.write_idx % self.num_slots], "green")
        try:
            # Bridge step 1: request TD texture memory on the Exporter's IPC stream so
            # that TD's CUDA work is enqueued on the same stream as the D2D copy — no
            # explicit event ordering needed.
            _pf_name = str(getattr(top_op, "pixel_format_name", "") or "")
            try:
                cm = top_op.cuda_memory(stream=int(self._exporter.ipc_stream.value))
            except Exception as cuda_err:
                self._log(f"cudaMemory() failed: {cuda_err}", force=True)
                return False

            # Dynamic geometry / dtype correction — close+reopen Exporter if needed.
            #
            # We use cm.size as the authoritative dtype signal because cm.data_type can be
            # None or stale on the transition frame when the TD source switches texture
            # format mid-stream.  Without this, a uint8→float32 flip produces 4× the
            # expected byte count but _cm_dtype_to_str still returns "uint8", the guard
            # sees "no change", and Exporter.export spams "Size mismatch" forever.
            #
            # cm.channels is the LIVE channel count from the CUDAMemoryRef — using the
            # cached spec value would cause stale `px` computation when the TD source
            # switches channel count (e.g. 4ch RGBA → 1ch mono), causing _resolve_frame_dtype
            # to compute the wrong bytes-per-pixel and misidentify the dtype.
            cm_channels = cm.channels
            resolved_dtype = _resolve_frame_dtype(
                cm.width, cm.height, cm_channels, cm.size, cm.data_type, self._current_spec.dtype
            )

            # pixelFormatName override — TD's immediate, authoritative format signal.
            # Applied on every frame when the name maps to a known dtype so that
            # dtype-shrink transitions (e.g. float32→uint8) are detected before TD's CUDA
            # allocation has caught up: cm.size lags permanently on a shrink, but
            # pixelFormatName and cm.data_type both update immediately.
            #
            # GpuFrame then uses self._exporter.data_size (dtype-derived) as the copy
            # length, reading only the valid front region of the (still-oversized) GPU
            # allocation — the same approach v1.5.1 uses with its fixed slot size.
            #
            _pf_name_override = False
            if _pf_name and _pf_name not in ("useinput",):
                _pf_mapped = _PIXEL_FMT_NAME_TO_DTYPE.get(_pf_name)
                if _pf_mapped is not None:
                    _pf_dtype, _pf_ch = _pf_mapped
                    if _pf_name != self._last_pixel_fmt_name:
                        # Format name changed — log the transition once.
                        if _pf_dtype != resolved_dtype or _pf_ch != cm_channels:
                            self._log(
                                f"pixelFormatName changed to {_pf_name!r} "
                                f"({_pf_dtype}/{_pf_ch}ch); cm-derived dtype is "
                                f"{resolved_dtype}/{cm_channels}ch (cm.size={cm.size})",
                                force=True,
                            )
                        self._last_pixel_fmt_name = _pf_name
                    # Always apply — gives correct dtype even before cm.size/cm.data_type
                    # catch up to the new format (they lag on dtype-shrink transitions).
                    resolved_dtype = _pf_dtype
                    cm_channels = _pf_ch
                    _pf_name_override = True

            # Defensive: if the byte count doesn't correspond to the resolved dtype AND no
            # authoritative pixelFormatName override is active, try physical-channel
            # inference (H2: RGBA-padded mono source where cm.channels=1 but the GPU
            # allocation is 4-channel RGBA).  With _pf_name_override=True the size mismatch
            # is expected on a dtype-shrink (cm.size still holds the old allocation); H2
            # must not fire there or it would wrongly re-derive the old dtype.
            px = cm.width * cm.height * cm_channels
            _size_matches = px > 0 and bool(cm.size) and DtypeCodec.itemsize(resolved_dtype) * px == cm.size
            if not _size_matches and not _pf_name_override:
                # H2: try to infer physical channel count from the buffer size.
                # TD may report cm.channels=1 (logical) for a mono TOP while allocating
                # RGBA-padded CUDA memory — e.g. mono float32 TOP: cm.channels=1 but
                # cm.size = W×H×4 comps×4 bytes.
                _wh = cm.width * cm.height
                _phys_dtype_map = {1: "uint8", 2: "uint16", 4: "float32"}
                _inferred = False
                for phys_ch in (4, 2):
                    if _wh > 0 and phys_ch != cm_channels:
                        phys_bytes = cm.size // (phys_ch * _wh) if (_wh * phys_ch) > 0 else 0
                        phys_dtype = _phys_dtype_map.get(phys_bytes)
                        if phys_dtype and phys_bytes * phys_ch * _wh == cm.size:
                            if not self._warned_dtype_size:
                                self._log(
                                    f"cm.channels={cm.channels} inconsistent with "
                                    f"cm.size={cm.size} (bpp={cm.size / px:.2f}); "
                                    f"inferring {phys_ch}ch {phys_dtype} from physical "
                                    "CUDA allocation (RGBA-padded mono source).",
                                    force=True,
                                )
                            cm_channels = phys_ch
                            resolved_dtype = phys_dtype
                            px = _wh * phys_ch
                            _inferred = True
                            break
                if not _inferred:
                    if not self._warned_dtype_size:
                        self._log(
                            f"Unsupported bytes-per-pixel "
                            f"({cm.size}/{px}={cm.size / px:.2f}); skipping frame. "
                            "Supported: 1 (uint8), 2 (uint16), 4 (float32).",
                            force=True,
                        )
                        self._warned_dtype_size = True
                    return False
            self._warned_dtype_size = False  # reset once a valid size is seen

            if (
                (cm.height, cm.width) != (self._current_spec.height, self._current_spec.width)
                or cm_channels != self._current_spec.channels
                or resolved_dtype != self._current_spec.dtype
            ):
                self._log(
                    f"Geometry/dtype change: "
                    f"{self._current_spec.width}x{self._current_spec.height}"
                    f"x{self._current_spec.channels} {self._current_spec.dtype}"
                    f" → {cm.width}x{cm.height}x{cm_channels} {resolved_dtype}",
                    force=True,
                )
                with contextlib.suppress(Exception):
                    self._exporter.close()
                new_spec = FrameSpec(
                    shm_name=self.shm_name,
                    height=cm.height,
                    width=cm.width,
                    channels=cm_channels,
                    dtype=resolved_dtype,
                    num_slots=self.num_slots,
                    device=self.device,
                    extra_flags=FLAGS_MONO_ALPHA if _pf_name.startswith("monoalpha") else 0,
                )
                self._exporter = Exporter.open(new_spec, policy=self._policy, cuda=None)
                self._current_spec = new_spec

                self._arm_same_stream_ordering()  # re-arm on the new Exporter's IPC stream

                # Force a monotonically-greater version in the SHM segment so the
                # receiver's `version != last_version` guard always fires.
                # On Windows the receiver's open handle keeps the old segment alive after
                # close()+unlink(), so Exporter.open() re-attaches to the same segment
                # (open-first logic, exporter.py:274-279).  set_version writes into that
                # shared mapping — the receiver will see VERSION_CHANGED on the next tick.
                self._ipc_version += 1
                with contextlib.suppress(Exception):
                    if self._exporter.shm_handle is not None:
                        set_version(self._exporter.shm_handle.buf, self._ipc_version)

                # Emit Status immediately on reopen — don't wait for the first PUBLISHED
                # frame so the format change is visible in the UI right away.
                _reopen_status = (
                    f"{new_spec.width}x{new_spec.height} {_pf_name}"
                    if _pf_name and _pf_name not in ("useinput",) and _pf_name in _PIXEL_FMT_NAME_TO_DTYPE
                    else f"{new_spec.width}x{new_spec.height} {resolved_dtype} {cm_channels}ch"
                )
                self._host.set_info_status(_reopen_status)

                # Re-fetch texture memory on the new Exporter's stream.
                try:
                    cm = top_op.cuda_memory(stream=int(self._exporter.ipc_stream.value))
                except Exception as cuda_err:
                    self._log(f"cudaMemory() after re-init failed: {cuda_err}", force=True)
                    return False

            # Grow-safety guard: after a dtype-grow reopen (e.g. uint8→float32 where the
            # new spec needs more bytes), TD may not have reallocated its texture yet and
            # cm.size is still the smaller value.  Skip this frame and force-cook to nudge
            # TD into reallocating; retry next cook.
            # On a dtype-shrink (e.g. float32→uint8) data_size < cm.size — guard never
            # fires, so the shrink path is always unblocked.
            if self._exporter.data_size > cm.size:
                top_op.cook(force=True)
                return False

            # Bridge step 2: wrap raw GPU pointer into GpuFrame and hand to Exporter.
            # Use self._exporter.data_size (dtype-derived) rather than cm.size so that on
            # a dtype-shrink we copy only the valid front region of the stale allocation.
            # In steady state and on grow (after TD reallocates) data_size == cm.size.
            frame = GpuFrame(ptr=cm.ptr, size=self._exporter.data_size)
            outcome = self._exporter.export(frame)

            if outcome is FrameOutcome.PUBLISHED:
                self._barrier.tick_and_maybe_release(os.getpid(), log_fn=self._log)
                _pub_status = (
                    f"{cm.width}x{cm.height} {_pf_name}"
                    if _pf_name and _pf_name not in ("useinput",) and _pf_name in _PIXEL_FMT_NAME_TO_DTYPE
                    else f"{cm.width}x{cm.height} {resolved_dtype} {cm_channels}ch"
                )
                self._host.set_info_status(_pub_status)
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
        self._closed = True
        self._export_buffer = None

        # Reset per-session state so the next activation starts completely clean.
        # Without this, _last_pixel_fmt_name, _warned_* etc. carry over from the
        # previous session and cause wrong-dtype initialization or stale overrides.
        self._last_pixel_fmt_name = ""
        self._last_pixel_fmt = ""
        self._last_fmt_needs_conv = False
        self._warned_dtype_size = False
        self._warned_format = False

        self._host.clear_status()
        self._host.set_param_enabled("Numslots", True)
