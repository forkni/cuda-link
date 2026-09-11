"""
CUDA IPC Extension for TouchDesigner - Dual-Mode Sender/Receiver
Supports both exporting (Sender) and importing (Receiver) GPU textures via CUDA IPC

Usage in TouchDesigner:
    Sender: ext.CUDAIPCExtension.export_frame(top_op)
    Receiver: ext.CUDAIPCExtension.import_frame(import_buffer)

Architecture:
    Sender: TD GPU -> cudaMemory() -> Persistent Buffer -> IPC Handle -> SharedMemory
    Receiver: SharedMemory -> IPC Handle -> Opened GPU Buffer -> scriptTOP.copyCUDAMemory()

Facade: delegates all Sender work to TDSenderEngine and all Receiver work to
TDReceiverEngine.  Mode switches create a fresh engine instance — zero state leak.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # op, run, ui are TD ambient globals injected into the COMP namespace at runtime.
    # Declared here so pyrefly can resolve the bare names used in this file.
    from _td_builtins import CUDAMemoryShape, op, run, ui  # noqa: F401

CUDALinkBootstrap = None  # type: ignore[assignment]  -- fallback if the sibling DAT is absent
with contextlib.suppress(ImportError):
    import CUDALinkBootstrap  # noqa: F401  -- registers sys.modules aliases when present

try:
    from td import COMP, TOP, CUDAMemoryShape
except ImportError:
    from typing import Any as COMP
    from typing import Any as TOP

    CUDAMemoryShape = None

# The extension MUST compile even when cuda_link is unresolved.  If this module raises,
# TD leaves ext.CUDAIPCExtension undefined and every Execute DAT callback throws
# td.tdAttributeError once per frame.  Degrade to an inert extension instead.
LIBRARY_READY: bool = True
LIBRARY_ERROR: str = ""
try:
    from SHMProtocol import (  # noqa: E402
        FLAGS_BFLOAT16,
        FORMAT_KIND_FLOAT,
        FORMAT_KIND_SIGNED,
        FORMAT_KIND_UNSIGNED,
        PROTOCOL_MAGIC,
        SHM_HEADER_SIZE,
        SLOT_SIZE,
    )
    from TDConfig import TDReceiverConfig, TDRuntimeState, TDSenderConfig  # noqa: E402
    from TDReceiver import TDReceiverEngine  # noqa: E402
    from TDSender import TDSenderEngine  # noqa: E402
except Exception as _library_error:  # noqa: BLE001 -- any cuda_link load failure must not break compile
    LIBRARY_READY = False
    LIBRARY_ERROR = f"{type(_library_error).__name__}: {_library_error}"

    # Names stay bound for __all__ / back-compat, but are None so misuse fails loudly
    # instead of silently reading a wrong protocol constant.
    FLAGS_BFLOAT16 = FORMAT_KIND_FLOAT = FORMAT_KIND_SIGNED = FORMAT_KIND_UNSIGNED = None
    PROTOCOL_MAGIC = SHM_HEADER_SIZE = SLOT_SIZE = None
    TDSenderEngine = TDReceiverEngine = None  # type: ignore[assignment]

    from dataclasses import dataclass as _dataclass

    @_dataclass
    class TDRuntimeState:  # type: ignore[no-redef]
        """Field-compatible stand-in so __init__ needs no degraded branch."""

        shm_name: str = ""
        num_slots: int = 3
        verbose: bool = False

        def update(self, field_name: str, value: object) -> None:
            setattr(self, field_name, value)

    class TDSenderConfig:  # type: ignore[no-redef]
        export_profile = False

        @classmethod
        def from_env(cls) -> TDSenderConfig:
            return cls()

    class TDReceiverConfig:  # type: ignore[no-redef]
        pass


# TDHost is stdlib-only (no bare-name cuda_link imports) -- always importable, so
# RealTDHost.set_warning_status() is available even in degraded mode.
from TDHost import RealTDHost, TDHost  # noqa: E402

# Re-export protocol constants for backward compatibility (tests import these from here)
__all__ = [
    "CUDAIPCExtension",
    "FORMAT_KIND_FLOAT",
    "FORMAT_KIND_SIGNED",
    "FORMAT_KIND_UNSIGNED",
    "PROTOCOL_MAGIC",
    "SLOT_SIZE",
    "SHM_HEADER_SIZE",
    "FLAGS_BFLOAT16",
]

# CuPy deferred import flag (tests may patch this)
CUPY_AVAILABLE: bool = False
cp = None

# Session-level dedup guard: track which COMP paths have already shown the install banner.
# Prevents the banner firing twice when an extension is re-compiled in the same TD session.
_banner_shown_for_comps: set[str] = set()
_notice_printed: bool = False  # fallback dedup when COMP storage is unavailable
_NOTICE_STORE_KEY = "cuda_link_notice_shown"


def _bootstrap_active() -> bool:
    """True when CUDALinkBootstrap resolved cuda_link.  Never raises (module may be absent)."""
    return bool(getattr(CUDALinkBootstrap, "_active", False))


def _bootstrap_error() -> str:
    return str(getattr(CUDALinkBootstrap, "last_error", "") or "")


class _NullEngine:
    """Inert engine used when cuda_link is unresolved.

    Exposes every method the facade delegates to, so CUDAIPCExtension constructs,
    ext.CUDAIPCExtension resolves, and the frame loop runs without raising.
    """

    verbose_performance = False

    def initialize(self, *a, **k) -> bool:
        return False

    def initialize_receiver(self) -> bool:
        return False

    def export_frame(self, *a, **k) -> bool:
        return False

    def import_frame(self, *a, **k) -> bool:
        return False

    def has_new_frame(self) -> bool:
        return False  # receiver loop does nothing

    def is_ready(self) -> bool:
        return False

    def _check_deferred_cleanup(self) -> None:
        return None

    def update_receiver_resolution(self, *a, **k) -> None:
        return None

    def update_receiver_format(self, *a, **k) -> None:
        return None

    def request_immediate_reconnect(self) -> None:
        return None

    def consume_pending_resolution(self):
        return None

    def consume_pending_format(self):
        return None

    def cleanup(self) -> None:
        return None

    def get_stats(self) -> dict:
        return {"ready": False, "error": LIBRARY_ERROR}


class CUDAIPCExtension:
    """TouchDesigner extension facade for dual-mode CUDA IPC texture sharing.

    Delegates all Sender work to TDSenderEngine and all Receiver work to
    TDReceiverEngine.  Mode switches tear down the old engine and create a fresh
    one — guaranteeing zero cross-mode state leak.

    Public API is unchanged from v1.x so existing .tox callback templates continue
    to work without modification.
    """

    def __init__(
        self,
        ownerComp: COMP,
        host: TDHost | None = None,
        config: TDSenderConfig | None = None,
    ) -> None:
        self.ownerComp = ownerComp
        self._host: TDHost = host if host is not None else RealTDHost(ownerComp)
        self._config: TDSenderConfig = config if config is not None else TDSenderConfig.from_env()

        _mode_val = self._host.param_value("Mode")
        self._mode: str = str(_mode_val) if _mode_val is not None else "Sender"

        _slots_val = self._host.param_value("Numslots")
        try:
            _num_slots: int = int(_slots_val) if _slots_val is not None else 3
        except (ValueError, TypeError):
            _num_slots = 3

        _dev_val = self._host.param_value("Cudadevice")
        try:
            self._device: int = int(_dev_val) if _dev_val is not None else 0
        except (ValueError, TypeError):
            self._device = 0

        _shm_val = self._host.param_value("Ipcmemname")
        _shm_name: str = str(_shm_val) if _shm_val is not None else "cudalink_ipc_TD>>Python"

        _debug_val = self._host.param_value("Debug")
        _verbose: bool = bool(_debug_val) if _debug_val is not None else False
        if self._config.export_profile:
            _verbose = True

        self._runtime_state = TDRuntimeState(
            shm_name=_shm_name,
            num_slots=_num_slots,
            verbose=_verbose,
        )

        _hide_val = self._host.param_value("Hidebuiltin")
        if _hide_val is not None:
            self._host.show_custom_only(bool(_hide_val))

        self._engine: TDSenderEngine | TDReceiverEngine | _NullEngine = self._make_engine()

        self._log(f"Extension initialized on {ownerComp} [Mode: {self._mode}]", force=True)

        if self._mode == "Receiver":
            self._host.set_param_enabled("Numslots", False)

        # Non-blocking availability notice.  NEVER ui.messageBox here: it is modal and
        # blocks TD's main thread, and each of the four shmem COMPs compiles its OWN
        # module object -- a module-level set cannot dedup across them (that is what
        # produced four stacked dialogs).  Deferred 5 frames so COMP operators finish
        # loading; delayRef=op.TDResources fires even when the timeline is paused.
        _comp_path = getattr(ownerComp, "path", str(ownerComp))
        if (
            _comp_path not in _banner_shown_for_comps
            and (not LIBRARY_READY or not _bootstrap_active())
            and not self._sibling_mirrors_available()
        ):
            _banner_shown_for_comps.add(_comp_path)
            with contextlib.suppress(NameError):
                run(  # noqa: F821  -- td global; NameError suppressed (non-TD context)
                    "args[0]()",
                    self._notify_library_unavailable,
                    delayFrames=5,
                    delayRef=op.TDResources,  # noqa: F821
                )

    # ------------------------------------------------------------------
    # Library availability detection + install banner
    # ------------------------------------------------------------------

    def _sibling_mirrors_available(self) -> bool:
        """Return True if the classic-mode mirror Text DATs are present as COMP siblings.

        Checks three key mirror DATs that are directly imported by bare name in the glue
        files (SHMProtocol in TDReceiver+TDSender, Exporter in TDSender, Env in TDConfig).
        If any is present, assumes the full 14-mirror classic deployment is in place and no
        install banner is needed.

        Uses ownerComp.op() which returns None when the DAT does not exist.
        """
        with contextlib.suppress(AttributeError, RuntimeError):
            for name in ("SHMProtocol", "Exporter", "Env"):
                if self.ownerComp.op(name) is not None:
                    return True
        return False

    def _claim_notice_once(self) -> bool:
        """True only for the first caller across all four sibling COMPs.

        Module globals cannot dedup: TD compiles a separate module object per Text DAT,
        so each shmem COMP has its own copy of this file's globals.  COMP storage is
        real shared state in the TD process, so the flag lives on the owning tox
        (parent of the shmem COMP), falling back to op.TDResources.
        """
        global _notice_printed
        holder = None
        with contextlib.suppress(AttributeError, RuntimeError, NameError):
            holder = self.ownerComp.parent()
        if holder is None:
            with contextlib.suppress(AttributeError, RuntimeError, NameError):
                holder = op.TDResources  # noqa: F821
        if holder is None:  # non-TD context (tests)
            if _notice_printed:
                return False
            _notice_printed = True
            return True
        try:
            if holder.fetch(_NOTICE_STORE_KEY, False, storeDefault=False):
                return False
            holder.store(_NOTICE_STORE_KEY, True)
        except (AttributeError, RuntimeError, TypeError):
            return True
        return True

    def _notify_library_unavailable(self) -> None:
        """Non-modal 'cuda_link not ready' notice.

        Under the Base Folder contract this state is NORMAL on every cold project load
        until Startstream, so it must never stall the main thread.  Per-COMP feedback is
        the yellow tint + Status par + warning_emitter badge; the textport line and the
        status bar fire once per tox.
        """
        detail = LIBRARY_ERROR or _bootstrap_error() or "Base Folder not set"
        short = "cuda_link not ready - set the StreamDiffusionTD Base Folder, then start the stream."

        with contextlib.suppress(AttributeError, RuntimeError):
            self._host.set_warning_status(short)

        if not self._claim_notice_once():
            return
        print(f"[CUDAIPCExtension] {short} ({detail})")
        with contextlib.suppress(NameError, AttributeError, RuntimeError):
            ui.status = short  # noqa: F821  -- non-blocking status bar

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _make_engine(self) -> TDSenderEngine | TDReceiverEngine | _NullEngine:
        if not LIBRARY_READY:
            return _NullEngine()
        # Reached only when the cuda_link import above succeeded, so TDSenderEngine /
        # TDReceiverEngine / *Config are the real classes here, never the None / stand-in
        # fallbacks bound in the except branch -- pyrefly can't see that control-flow
        # invariant across the module-level try/except, hence the targeted ignores below.
        rs = self._runtime_state
        if self._mode == "Sender":
            return TDSenderEngine(  # type: ignore[not-callable]
                host=self._host,
                config=self._config,  # type: ignore[bad-argument-type]
                cuda=None,
                log_fn=self._log,
                num_slots=rs.num_slots,
                device=self._device,
                shm_name=rs.shm_name,
                verbose=rs.verbose,
            )
        return TDReceiverEngine(  # type: ignore[not-callable]
            host=self._host,
            config=TDReceiverConfig(),  # type: ignore[bad-argument-type]
            cuda=None,
            log_fn=self._log,
            num_slots=rs.num_slots,
            device=self._device,
            shm_name=rs.shm_name,
            verbose=rs.verbose,
        )

    # ------------------------------------------------------------------
    # Logging (façade owns this; engine holds a reference to it)
    # ------------------------------------------------------------------

    def _log(self, msg: str, force: bool = False) -> None:
        prefix = f"[CUDAIPCExtension:{self._mode}]"
        if force or self._runtime_state.verbose:
            print(f"{prefix} {msg}")

    # ------------------------------------------------------------------
    # Public API — all delegate to engine
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def library_ready(self) -> bool:
        """False while this instance is running inert (cuda_link was unresolved at compile)."""
        return LIBRARY_READY and not isinstance(self._engine, _NullEngine)

    def initialize(self, width: int, height: int, channels: int = 4, buffer_size: int | None = None) -> bool:
        """Delegate to sender engine's initialize() (kept for test injection)."""
        return self._engine.initialize(width, height, channels, buffer_size)

    def export_frame(self, top_op: TOP | None = None) -> bool:
        if self._mode != "Sender":
            return False
        return self._engine.export_frame(top_op)

    def import_frame(self, import_buffer: TOP) -> bool:
        if self._mode != "Receiver":
            return False
        handle = self._host.wrap_top(import_buffer) if import_buffer is not None else None
        return self._engine.import_frame(handle)

    def _check_deferred_cleanup(self) -> None:
        if self._mode == "Sender":
            self._engine._check_deferred_cleanup()

    def update_receiver_resolution(self, import_buffer: TOP) -> None:
        if self._mode == "Receiver":
            handle = self._host.wrap_top(import_buffer) if import_buffer is not None else None
            self._engine.update_receiver_resolution(handle)

    def update_receiver_format(self, import_buffer: TOP) -> None:
        """Update ImportBuffer Script TOP pixel format after a dtype change.

        Call from the same Execute DAT location as update_receiver_resolution so
        par.format is updated before the next copyCUDAMemory cook. Changing par.format
        causes TD to reallocate the output texture at the correct bit depth.
        """
        if self._mode == "Receiver":
            handle = self._host.wrap_top(import_buffer) if import_buffer is not None else None
            self._engine.update_receiver_format(handle)

    def is_active(self) -> bool:
        """Delegate to host's active-parameter check (hot-path safe)."""
        return self._host.is_active()

    def has_new_frame(self) -> bool:
        """Return True when the receiver engine has new data or needs to run.

        Delegates to TDReceiverEngine.has_new_frame() in Receiver mode.
        Always returns True in Sender mode (unused code path there).
        """
        if self._mode != "Receiver":
            return True
        return self._engine.has_new_frame()

    def initialize_receiver(self) -> bool:
        """Delegate to receiver engine's initialize_receiver() (backward compat)."""
        return self._engine.initialize_receiver()

    def cleanup(self) -> None:
        self._engine.cleanup()

    def __delTD__(self) -> None:
        self.cleanup()

    def is_ready(self) -> bool:
        return self._engine.is_ready()

    def get_stats(self) -> dict:
        return self._engine.get_stats()

    def switch_mode(self, new_mode: str) -> None:
        if new_mode == self._mode:
            return
        self._log(f"Switching mode: {self._mode} -> {new_mode}", force=True)
        # Tear down old engine via full cleanup() so the receiver registry is updated.
        self.cleanup()
        self._mode = new_mode
        # When switching to Sender: re-read num_slots from UI (receiver may have updated it)
        if new_mode == "Sender":
            _ns = self._host.param_value("Numslots")
            if _ns is not None:
                with contextlib.suppress(ValueError, TypeError):
                    self._runtime_state.num_slots = int(_ns)
        self._engine = self._make_engine()
        self._host.set_param_enabled("Numslots", new_mode == "Sender")
        self._log(f"Mode switched to {new_mode}. Will initialize on next frame.", force=True)

    # ------------------------------------------------------------------
    # Runtime config accessors
    # ------------------------------------------------------------------

    @property
    def shm_name(self) -> str:
        return self._runtime_state.shm_name

    @property
    def num_slots(self) -> int:
        return self._runtime_state.num_slots

    @property
    def verbose_performance(self) -> bool:
        return self._runtime_state.verbose

    @verbose_performance.setter
    def verbose_performance(self, value: bool) -> None:
        self._runtime_state.update("verbose", value)
        self._engine.verbose_performance = value

    def request_immediate_reconnect(self) -> None:
        """Force next import_frame to attempt reconnection (called from parexecute callbacks)."""
        if self._mode == "Receiver":
            self._engine.request_immediate_reconnect()

    def reconfigure_and_reinit(self, field_name: str, new_value: object) -> None:
        """Update a runtime config field and immediately recreate the engine.

        Caller is responsible for pre-validation (range checks, mode guards).
        The new engine initialises lazily on the next export_frame / import_frame call.
        """
        self._log(f"{field_name} changed - reinitializing", force=True)
        self.cleanup()
        self._runtime_state.update(field_name, new_value)
        self._engine = self._make_engine()
        if self._mode == "Receiver":
            self.request_immediate_reconnect()

    def consume_pending_resolution(self) -> tuple | None:
        """Return (width, height) if resolution update is pending, else None.

        Called from script_top_callbacks.onCook to drive ImportBuffer Script TOP par updates.
        """
        if self._mode == "Receiver":
            return self._engine.consume_pending_resolution()
        return None

    def consume_pending_format(self) -> str | None:
        """Return the TD par.format string if a pixel-format update is pending, else None.

        Called from script_top_callbacks.onCook (fallback path) to apply par.format changes
        that were triggered mid-stream by _refresh_on_version_change.  Mirrors
        consume_pending_resolution — same consume-and-clear semantics.
        """
        if self._mode == "Receiver":
            return self._engine.consume_pending_format()
        return None
