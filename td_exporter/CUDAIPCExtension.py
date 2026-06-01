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

with contextlib.suppress(ImportError):
    import CUDALinkBootstrap  # noqa: F401  -- registers sys.modules aliases when present

try:
    from td import COMP, TOP, CUDAMemoryShape
except ImportError:
    from typing import Any as COMP
    from typing import Any as TOP

    CUDAMemoryShape = None

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
from TDHost import RealTDHost, TDHost  # noqa: E402
from TDReceiver import TDReceiverEngine  # noqa: E402
from TDSender import TDSenderEngine  # noqa: E402

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

        self._engine: TDSenderEngine | TDReceiverEngine = self._make_engine()

        self._log(f"Extension initialized on {ownerComp} [Mode: {self._mode}]", force=True)

        if self._mode == "Receiver":
            self._host.set_param_enabled("Numslots", False)

        # Schedule the install banner if neither library mode nor classic mirror DATs are
        # available.  Deferred 5 frames so all COMP operators finish loading first.
        # delayRef=op.TDResources fires even when the TD timeline is paused.
        _comp_path = getattr(ownerComp, "path", str(ownerComp))
        if (
            _comp_path not in _banner_shown_for_comps
            and not CUDALinkBootstrap._active
            and not self._sibling_mirrors_available()
        ):
            _banner_shown_for_comps.add(_comp_path)
            with contextlib.suppress(NameError):
                run(  # noqa: F821  -- td global; NameError suppressed (non-TD context)
                    "args[0]()",
                    self._show_install_banner,
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

    def _show_install_banner(self) -> None:
        """Show a native TD modal dialog when cuda_link is missing.

        Called via run() with delayFrames=5 so the dialog fires after the project
        finishes loading.  Guards for the non-TD context (tests / editable installs)
        where 'ui' is not available.

        Button mapping:
          0 → Open Preferences  (opens Edit → Preferences so user can set Python Module Path)
          1 → Copy Install Command  (puts 'install_td_library.cmd' on the clipboard)
          2 → Dismiss
        Always sets COMP warning status so the yellow tint persists after the dialog closes.
        """
        try:
            from td import ui  # noqa: PLC0415  -- deferred; not available outside TD
        except (ImportError, NameError):
            return

        msg = (
            "The cuda_link package is not installed and no mirror Text DATs were found.\n\n"
            "To fix, choose one of:\n"
            "  • Run install_td_library.cmd and pick an install target\n"
            "  • Set CUDALINK_LIB_PATH=<install folder> before launching TD\n"
            "  • Edit → Preferences → Python 32/64 bit Module Path → add the folder\n\n"
            "The component will not export or receive frames until cuda_link is available."
        )
        result = ui.messageBox(
            "cuda-link: Library Not Found",
            msg,
            buttons=["Open Preferences", "Copy Install Command", "Dismiss"],
        )
        if result == 0:  # Open Preferences
            ui.openPreferences()
        elif result == 1:  # Copy install_td_library.cmd path to clipboard
            ui.clipboard = "install_td_library.cmd"
            ui.status = "install_td_library.cmd name copied — run it from the repo root in a terminal."

        # Always tint the COMP yellow so the missing-library state is visible in the network.
        self._host.set_warning_status("cuda_link not available — run install_td_library.cmd or set CUDALINK_LIB_PATH")

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _make_engine(self) -> TDSenderEngine | TDReceiverEngine:
        rs = self._runtime_state
        if self._mode == "Sender":
            return TDSenderEngine(
                host=self._host,
                config=self._config,
                cuda=None,
                log_fn=self._log,
                num_slots=rs.num_slots,
                device=self._device,
                shm_name=rs.shm_name,
                verbose=rs.verbose,
            )
        return TDReceiverEngine(
            host=self._host,
            config=TDReceiverConfig(),
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
        # Tear down old engine (guaranteed no state leak — new engine is a fresh instance)
        self._engine.cleanup()
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
