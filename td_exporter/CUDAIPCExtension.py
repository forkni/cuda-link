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

try:
    from td import COMP, TOP, CUDAMemoryShape
except ImportError:
    from typing import Any as COMP
    from typing import Any as TOP

    CUDAMemoryShape = None

from TDConfig import TDSenderConfig  # noqa: E402
from TDHost import RealTDHost, TDHost  # noqa: E402
from TDReceiver import TDReceiverEngine  # noqa: E402
from TDSender import (  # noqa: E402
    FLAGS_BFLOAT16,
    FORMAT_KIND_FLOAT,
    FORMAT_KIND_SIGNED,
    FORMAT_KIND_UNSIGNED,
    PROTOCOL_MAGIC,
    SHM_HEADER_SIZE,
    SLOT_SIZE,
    TDSenderEngine,
)

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

        # Read construction params once (engine uses them at build time)
        _slots_val = self._host.param_value("Numslots")
        try:
            self._num_slots: int = int(_slots_val) if _slots_val is not None else 3
        except (ValueError, TypeError):
            self._num_slots = 3

        _dev_val = self._host.param_value("Cudadevice")
        try:
            self._device: int = int(_dev_val) if _dev_val is not None else 0
        except (ValueError, TypeError):
            self._device = 0

        _shm_val = self._host.param_value("Ipcmemname")
        self._shm_name: str = str(_shm_val) if _shm_val is not None else "cudalink_output_ipc"

        _debug_val = self._host.param_value("Debug")
        self._verbose: bool = bool(_debug_val) if _debug_val is not None else False
        if self._config.export_profile:
            self._verbose = True

        _hide_val = self._host.param_value("Hidebuiltin")
        if _hide_val is not None:
            self._host.show_custom_only(bool(_hide_val))

        self._engine: TDSenderEngine | TDReceiverEngine = self._make_engine()

        self._log(f"Extension initialized on {ownerComp} [Mode: {self._mode}]", force=True)

        if self._mode == "Receiver":
            self._host.set_param_enabled("Numslots", False)

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _make_engine(self) -> TDSenderEngine | TDReceiverEngine:
        if self._mode == "Sender":
            return TDSenderEngine(
                host=self._host,
                config=self._config,
                cuda=None,
                log_fn=self._log,
                num_slots=self._num_slots,
                device=self._device,
                shm_name=self._shm_name,
                verbose=self._verbose,
            )
        return TDReceiverEngine(
            host=self._host,
            config=self._config,
            cuda=None,
            log_fn=self._log,
            num_slots=self._num_slots,
            device=self._device,
            shm_name=self._shm_name,
            verbose=self._verbose,
        )

    # ------------------------------------------------------------------
    # Logging (façade owns this; engine holds a reference to it)
    # ------------------------------------------------------------------

    def _log(self, msg: str, force: bool = False) -> None:
        prefix = f"[CUDAIPCExtension:{self._mode}]"
        if force or self._verbose:
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
        return self._engine.export_frame(top_op)

    def import_frame(self, import_buffer: TOP) -> bool:
        return self._engine.import_frame(import_buffer)

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
                    self._num_slots = int(_ns)
        self._engine = self._make_engine()
        self._host.set_param_enabled("Numslots", new_mode == "Sender")
        self._log(f"Mode switched to {new_mode}. Will initialize on next frame.", force=True)

    # ------------------------------------------------------------------
    # Attribute bridges — callbacks in parexecute_callbacks.py write
    # these directly; properties propagate to the current engine.
    # ------------------------------------------------------------------

    @property
    def shm_name(self) -> str:
        return self._engine.shm_name

    @shm_name.setter
    def shm_name(self, value: str) -> None:
        self._shm_name = value
        self._engine.shm_name = value

    @property
    def num_slots(self) -> int:
        return self._engine.num_slots

    @num_slots.setter
    def num_slots(self, value: int) -> None:
        self._num_slots = value
        self._engine.num_slots = value

    @property
    def verbose_performance(self) -> bool:
        return self._engine.verbose_performance

    @verbose_performance.setter
    def verbose_performance(self, value: bool) -> None:
        self._verbose = value
        self._engine.verbose_performance = value

    @property
    def _rx_frames_since_last_retry(self) -> int:
        return getattr(self._engine, "_rx_frames_since_last_retry", 0)

    @_rx_frames_since_last_retry.setter
    def _rx_frames_since_last_retry(self, value: int) -> None:
        with contextlib.suppress(AttributeError):
            self._engine._rx_frames_since_last_retry = value

    @property
    def _rx_retry_interval_frames(self) -> int:
        return getattr(self._engine, "_rx_retry_interval_frames", 1)

    @_rx_retry_interval_frames.setter
    def _rx_retry_interval_frames(self, value: int) -> None:
        with contextlib.suppress(AttributeError):
            self._engine._rx_retry_interval_frames = value

    @property
    def _rx_needs_resolution_update(self) -> bool:
        return getattr(self._engine, "_rx_needs_resolution_update", False)

    @_rx_needs_resolution_update.setter
    def _rx_needs_resolution_update(self, value: bool) -> None:
        with contextlib.suppress(AttributeError):
            self._engine._rx_needs_resolution_update = value

    @property
    def _rx_width(self) -> int:
        return getattr(self._engine, "_rx_width", 0)

    @property
    def _rx_height(self) -> int:
        return getattr(self._engine, "_rx_height", 0)

    # --- Engine state bridges (for tests and legacy attribute access) ---

    @property
    def cuda(self) -> object:
        return self._engine.cuda

    @cuda.setter
    def cuda(self, value: object) -> None:
        self._engine.cuda = value

    @property
    def _initialized(self) -> bool:
        return self._engine._initialized

    @property
    def dev_ptrs(self) -> list:
        return getattr(self._engine, "dev_ptrs", [])

    @property
    def ipc_handles(self) -> list:
        return getattr(self._engine, "ipc_handles", [])

    @property
    def shm_handle(self) -> object:
        return self._engine.shm_handle

    @shm_handle.setter
    def shm_handle(self, value: object) -> None:
        self._engine.shm_handle = value

    @property
    def write_idx(self) -> int:
        return getattr(self._engine, "write_idx", 0)

    @property
    def frame_count(self) -> int:
        return self._engine.frame_count
