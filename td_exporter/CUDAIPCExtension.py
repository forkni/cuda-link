# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
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
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

if TYPE_CHECKING:
    # op, run, ui are TD ambient globals injected into the COMP namespace at runtime.
    # Declared here so pyrefly can resolve the bare names used in this file.
    from _td_builtins import CUDAMemoryShape, op, run, ui  # noqa: F401

CUDALinkBootstrap = None  # type: ignore[assignment]  -- fallback if the sibling DAT is absent
# Any failure inside the bootstrap (absent DAT, or a bug it did not catch itself) must
# degrade to mirror mode, never stop this module from compiling -- see the note below.
with contextlib.suppress(Exception):
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
        @classmethod
        def from_env(cls) -> TDReceiverConfig:
            return cls()


@dataclass(frozen=True)
class LoadState:
    """How this COMP's code was loaded, read once from CUDALinkBootstrap and this module.

    ``LoadState.read()`` never raises: the sibling CUDALinkBootstrap DAT may be absent, or an
    older one may lack a field (then it reads as "").
    """

    library_ready: bool  # the cuda_link glue imported (LIBRARY_READY)
    library_error: str  # why it did not (LIBRARY_ERROR), else ""
    active: bool  # CUDALinkBootstrap resolved an installed cuda_link (library mode)
    version: str  # the version CUDALinkBootstrap requires an install to match (MIRROR_VERSION)
    failure_kind: str  # why the bootstrap gave up: "missing", "mismatch", "rival", "broken" or ""
    detail: str  # the bootstrap's full reason (last_error): every layer it tried and why

    @classmethod
    def read(cls) -> LoadState:
        bootstrap = CUDALinkBootstrap
        return cls(
            library_ready=LIBRARY_READY,
            library_error=LIBRARY_ERROR,
            active=bool(getattr(bootstrap, "_active", False)),
            version=str(getattr(bootstrap, "MIRROR_VERSION", "") or ""),
            failure_kind=str(getattr(bootstrap, "failure_kind", "") or ""),
            detail=str(getattr(bootstrap, "last_error", "") or ""),
        )

    @property
    def mode(self) -> str:
        """The load mode the Info page header names."""
        if not self.library_ready:
            return "not loaded"
        return "library mode" if self.active else "classic mode"


def _foreign_alias_error(shm_module: object) -> str:
    """ "" unless ``shm_module`` (the ``SHMProtocol`` bare-name alias) is a real cuda_link
    submodule -- i.e. it has a ``__file__`` under a ``cuda_link`` package directory.

    The bare names (SHMProtocol, TDConfig, ...) are process-wide sys.modules aliases (see
    CUDALinkBootstrap._activate).  Mirror-DAT modules (classic mode) have no ``__file__``,
    so this is always "" for them.  Callers only need this check when THIS COMP's own
    bootstrap did NOT activate -- rival, mismatch, missing, or broken -- because that is
    the only situation where a non-empty result means the alias belongs to a DIFFERENT
    COMP's cuda_link install rather than this COMP's own successful resolution.
    """
    shm_file = getattr(shm_module, "__file__", "") or ""
    if not shm_file:
        return ""
    # Split on both separators: a Windows install path must be recognised even when
    # this code runs under a POSIX interpreter (CI), where "\\" is not a separator.
    if "cuda_link" not in os.path.normcase(shm_file).replace("\\", "/").split("/"):
        return ""
    return (
        f"SHMProtocol resolved to {shm_file}, a cuda_link install aliased by a "
        "different COMP in this TD process; this COMP's own bootstrap did not "
        "activate. Restart TouchDesigner to re-resolve consistently."
    )


# If THIS COMP's own bootstrap did not activate but a DIFFERENT COMP already aliased some
# other cuda_link install in this process, the import above still "succeeds": it silently
# binds to that other install's protocol code instead of failing.  A real engine built on
# it would then run wrong-version SHM framing.  Detect that case and force degraded mode.
if LIBRARY_READY and not LoadState.read().active:
    _rival_error = _foreign_alias_error(sys.modules.get("SHMProtocol"))
    if _rival_error:
        LIBRARY_READY = False
        LIBRARY_ERROR = _rival_error

# TDHost is stdlib-only (no bare-name cuda_link imports) -- always importable, so
# RealTDHost.set_warning_status() is available even in degraded mode.
from TDHost import STATUS_LIBRARY, STATUS_NATIVE_TOPS, RealTDHost, TDHost  # noqa: E402

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


# ---------------------------------------------------------------------------
# Info page: live version label and the C++ TOPs row
# ---------------------------------------------------------------------------

#: Header par on the Info page. Its label is set from the loaded code on every init. TD saves
#: the label in the .toe/.tox too, but nothing reads the saved one: each load writes it again,
#: so it cannot go stale the way the saved Semverversion value did (1.10.2 in a 1.14.1 project).
INFO_HEADER_PAR = "Cudalinkheader2"  # the Info page header; Cudalinkheader is on Settings
INFO_HEADER_TAGLINE = "Zero-copy GPU texture sharing via CUDA IPC"
#: Read-only string par on the Info page: are the C++ TOPs installed, and which version.
NATIVE_TOPS_PAR = "Nativetops"
#: install_cpp_top.py layout: <Documents>/Derivative/Plugins/CudaLink/VERSION.txt
_NATIVE_TOPS_SUBPATH = ("Derivative", "Plugins", "CudaLink", "VERSION.txt")


def _documents_dir() -> Path:
    """The user's real Documents folder (FOLDERID_Documents, follows a OneDrive redirect).

    The same lookup as scripts/install_cpp_top.py, which writes the file this reads. Falls
    back to ~/Documents. Never raises.
    """
    if os.name == "nt":
        with contextlib.suppress(Exception):
            import ctypes  # noqa: PLC0415 -- Windows-only, used once per init
            import uuid  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415

            class _GUID(ctypes.Structure):
                _fields_ = [
                    ("d1", wintypes.DWORD),
                    ("d2", wintypes.WORD),
                    ("d3", wintypes.WORD),
                    ("d4", ctypes.c_ubyte * 8),
                ]

            u = uuid.UUID("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")  # FOLDERID_Documents
            g = _GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
            out = ctypes.c_wchar_p()
            shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
            try:
                hr = shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out))
                if hr == 0 and out.value:
                    return Path(out.value)
            finally:
                # The caller frees the buffer on success and on failure (NULL is a no-op).
                ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
    return Path.home() / "Documents"


def native_tops_version_file() -> Path:
    """Where install_cpp_top.py writes VERSION.txt for the installed C++ TOPs."""
    return _documents_dir().joinpath(*_NATIVE_TOPS_SUBPATH)


def native_tops_status(tox_version: str, version_file: Path | None = None) -> tuple[str, str]:
    """(row text, warning) for the C++ TOPs row. warning is "" when there is nothing to fix.

    Reads the first "cuda-link <version>", "build <sha>" and "cuda toolkit <x.y>" lines of
    VERSION.txt (the install record). The row shows the build sha when the record has one,
    else the CUDA toolkit. Not installed is not a warning: the C++ TOPs are optional. A
    version that differs from this .tox is one. Never raises.
    """
    path = version_file if version_file is not None else native_tops_version_file()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return "not installed", ""
    except OSError as e:
        return f"unreadable: {type(e).__name__}", ""
    fields: dict[str, str] = {}
    for line in lines:
        for key in ("cuda-link ", "build ", "cuda toolkit "):
            if line.startswith(key):
                fields.setdefault(key, line[len(key) :].strip())
    version = fields.get("cuda-link ", "")
    if not version or version == "unknown":
        return "installed, version unknown", ""
    build = fields.get("build ", "")
    cuda = fields.get("cuda toolkit ", "")
    if build and build != "unknown":
        text = f"{version} (build {build})"
    elif cuda and cuda != "unknown":
        text = f"{version} (CUDA {cuda})"
    else:
        text = version
    if tox_version and version != tox_version:
        warning = f"C++ TOPs are {version}, this .tox is {tox_version}: install the same version of both"
        return f"{text} - differs from {tox_version}", warning
    return text, ""


class _Engine(Protocol):
    """What the facade calls on its engine, in either mode.

    TDSenderEngine and TDReceiverEngine both satisfy it: each has the other role's verbs as
    no-ops, so the facade calls the engine without a mode check. _NullEngine satisfies it
    while cuda_link is unresolved.
    """

    verbose_performance: bool

    def initialize(self, width: int, height: int, channels: int = 4, buffer_size: int | None = None) -> bool: ...
    def initialize_receiver(self) -> bool: ...
    def export_frame(self, top_op: object = None) -> bool: ...
    def import_frame(self, handle: object) -> bool: ...
    def has_new_frame(self) -> bool: ...
    def wait_for_new_frame(self) -> bool: ...
    def is_ready(self) -> bool: ...
    def _check_deferred_cleanup(self) -> None: ...
    def update_receiver_resolution(self, handle: object) -> object: ...
    def update_receiver_format(self, handle: object) -> object: ...
    def request_immediate_reconnect(self) -> None: ...
    def consume_pending_resolution(self) -> tuple | None: ...
    def consume_pending_format(self) -> str | None: ...
    def cleanup(self) -> None: ...
    def get_stats(self) -> dict: ...
    def slot_meta_channels(self) -> dict: ...


class _NullEngine:
    """Inert engine used when cuda_link is unresolved.

    Exposes every method the facade delegates to, so CUDAIPCExtension constructs,
    ext.CUDAIPCExtension resolves, and the frame loop runs without raising.
    """

    verbose_performance = False

    def __init__(self, mode: str = "Sender", error: str = "") -> None:
        self._mode = mode
        self._error = error  # LoadState.library_error

    def initialize(self, *a, **k) -> bool:
        return False

    def initialize_receiver(self) -> bool:
        return False

    def export_frame(self, *a, **k) -> bool:
        return False

    def import_frame(self, *a, **k) -> bool:
        return False

    def has_new_frame(self) -> bool:
        # Sender: True, as from TDSenderEngine. Receiver: False, so the receiver loop does nothing.
        return self._mode != "Receiver"

    def wait_for_new_frame(self, *a, **k) -> bool:
        return False

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
        return {"ready": False, "error": self._error}

    def slot_meta_channels(self) -> dict:
        return {}


def par_enable_state(mode: str, active: bool) -> dict[str, bool]:
    """Enable state of the pars that depend on Mode and Active. The one copy of the rule.

    Numslots: the Sender owns the slot count, and it cannot change while the ring is open.
    Streamid is read only by the Sender, Framewaitms only by the Receiver. Every mode that
    is not Sender gets the receiver engine (_make_engine), so the test is on Sender only.
    """
    sender = mode == "Sender"
    return {"Numslots": sender and not active, "Streamid": sender, "Framewaitms": not sender}


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
        # Resolved once here, like the sender config: a mode switch does not read the environment again.
        self._receiver_config: TDReceiverConfig = TDReceiverConfig.from_env()
        # Read once: the bootstrap ran when this module compiled, and nothing changes it later.
        self._load = LoadState.read()

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

        # The Hidebuiltin value last applied; None while the par was never read.
        self._custom_only: bool | None = None
        # The Active value last applied. The frame path reads Active live (is_active); this
        # copy is only for resync_pars, to run the deactivation cleanup of a missed change.
        self._active_seen: bool = self._host.is_active()
        # The value on_par_change last got for each par. resync_pars runs on more than one
        # callback: it must not send again a value that a handler already got and rejected,
        # or one that the Parameter Execute DAT already reported.
        self._dispatched: dict[str, Any] = {}
        # TD absolute frame of the first resync_pars() call; None before it.
        self._resync_frame: int | None = None
        _hide_val = self._host.param_value("Hidebuiltin")
        if _hide_val is not None:
            self._custom_only = bool(_hide_val)
            self._host.show_custom_only(self._custom_only)

        self._engine: _Engine = self._make_engine()

        self._log(f"Extension initialized on {ownerComp} [Mode: {self._mode}]", force=True)

        self.sync_par_enable()

        self._show_build_info()

        # Non-blocking availability notice.  NEVER ui.messageBox here: it is modal and
        # blocks TD's main thread, and each of the four shmem COMPs compiles its OWN
        # module object -- a module-level set cannot dedup across them (that is what
        # produced four stacked dialogs).  Deferred 5 frames so COMP operators finish
        # loading; delayRef=op.TDResources fires even when the timeline is paused.
        # Scheduled on every init, also after Re-Init Extensions: the new RealTDHost starts
        # with a clean tint and Status, so a library that is still missing must be reported
        # again.  _claim_notice_once keeps the Textport lines to one per process.
        if (not self._load.library_ready or not self._load.active) and not self._sibling_mirrors_available():
            with contextlib.suppress(NameError):
                run(  # noqa: F821  -- td global; NameError suppressed (non-TD context)
                    "args[0]()",
                    self._notify_library_unavailable,
                    delayFrames=5,
                    delayRef=op.TDResources,  # noqa: F821
                )

    # ------------------------------------------------------------------
    # Info page
    # ------------------------------------------------------------------

    def _show_build_info(self) -> None:
        """Set the Info header label and the C++ TOPs row from the loaded code. Never raises.

        A .tox without these pars (older builds) is left as it is: the host writes are no-ops.
        """
        try:
            version = self._load.version
            self._host.set_param_label(
                INFO_HEADER_PAR,
                f"CUDA-Link {version or 'unknown'} ({self._load.mode})   |   {INFO_HEADER_TAGLINE}",
            )
            text, warning = native_tops_status(version)
            self._host.set_param_value(NATIVE_TOPS_PAR, text)
            if warning:
                self._log(warning, force=True)
                self._schedule_native_tops_warning(warning)
        except Exception as e:  # noqa: BLE001 -- the Info page must never break the extension
            self._log(f"Info page update failed: {type(e).__name__}: {e}", force=True)

    def _schedule_native_tops_warning(self, warning: str) -> None:
        """Set the C++ TOPs warning 5 frames after init, as the library notice does.

        The warning tint cooks the warning_emitter child, which may not be loaded yet while
        the extension initializes. Outside TD (no ``run`` global), or when scheduling fails,
        it is set at once.
        """
        try:
            run(  # noqa: F821  -- td global
                "args[0](args[1])",
                self._apply_native_tops_warning,
                warning,
                delayFrames=5,
                delayRef=op.TDResources,  # noqa: F821
            )
        except Exception:  # noqa: BLE001 -- NameError outside TD; any failure: set it now
            self._apply_native_tops_warning(warning)

    def _apply_native_tops_warning(self, warning: str) -> None:
        """Set the C++ TOPs condition (STATUS_NATIVE_TOPS): the engine's clear_status() keeps it.

        Never raises: the COMP may be gone after the delay.
        """
        try:
            self._host.set_condition(STATUS_NATIVE_TOPS, "warning", warning)
        except Exception as e:  # noqa: BLE001 -- td.tdError on a deleted COMP is not RuntimeError
            self._log(f"C++ TOPs warning not shown: {type(e).__name__}: {e}", force=True)

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

    def _claim_notice_once(self, reason: tuple[str, str]) -> bool:
        """True only for the first caller reporting this exact reason, once per TD process.

        Module globals cannot dedup: TD compiles a separate module object per Text DAT, so
        each shmem COMP has its own copy of this file's globals.  ``sys`` is the same
        singleton object across the whole process regardless of how many module objects
        import it, so a plain attribute on it is real shared state -- with no need for COMP
        storage, which pickles into the saved .toe and would suppress the notice forever
        for a later TD process (a fresh launch of the same .toe) that never printed it.

        Deduped on ``reason`` (the ``(short, detail)`` message pair) rather than the
        process id: a PID-keyed sentinel suppresses every later call in this process even
        when a *second* sibling COMP has a genuinely different failure reason to report.
        """
        seen = getattr(sys, "_cudalink_notices", None)
        if seen is None:
            seen = set()
            sys._cudalink_notices = seen
        if reason in seen:
            return False
        seen.add(reason)
        return True

    def _notify_library_unavailable(self) -> None:
        """Non-modal 'cuda_link not ready' notice.

        This state is NORMAL on every cold project load until a matching install is
        resolved, so it must never stall the main thread.  The short Status line names
        the specific reason CUDALinkBootstrap gave up (not found, wrong version, a rival
        already loaded, or -- when bootstrap succeeded but this extension's own glue
        import still failed -- an import bug) so it is never just "not found" when
        something else actually went wrong; the full reason -- every layer it tried and
        why -- goes to the Textport only, where it fits.  Per-COMP feedback is the yellow
        tint + Status par + warning_emitter badge; the textport line and the status bar
        fire once per process across all sibling COMPs.
        """
        load = self._load
        ver = load.version
        version_clause = f"cuda_link {ver}" if ver else "cuda_link"
        kind = load.failure_kind
        if load.active and not load.library_ready:
            short = f"{version_clause} loaded but the component failed to import it (see Textport)"
        elif kind == "broken":
            short = f"{version_clause} found but failed to import (see Textport)"
        elif kind == "mismatch":
            short = (
                f"found but not version {ver or '?'} - install a matching cuda_link next to this .toe, then restart TD"
            )
        elif kind == "rival":
            short = "another cuda_link is already loaded - restart TD"
        else:
            short = (
                f"{version_clause} not found - install it next to this .toe or set CUDALINK_LIB_PATH, then restart TD"
            )

        bootstrap_detail = load.detail
        extension_detail = load.library_error
        detail = bootstrap_detail or extension_detail or "cuda_link unavailable"

        with contextlib.suppress(AttributeError, RuntimeError):
            self._host.set_condition(STATUS_LIBRARY, "warning", short)

        if not self._claim_notice_once((short, detail)):
            return
        print(f"[CUDAIPCExtension] {short}")
        print(detail)
        if extension_detail and extension_detail != detail:
            print(extension_detail)
        with contextlib.suppress(NameError, AttributeError, RuntimeError):
            ui.status = short  # noqa: F821  -- non-blocking status bar

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _make_engine(self) -> _Engine:
        if not self._load.library_ready:
            return _NullEngine(self._mode, self._load.library_error)
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
            config=self._receiver_config,  # type: ignore[bad-argument-type]
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
        return self._load.library_ready and not isinstance(self._engine, _NullEngine)

    def initialize(self, width: int, height: int, channels: int = 4, buffer_size: int | None = None) -> bool:
        """Delegate to sender engine's initialize() (kept for test injection)."""
        return self._engine.initialize(width, height, channels, buffer_size)

    def export_frame(self, top_op: TOP | None = None) -> bool:
        return self._engine.export_frame(top_op)

    def _handle(self, import_buffer: TOP | None) -> object:
        return self._host.wrap_top(import_buffer) if import_buffer is not None else None

    def import_frame(self, import_buffer: TOP) -> bool:
        return self._engine.import_frame(self._handle(import_buffer))

    def _check_deferred_cleanup(self) -> None:
        self._engine._check_deferred_cleanup()

    def update_receiver_resolution(self, import_buffer: TOP) -> None:
        self._engine.update_receiver_resolution(self._handle(import_buffer))

    def update_receiver_format(self, import_buffer: TOP) -> None:
        """Update ImportBuffer Script TOP pixel format after a dtype change.

        Call from the same Execute DAT location as update_receiver_resolution so
        par.format is updated before the next copyCUDAMemory cook. Changing par.format
        causes TD to reallocate the output texture at the correct bit depth.
        """
        self._engine.update_receiver_format(self._handle(import_buffer))

    def sync_mono_alpha_fix(self, import_buffer: TOP, fmt: str) -> None:
        """Set the MonoAlphaFix Reorder TOP for the pixel format ``fmt`` of ImportBuffer.

        script_top_callbacks calls this after it sets par.format in onCook. The path through
        update_receiver_format does the same step in TOPHandle.set_format.
        """
        if import_buffer is not None:
            self._host.wrap_top(import_buffer).sync_mono_alpha_fix(fmt)

    def is_active(self) -> bool:
        """Delegate to host's active-parameter check (hot-path safe)."""
        return self._host.is_active()

    def clear_status(self) -> None:
        """Clear the engine's warning or error and reset the Status text (deactivation).

        A C++ TOPs or library condition stays: it holds while the COMP is off too.
        """
        self._host.clear_status()

    def has_new_frame(self) -> bool:
        """Return True when the receiver engine has new data or needs to run.

        Delegates to TDReceiverEngine.has_new_frame() in Receiver mode.
        Always returns True in Sender mode (unused code path there).
        """
        return self._engine.has_new_frame()

    def wait_for_new_frame(self) -> bool:
        """Receiver: wait up to the Framewaitms par for the sender's doorbell.

        Call after has_new_frame() returned False. True means a frame (or a version
        change / shutdown) came within the budget and ImportBuffer must cook.
        """
        return self._engine.wait_for_new_frame()

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

    def slot_meta_channels(self) -> dict:
        """Channel name -> value for the `slot_meta` Script CHOP (empty while inert)."""
        return self._engine.slot_meta_channels()

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
        self.sync_par_enable()
        self._log(f"Mode switched to {new_mode}. Will initialize on next frame.", force=True)

    def sync_par_enable(self) -> None:
        """Apply par_enable_state() for the current Mode and Active.

        Called at load, on a Mode switch, and by parexecute_callbacks when Active changes.
        The engines do not write an enable state.
        """
        for name, enabled in par_enable_state(self._mode, self._host.is_active()).items():
            self._host.set_param_enabled(name, enabled)

    # ------------------------------------------------------------------
    # Parameter changes (Parameter Execute DAT) and the first-frame resync
    # ------------------------------------------------------------------

    def on_par_change(self, name: str, value: object, prev: object = None) -> None:
        """Apply a change of one monitored par: PARS says what each one does.

        parexecute_callbacks.onValueChange calls this with the par's name, its new value and
        its previous value. A par that is not in PARS is logged and ignored. So is a live par
        with no apply: its reader sees the new value on the next frame.
        """
        self._log(f"Parameter '{name}' changed: {prev} -> {value}", force=False)
        self._dispatched[name] = value
        spec = _PARS_BY_NAME.get(name)
        if spec is not None and spec.apply is not None:
            spec.apply(self, value, prev)

    def resync_pars(self) -> bool:
        """Apply the par writes that this extension missed. Return True to be called again.

        The Execute DAT calls this on each frame callback until it returns False. It returns
        True during the frame of the first call, and False on the first call in a later
        frame. TouchDesigner runs Frame Start inside loadTox(), before the script that
        called loadTox() writes its pars (seen in TD 2025.33230, 2026-11-23), so one call
        on the first callback finds nothing to apply. All the writes of that script are
        visible on the next frame. A host that cannot tell the frame gets one call.

        __init__ reads the pars with the "reinit" policy (Mode, Ipcmemname, Numslots,
        Cudadevice, Debug and Hidebuiltin). After that, only the
        Parameter Execute DAT reports a change, and it reports nothing that happened before it
        first cooked. So a script that sets these pars in the same call as loadTox() or
        create() leaves the extension on the old values, with no error (2026-11-21: a Receiver
        set up this way stayed a Sender on the default Ipcmemname). Each par in PARS whose
        value differs from the state the extension keeps goes through on_par_change, the same
        path as a normal change. One par that fails does not stop the others. Last, the enable
        states follow the Mode and Active of this frame.
        """
        frame = self._host.abs_frame()
        if self._resync_frame is None:
            self._resync_frame = frame
        again = frame is not None and frame == self._resync_frame
        for spec in PARS:
            if spec.state is None:
                continue  # a live par with no kept copy: nothing that can be stale
            if spec.name == "Debug" and self._config.export_profile:
                continue  # CUDALINK_EXPORT_PROFILE forces verbose on; Debug Off must not undo it
            try:
                value = self._host.param_value(spec.name)
                if value is None:
                    continue  # a COMP without the par
                if spec.name in self._dispatched and spec.kind(self._dispatched[spec.name]) == spec.kind(value):
                    continue  # a handler already got this value (and may have rejected it)
                seen = spec.state(self)
                if seen is None or spec.kind(value) != spec.kind(seen):
                    self.on_par_change(spec.name, value, seen)
            except Exception:  # noqa: BLE001,S112 -- isolate each par
                continue
        self.sync_par_enable()
        return again

    def _on_active_change(self, value: object, prev: object) -> None:  # noqa: ARG002
        self._active_seen = bool(value)
        if value:
            self._log("Component activated", force=True)
            if self._mode == "Sender":
                self._log("Sender mode ready - will initialize on first frame export", force=False)
            elif self._mode == "Receiver":
                self._log("Receiver mode activated - will attempt connection", force=False)
        else:
            self._log("Component deactivated - cleaning up", force=True)
            self.cleanup()
            self.clear_status()
        self.sync_par_enable()  # Numslots is editable only in an inactive Sender

    def _on_ipcmemname_change(self, value: object, prev: object) -> None:
        new_name = str(value)
        if new_name != (str(prev) if prev is not None else ""):
            self.reconfigure_and_reinit("shm_name", new_name)

    def _on_numslots_change(self, value: object, prev: object) -> None:
        new_slots = int(value)  # type: ignore[call-overload]  # a menu par gives a str
        if new_slots == (int(prev) if prev is not None else 0):  # type: ignore[call-overload]
            return
        # The slot count comes from the sender through SharedMemory. The par is greyed out in
        # Receiver mode; a script can still write it.
        if self._mode == "Receiver":
            self._log("Numslots change ignored in Receiver mode (controlled by sender)", force=True)
            return
        # Greyed out while Active, but a script write does not respect the enable state.
        if self.is_active():
            self._log("Numslots change ignored while Active (deactivate first)", force=True)
            return
        # Hard bounds: below 2 there is no double buffering. Above 10 fails the receiver's
        # protocol guard (TDReceiver num_slots > 10) and the 10-entry NVTX slot-name tuple
        # that TDSender.export_frame indexes by write_idx % num_slots.
        if new_slots < 2 or new_slots > 10:
            self._log(f"WARNING: Numslots={new_slots} outside supported range (2-10); change rejected", force=True)
            return
        # 2-5 slots is the tested range; 6-10 works but is less exercised.
        if new_slots > 5:
            self._log(f"WARNING: Numslots={new_slots} outside recommended range (2-5)", force=True)
        self.reconfigure_and_reinit("num_slots", new_slots)

    def _on_cudadevice_change(self, value: object, prev: object) -> None:  # noqa: ARG002
        try:
            new_device = int(value)  # type: ignore[call-overload]
        except (ValueError, TypeError):  # a Str par on a COMP built by hand: text, or empty
            self._log(f"WARNING: Cudadevice={value!r} is not a device index; change rejected", force=True)
            return
        if new_device == self._device:
            return
        if new_device < 0:
            self._log(f"WARNING: Cudadevice={new_device} is not a device index; change rejected", force=True)
            return
        # The device is fixed when the engine is made, so make a new engine.
        self._log("device changed - reinitializing", force=True)
        self.cleanup()
        self._device = new_device
        self._engine = self._make_engine()
        self.request_immediate_reconnect()  # a no-op in Sender mode

    def _on_debug_change(self, value: object, prev: object) -> None:  # noqa: ARG002
        self.verbose_performance = bool(value)
        self._log("Debug logging ENABLED" if value else "Debug logging DISABLED", force=True)

    def _on_hidebuiltin_change(self, value: object, prev: object) -> None:  # noqa: ARG002
        self._custom_only = bool(value)
        self._host.show_custom_only(self._custom_only)
        self._log(f"Built-in parameters {'hidden' if value else 'visible'}", force=True)

    def _on_mode_change(self, value: object, prev: object) -> None:
        new_mode = str(value)
        old_mode = str(prev) if prev is not None else ""
        if new_mode == old_mode:
            return
        self._log(f"Mode switching: {old_mode} -> {new_mode}", force=True)
        try:
            self.switch_mode(new_mode)
            self._log(f"Mode switch complete: now in {new_mode} mode", force=True)
        except (AttributeError, RuntimeError) as e:
            self._log(f"ERROR switching mode: {e}", force=True)
            return
        # The `bg` Select TOP shows the buffer of the mode. For a Mono+Alpha stream,
        # TOPHandle.sync_mono_alpha_fix moves it to MonoAlphaFix when the stream connects.
        buffer = _MODE_DISPLAY_BUFFER.get(new_mode)
        if buffer is None:
            return
        try:
            if self._host.set_display_buffer(buffer):
                self._log(f"Updated bg selectTOP -> {buffer}", force=False)
        except (AttributeError, RuntimeError) as e:
            self._log(f"Could not update bg selectTOP: {e}", force=False)

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
        self.request_immediate_reconnect()  # a no-op in Sender mode

    def consume_pending_resolution(self) -> tuple | None:
        """Return (width, height) if resolution update is pending, else None.

        Called from script_top_callbacks.onCook to drive ImportBuffer Script TOP par updates.
        """
        return self._engine.consume_pending_resolution()

    def consume_pending_format(self) -> str | None:
        """Return the TD par.format string if a pixel-format update is pending, else None.

        Called from script_top_callbacks.onCook (fallback path) to apply par.format changes
        that were triggered mid-stream by _refresh_on_version_change.  Mirrors
        consume_pending_resolution — same consume-and-clear semantics.
        """
        return self._engine.consume_pending_format()


_MODE_DISPLAY_BUFFER = {"Sender": "ExportBuffer", "Receiver": "ImportBuffer"}


@dataclass(frozen=True)
class ParSpec:
    """One custom par that the extension or its engine reads.

    policy: "reinit" -- __init__ reads the par and the extension keeps the value. A change
            needs apply, and resync_pars compares the par with state.
            "live" -- the reader evaluates the par each time it needs it (Active at each
            frame, Streamid at each publish, Framewaitms at each waiting frame). A live par
            keeps a copy only when a change has an action that resync_pars must replay
            (Active: the cleanup of a deactivation).
    kind:   the type its value and the kept state compare as (resync_pars).
    state:  reads the value the extension keeps for it; None when it keeps nothing.
    apply:  what a change does, as (extension, new value, previous value); None when a
            change needs no action.
    """

    name: str
    policy: Literal["reinit", "live"]
    kind: Callable[[Any], Any]
    state: Callable[[CUDAIPCExtension], Any] | None
    apply: Callable[[CUDAIPCExtension, Any, Any], None] | None

    def __post_init__(self) -> None:
        if self.policy not in ("reinit", "live"):  # Literal is not checked at runtime
            raise ValueError(f"{self.name}: policy must be 'reinit' or 'live', not {self.policy!r}")
        kept = self.state is not None
        if (self.policy == "reinit" and not kept) or (kept and self.apply is None):
            raise ValueError(f"{self.name}: a reinit par needs state and apply; a kept state needs apply")


# Mode comes first: resync_pars walks this order, and the later pars must see the engine of
# the new mode.
PARS: tuple[ParSpec, ...] = (
    ParSpec("Mode", "reinit", str, lambda ext: ext.mode, CUDAIPCExtension._on_mode_change),
    ParSpec("Active", "live", bool, lambda ext: ext._active_seen, CUDAIPCExtension._on_active_change),
    ParSpec("Ipcmemname", "reinit", str, lambda ext: ext.shm_name, CUDAIPCExtension._on_ipcmemname_change),
    ParSpec("Numslots", "reinit", int, lambda ext: ext.num_slots, CUDAIPCExtension._on_numslots_change),
    # Not in the shipped .tox (it uses device 0); a COMP built by hand for a second GPU has it.
    ParSpec("Cudadevice", "reinit", int, lambda ext: ext._device, CUDAIPCExtension._on_cudadevice_change),
    # The clamp and the default of these two are with their readers:
    # TDSenderEngine._stream_id and TDReceiverEngine.frame_wait_ms.
    ParSpec("Streamid", "live", int, None, None),
    ParSpec("Framewaitms", "live", int, None, None),
    ParSpec("Debug", "reinit", bool, lambda ext: ext.verbose_performance, CUDAIPCExtension._on_debug_change),
    ParSpec("Hidebuiltin", "reinit", bool, lambda ext: ext._custom_only, CUDAIPCExtension._on_hidebuiltin_change),
)
_PARS_BY_NAME = {spec.name: spec for spec in PARS}
