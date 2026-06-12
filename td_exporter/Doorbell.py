"""Win32 named auto-reset event doorbell for single-consumer frame notification.

The producer calls create_doorbell() once at open() and signal() after each
publish_frame(). The consumer calls open_doorbell() and wait() instead of
poll-sleeping. Because the event is auto-reset, exactly one waiter is woken per
signal — this is intentionally a single-consumer primitive.

Platform:
    Windows only (os.name == "nt"). All public functions are no-ops on other
    platforms: create_doorbell/open_doorbell return None, signal/close are
    silent, wait returns False. Callers are written to tolerate None handles so
    the same code path runs on Linux/macOS without any conditional guards.

Naming:
    doorbell_event_name(shm_name) → r"Local\\cudalink_db_<shm_name>"

    The "Local\\" prefix scopes the event to the current Windows session, which
    is the correct choice for producer/consumer processes running under the same
    user account. Use "Global\\" only if producer and consumer are in different
    Windows sessions (e.g. a system service communicating with a desktop app) —
    that requires SeCreateGlobalPrivilege. The combined name must not exceed
    ~260 characters (MAX_PATH); shared-memory names are normally short, so this
    is not a practical concern.

This module has no relative imports so sync_td_wrapper mirrors it as
"byte_identical" (Doorbell.py in td_exporter/).
"""

import ctypes
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Win32 kernel32 setup — skipped entirely on non-Windows
# ---------------------------------------------------------------------------

if os.name == "nt":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # CreateEventW(lpEventAttributes, bManualReset, bInitialState, lpName) -> HANDLE
    _k32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    _k32.CreateEventW.restype = ctypes.c_void_p

    # OpenEventW(dwDesiredAccess, bInheritHandle, lpName) -> HANDLE
    _k32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    _k32.OpenEventW.restype = ctypes.c_void_p

    _k32.SetEvent.argtypes = [ctypes.c_void_p]
    _k32.SetEvent.restype = ctypes.c_int

    _k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k32.WaitForSingleObject.restype = ctypes.c_uint32

    _k32.CloseHandle.argtypes = [ctypes.c_void_p]
    _k32.CloseHandle.restype = ctypes.c_int
else:
    _k32 = None

# Win32 constants
_EVENT_MODIFY_STATE: int = 0x0002
_SYNCHRONIZE: int = 0x00100000
_WAIT_OBJECT_0: int = 0x0
_WAIT_TIMEOUT: int = 0x00000102
_INFINITE: int = 0xFFFFFFFF

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PREFIX = r"Local\cudalink_db_"


def doorbell_event_name(shm_name: str) -> str:
    """Return the Win32 named-event name for a given SHM segment name.

    Format: ``Local\\cudalink_db_<shm_name>``

    The "Local\\" prefix restricts the event to the current Windows session.
    The combined name must not exceed ~260 characters (Windows MAX_PATH limit).
    """
    return _PREFIX + shm_name


def create_doorbell(name: str) -> object:
    """Create an auto-reset, initially non-signaled Win32 named event.

    Should be called by the producer (Exporter) after the SHM segment name is
    known. Returns an opaque handle on Windows; None on non-Windows (no-op).

    The event is auto-reset (bManualReset=False) so a single SetEvent() wakes
    exactly one waiter — correct for the single-consumer use case.
    """
    if _k32 is None:
        return None
    handle = _k32.CreateEventW(None, False, False, name)
    if handle is None or handle == 0:
        err = ctypes.get_last_error()
        logger.warning("CreateEventW(%r) failed: error %d", name, err)
        return None
    logger.debug("Created doorbell event %r handle=0x%x", name, handle)
    return handle


def open_doorbell(name: str) -> object:
    """Open an existing Win32 named event for signalling + waiting.

    Should be called by the consumer (Importer). Returns None (not an error)
    when the event does not exist yet — the producer may not have called
    create_doorbell() yet. The caller falls back to polling in that case.
    """
    if _k32 is None:
        return None
    handle = _k32.OpenEventW(_EVENT_MODIFY_STATE | _SYNCHRONIZE, False, name)
    if handle is None or handle == 0:
        err = ctypes.get_last_error()
        if err == 2:  # ERROR_FILE_NOT_FOUND — producer not ready yet
            logger.debug("Doorbell event %r not found (producer not ready)", name)
        else:
            logger.warning("OpenEventW(%r) failed: error %d", name, err)
        return None
    logger.debug("Opened doorbell event %r handle=0x%x", name, handle)
    return handle


def signal(handle: object) -> None:
    """Signal (SetEvent) the doorbell — wakes one blocked consumer.

    No-op when handle is None (non-Windows or creation failed).
    Errors are logged but not raised — signal is best-effort; the consumer
    always has a timeout fallback.
    """
    if handle is None or _k32 is None:
        return
    if not _k32.SetEvent(handle):
        err = ctypes.get_last_error()
        logger.warning("SetEvent failed: error %d", err)


def wait(handle: object, timeout_ms: int) -> bool:
    """Block until the doorbell is signaled or timeout_ms elapses.

    Returns True if woken by a signal, False on timeout or when handle is None
    (non-Windows / disabled). The auto-reset event is cleared automatically on
    a True return — no explicit reset needed.
    """
    if handle is None or _k32 is None:
        return False
    result = _k32.WaitForSingleObject(handle, ctypes.c_uint32(timeout_ms))
    return result == _WAIT_OBJECT_0


def close(handle: object) -> None:
    """Close (CloseHandle) the doorbell handle. No-op when handle is None."""
    if handle is None or _k32 is None:
        return
    if not _k32.CloseHandle(handle):
        err = ctypes.get_last_error()
        logger.debug("CloseHandle doorbell failed: error %d", err)
