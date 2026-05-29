"""
Windows console-control-handler installation helper.

Provides ``install_console_ctrl_handler`` — a single entry point that
registers a typed ``SetConsoleCtrlHandler`` callback and returns a
``ConsoleShutdownState`` the caller reads to detect shutdown signals.

Usage (subprocess entry-point scripts)::

    from cuda_link._console import install_console_ctrl_handler, run_with_watchdog

    def _do_cleanup() -> None:
        ...  # idempotent resource teardown

    _shutdown = install_console_ctrl_handler("[myapp]", _do_cleanup, defer_close=True)

    try:
        while not _shutdown.stop_requested:
            ...
    finally:
        _do_cleanup()
        if _shutdown.shutdown_via not in ("ctrl_break", "ctrl_close"):
            input("Press Enter to close ...")

On non-win32 platforms ``install_console_ctrl_handler`` is a no-op that returns
an inert state object, so callers need no platform guards.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable

# Windows console-control event codes (same values on all platforms; safe as module constants).
_CTRL_C_EVENT = 0
_CTRL_BREAK_EVENT = 1
_CTRL_CLOSE_EVENT = 2
_CTRL_LOGOFF_EVENT = 5
_CTRL_SHUTDOWN_EVENT = 6


@dataclass
class ConsoleShutdownState:
    """Mutable state updated by the installed console-control handler.

    Callers read these fields from the main loop; they are set only by
    the installed handler callback (or left at their defaults on non-win32
    platforms).

    Attributes:
        shutdown_via:   ``"ctrl_c"`` / ``"ctrl_break"`` / ``"ctrl_close"``, or
                        ``None`` if no shutdown has been signalled yet.
        stop_requested: ``True`` when ``defer_close=True`` and a
                        CLOSE/LOGOFF/SHUTDOWN event fires — the main loop should
                        break and run cleanup from the main thread.
        _handler_ref:   Internal. Keeps the WINFUNCTYPE routine object alive so
                        Windows never calls freed memory.  Do not use directly.
    """

    shutdown_via: str | None = None
    stop_requested: bool = False
    _handler_ref: object = field(default=None, repr=False)


def install_console_ctrl_handler(
    prefix: str,
    on_cleanup: Callable[[], None],
    *,
    defer_close: bool = False,
) -> ConsoleShutdownState:
    """Register a Windows console-control handler and return its state object.

    On non-win32 platforms this is a no-op: an inert ``ConsoleShutdownState``
    is returned (all fields at defaults), so callers need no platform guards.

    Args:
        prefix:      Log prefix printed in console messages, e.g. ``"[sender]"``.
        on_cleanup:  Idempotent cleanup callable.  Called directly from the handler
                     when ``defer_close=False`` and a CLOSE/LOGOFF/SHUTDOWN event
                     fires.  When ``defer_close=True`` the handler only sets
                     ``state.stop_requested = True``; the *caller* is responsible
                     for running ``on_cleanup`` from the main thread once it sees
                     ``stop_requested``.
        defer_close: When ``True``, CTRL_CLOSE_EVENT / CTRL_LOGOFF_EVENT /
                     CTRL_SHUTDOWN_EVENT set ``state.stop_requested`` instead of
                     calling ``on_cleanup`` inline.  Use this when the cleanup body
                     contains CUDA calls that race with an in-flight GPU operation
                     on the main thread (e.g. an active ``cudaMemcpy`` when the
                     console X-button fires).

    Returns:
        A ``ConsoleShutdownState`` instance that is updated by the handler.
    """
    state = ConsoleShutdownState()

    if sys.platform != "win32":
        return state

    from ctypes import wintypes as _wintypes

    _handler_cls = ctypes.WINFUNCTYPE(_wintypes.BOOL, _wintypes.DWORD)

    _k32 = ctypes.windll.kernel32
    # arg 0 is PHANDLER_ROUTINE — a Win32 function pointer.  We use c_void_p (not a
    # strict WINFUNCTYPE argtype) so that the documented "restore-default Ctrl+C" call
    # (passing NULL) works: ctypes refuses None for a strict WINFUNCTYPE argtype, but
    # accepts it for c_void_p (same pointer ABI).
    _k32.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, _wintypes.BOOL]
    _k32.SetConsoleCtrlHandler.restype = _wintypes.BOOL

    def _ctrl_handler(ctrl_type: int) -> bool:
        if ctrl_type == _CTRL_C_EVENT:
            state.shutdown_via = "ctrl_c"
            print(f"\n{prefix} Ctrl+C — stopping ...", flush=True)
            return False  # chain to Python default → KeyboardInterrupt in main
        if ctrl_type == _CTRL_BREAK_EVENT:
            state.shutdown_via = "ctrl_break"
            print(f"\n{prefix} Ctrl+Break / launcher shutdown — stopping ...", flush=True)
            return False  # chain to Python default → KeyboardInterrupt in main
        if ctrl_type in (_CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT):
            state.shutdown_via = "ctrl_close"
            if defer_close:
                # Signal the main loop to break; cleanup runs from the main thread,
                # avoiding the race between the handler thread and an in-flight GPU op.
                state.stop_requested = True
                print(
                    f"\n{prefix} Console control event {ctrl_type} "
                    f"(close/logoff/shutdown) — signaling main loop to stop ...",
                    flush=True,
                )
            else:
                print(
                    f"\n{prefix} Console control event {ctrl_type} (close/logoff/shutdown) — running cleanup ...",
                    flush=True,
                )
                on_cleanup()
                print(f"{prefix} Cleanup complete.", flush=True)
            return True  # handled — OS grace period covers main's exit + cleanup
        return False

    # CREATE_NEW_PROCESS_GROUP disables Ctrl+C delivery by default; re-enable it
    # before installing our own handler.
    _k32.SetConsoleCtrlHandler(None, False)

    # MUST keep the routine object alive for the process lifetime: store it on the
    # returned state so the caller's reference to `state` is sufficient.
    handler_ref = _handler_cls(_ctrl_handler)
    state._handler_ref = handler_ref

    if not _k32.SetConsoleCtrlHandler(handler_ref, True):
        print(f"{prefix} WARNING: SetConsoleCtrlHandler failed — console-close cleanup unavailable")

    return state


def run_with_watchdog(
    fn: Callable[[], None],
    timeout_s: float,
    label: str,
    prefix: str,
) -> None:
    """Run *fn* in a daemon thread, printing a warning if it exceeds *timeout_s*.

    Returns as soon as *fn* completes, or after *timeout_s* seconds (leaving the
    thread running as a daemon — the OS reclaims it on process exit).

    This pattern is used for CUDA resource teardown when ncu kernel-replay has
    paused the GPU command queue: ``cudaFree`` / graph-exec teardown / stream
    destruction can block indefinitely in that state.  The watchdog bounds total
    cleanup time so ``main()`` can return and ncu finalizes.

    Args:
        fn:        Callable to run in the daemon thread.  Must swallow its own
                   exceptions (print and return) rather than re-raising.
        timeout_s: Seconds to wait before emitting the timeout message.
        label:     Short description for the timeout line (e.g. ``"exporter.close()"``)
        prefix:    Log prefix (e.g. ``"[sender]"``).
    """
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        print(f"{prefix} {label} timed out — OS will reclaim resources on process exit", flush=True)
