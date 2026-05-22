"""
CUDA-Link Example — Python Receiver (subprocess target)

Receives RGBA frames from TouchDesigner via CUDA IPC.
Run as a subprocess launched by example_receiver_launcher.py (Execute DAT),
or directly from the command line:

    python td_exporter/example_receiver_python.py

Pipeline:  CUDAIPCLink_to_Python  (Sender mode, in TouchDesigner)
               ↓  CUDA IPC  (cudalink_input_ipc)
           this script  (separate OS process)
               ↓
           Prints frame stats  →  shape, dtype, FPS, latency

TD Setup (handled by example_receiver_launcher.py Execute DAT):
    CUDAIPCLink_to_Python → Mode=Sender, Ipcmemname=cudalink_input_ipc, Active=ON
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
import time

if os.environ.get("CUDALINK_IMPORT_PROFILE", "0") == "1":
    logging.basicConfig(level=logging.DEBUG, format="[lib] %(message)s", stream=sys.stdout)

_probe_log_file = os.environ.get("CUDALINK_PROBE_LOG_FILE", "")
if _probe_log_file:
    _root_logger = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) for h in _root_logger.handlers):
        _fh = logging.FileHandler(_probe_log_file, mode="w", encoding="utf-8")
        _fh.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
        _root_logger.addHandler(_fh)
        if _root_logger.level == logging.NOTSET:
            _root_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Windows console control handler — ensures CUDA IPC cleanup runs on
# console X-button close (CTRL_CLOSE_EVENT), which does NOT raise
# KeyboardInterrupt in Python by default.
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    from ctypes import wintypes as _wintypes

    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1
    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6

    _HandlerRoutine = ctypes.WINFUNCTYPE(_wintypes.BOOL, _wintypes.DWORD)

    _k32 = ctypes.windll.kernel32
    # arg 0 is PHANDLER_ROUTINE — a Win32 function pointer. We use c_void_p (not
    # WINFUNCTYPE) because the documented "restore default Ctrl+C" call passes NULL
    # there, and ctypes refuses None for a strict WINFUNCTYPE argtype. c_void_p
    # accepts both None (== NULL) and WINFUNCTYPE instances (same pointer ABI).
    _k32.SetConsoleCtrlHandler.argtypes = [ctypes.c_void_p, _wintypes.BOOL]
    _k32.SetConsoleCtrlHandler.restype = _wintypes.BOOL

# Module-level refs so the handler thread can access them regardless of stack.
_importer_ref = None
_cleaned_up = False
_shutdown_via: str | None = None


def _do_cleanup() -> None:
    """Idempotent CUDA IPC cleanup — safe to call from handler thread and from finally:."""
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    try:
        if _importer_ref is not None:
            _importer_ref.close()
    except Exception as exc:
        print(f"[receiver] cleanup: importer.close error: {exc}")


if sys.platform == "win32":

    def _ctrl_handler(ctrl_type: int) -> bool:
        global _shutdown_via
        if ctrl_type == CTRL_C_EVENT:
            _shutdown_via = "ctrl_c"
            print("\n[receiver] Ctrl+C — stopping ...", flush=True)
            return False
        if ctrl_type == CTRL_BREAK_EVENT:
            _shutdown_via = "ctrl_break"
            print("\n[receiver] Ctrl+Break / launcher shutdown — stopping ...", flush=True)
            return False
        if ctrl_type in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
            print(
                f"\n[receiver] Console control event {ctrl_type} (close/logoff/shutdown) — running cleanup ...",
                flush=True,
            )
            _do_cleanup()
            print("[receiver] Cleanup complete.", flush=True)
            return True
        return False

    _k32.SetConsoleCtrlHandler(None, False)

    _ctrl_handler_ref = _HandlerRoutine(_ctrl_handler)
    if not _k32.SetConsoleCtrlHandler(_ctrl_handler_ref, True):
        print("[receiver] WARNING: SetConsoleCtrlHandler failed — console-close cleanup unavailable")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHM_NAME = "cudalink_input_ipc"
DEVICE = 0
TIMEOUT_MS = 5000.0
REPORT_EVERY = 150  # Print status every N frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global _importer_ref

    try:
        from cuda_link import ImportOutcome, ImportSpec, Importer
    except ImportError:
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
        src_dir = os.path.normpath(src_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            from cuda_link import ImportOutcome, ImportSpec, Importer
        except ImportError:
            print(f"[receiver] ERROR: cuda_link not found. Searched: {src_dir}")
            print("[receiver]   Run: pip install cuda-link  (from the project root)")
            sys.exit(1)

    print("=" * 58)
    print("  CUDA-Link Example  --  TouchDesigner -> Python Receiver")
    print("=" * 58)
    print(f"  channel   : {SHM_NAME}")
    print(f"  device    : {DEVICE}")
    print(f"  timeout   : {TIMEOUT_MS:.0f} ms")
    print()
    print("  TD: CUDAIPCLink_to_Python  Mode=Sender  Active=ON")
    print()

    spec = ImportSpec(
        shm_name=SHM_NAME,
        device=DEVICE,
        shape=None,   # auto-detected from TD sender
        dtype=None,   # auto-detected from TD sender
        timeout_ms=TIMEOUT_MS,
    )

    print("[receiver] Opening CUDA IPC channel — waiting for TD sender to publish ...\n")
    try:
        importer = Importer.open(spec)
    except Exception as exc:
        print(f"[receiver] ERROR: Importer.open() failed: {exc}")
        sys.exit(1)
    _importer_ref = importer

    profile_on = os.environ.get("CUDALINK_IMPORT_PROFILE", "0") == "1"
    frame_count = 0
    no_frame_count = 0
    start_time = time.perf_counter()
    last_report = start_time

    try:
        while True:
            result = importer.get_frame_numpy()

            if result.outcome is ImportOutcome.NEW_FRAME:
                frame = result.frame
                frame_count += 1
                no_frame_count = 0

                now = time.perf_counter()
                if frame_count % REPORT_EVERY == 0 or (now - last_report) >= 5.0:
                    elapsed = now - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                    latency_ms = importer.last_latency
                    if profile_on:
                        stats = importer.get_stats()
                        profile_suffix = (
                            f" | wait={stats.get('total_wait_event_time', 0.0) / max(frame_count, 1):.1f} µs/f"
                        )
                    else:
                        profile_suffix = ""
                    print(
                        f"  Frame {frame_count:5d} | {fps:5.1f} FPS | "
                        f"shape={frame.shape} dtype={frame.dtype} | "
                        f"latency={latency_ms:.2f} ms"
                        f"{profile_suffix}"
                    )
                    last_report = now

            elif result.outcome is ImportOutcome.NO_FRAME:
                no_frame_count += 1
                time.sleep(0.001)

            elif result.outcome is ImportOutcome.SHUTDOWN:
                print("[receiver] TD sender shut down — exiting.")
                break

            elif result.outcome is ImportOutcome.RECONNECTING:
                print("[receiver] Producer restarted — reconnecting ...", flush=True)

            elif result.outcome is ImportOutcome.TIMEOUT:
                print(f"[receiver] Frame wait timed out after {TIMEOUT_MS:.0f} ms — TD sender may be paused.")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n[receiver] Stopped after {frame_count} frames.")

    finally:
        _do_cleanup()
        total = time.perf_counter() - start_time
        avg_fps = frame_count / total if total > 0 else 0.0
        print(
            f"[receiver] Done — {frame_count} frames in {total:.1f}s  ({avg_fps:.1f} FPS avg)",
            flush=True,
        )
        print("[receiver] TD Sender will detect consumer disconnect on next cook.", flush=True)

        if _shutdown_via != "ctrl_break":
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input("\n[receiver] Press Enter to close this window ...")


if __name__ == "__main__":
    main()
