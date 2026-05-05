"""
CUDA-Link Example — Python Sender (subprocess target)

Sends animated solid RGBA color frames to TouchDesigner via CUDA IPC.
Run as a subprocess launched by example_sender_launcher.py (Execute DAT),
or directly from the command line:

    python td_exporter/example_sender_python.py

Pipeline:  this script  (separate OS process)
               ↓  CUDA IPC  (cudalink_output_ipc)
           CUDAIPCLink_from_Python  (Receiver mode, in TouchDesigner)
               ↓
           Script TOP output  →  cycling solid colors

TD Setup (handled by example_sender_launcher.py Execute DAT):
    CUDAIPCLink_from_Python → Mode=Receiver, Ipcmemname=cudalink_output_ipc, Active=ON
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import struct
import sys
import time

# When CUDALINK_EXPORT_PROFILE=1 the lib promotes self.debug=True and emits
# [PROFILE] lines via logger.debug(). Configure the root logger so those
# messages reach stdout (standard Python logging convention requires the host
# application to set up handlers; the lib itself cannot do it).
if os.environ.get("CUDALINK_EXPORT_PROFILE", "0") == "1":
    logging.basicConfig(level=logging.DEBUG, format="[lib] %(message)s", stream=sys.stdout)

# ---------------------------------------------------------------------------
# Windows console control handler — ensures GPU IPC cleanup runs even when
# the user closes the console window via the X button (CTRL_CLOSE_EVENT),
# which does NOT raise KeyboardInterrupt in Python by default.
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    from ctypes import wintypes as _wintypes

    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1
    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6

    _HandlerRoutine = ctypes.WINFUNCTYPE(_wintypes.BOOL, _wintypes.DWORD)

# Module-level refs so the handler thread can access them regardless of stack.
_cuda_ref = None
_exporter_ref = None
_staging_ptr_ref = None
_cleaned_up = False
# Track which event triggered shutdown — controls end-of-main "Press Enter" pause:
#   "ctrl_c" → user pressed Ctrl+C in console → pause (let user read messages).
#   "ctrl_break" → launcher sent CTRL_BREAK_EVENT (graceful .toe close) → no pause.
#   None → main loop exited some other way → pause as a safety net.
_shutdown_via: str | None = None


def _do_cleanup() -> None:
    """Idempotent GPU IPC cleanup — safe to call from handler thread and from finally:."""
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    try:
        if _staging_ptr_ref is not None and _cuda_ref is not None:
            _cuda_ref.free(_staging_ptr_ref)
    except Exception as exc:
        print(f"[sender] cleanup: cuda.free error: {exc}")
    try:
        if _exporter_ref is not None:
            _exporter_ref.cleanup()
    except Exception as exc:
        print(f"[sender] cleanup: exporter.cleanup error: {exc}")


if sys.platform == "win32":

    def _ctrl_handler(ctrl_type: int) -> bool:
        global _shutdown_via
        if ctrl_type == CTRL_C_EVENT:
            _shutdown_via = "ctrl_c"
            print("\n[sender] Ctrl+C — stopping ...", flush=True)
            return False  # Chain to Python's default → raises KeyboardInterrupt in main.
        if ctrl_type == CTRL_BREAK_EVENT:
            _shutdown_via = "ctrl_break"
            print("\n[sender] Ctrl+Break / launcher shutdown — stopping ...", flush=True)
            return False  # Chain to Python's default → raises KeyboardInterrupt in main.
        if ctrl_type in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
            # Console X-button, user logoff, or system shutdown.
            # OS allows ~5 s (CLOSE) or ~20 s (LOGOFF/SHUTDOWN) before forced termination.
            # The main loop's finally: won't run for these — we MUST clean up here.
            print(
                f"\n[sender] Console control event {ctrl_type} (close/logoff/shutdown) — running cleanup ...",
                flush=True,
            )
            _do_cleanup()
            print("[sender] Cleanup complete.", flush=True)
            return True  # Handled — OS still terminates after return.
        return False

    # The launcher uses CREATE_NEW_PROCESS_GROUP, which DISABLES Ctrl+C delivery to the
    # child process by default. SetConsoleCtrlHandler(NULL, FALSE) re-enables it before
    # we install our own handler.
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)

    # MUST be module-level; a local variable would be GC'd and Windows would call freed memory.
    _ctrl_handler_ref = _HandlerRoutine(_ctrl_handler)
    if not ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler_ref, True):
        print("[sender] WARNING: SetConsoleCtrlHandler failed — console-close cleanup unavailable")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHM_NAME = "cudalink_output_ipc"
WIDTH = 512
HEIGHT = 512
DTYPE = "uint8"  # "uint8" or "float32"
NUM_SLOTS = 3
TARGET_FPS = 60.0
FRAMES_PER_COLOR = 30  # Hold each solid color this many frames
REPORT_EVERY = 150  # Print status every N frames


# ---------------------------------------------------------------------------
# Color cycle  (RGBA uint8)
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 0, 0, 255),  # Red
    (0, 255, 0, 255),  # Green
    (0, 0, 255, 255),  # Blue
    (255, 255, 0, 255),  # Yellow
    (0, 255, 255, 255),  # Cyan
    (255, 0, 255, 255),  # Magenta
    (255, 255, 255, 255),  # White
    (64, 64, 64, 255),  # Grey
]
_COLOR_NAMES = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White", "Grey"]


# ---------------------------------------------------------------------------
# GPU fill helpers
# ---------------------------------------------------------------------------


def _fill_ctypes(cuda: object, ptr: object, data_size: int, color: tuple) -> None:
    """Write a solid RGBA color into a GPU buffer via H2D ctypes copy."""
    r, g, b, a = color
    if DTYPE == "uint8":
        pixel = bytes([int(r), int(g), int(b), int(a)])
        data = pixel * (data_size // 4)
        buf = (ctypes.c_uint8 * data_size).from_buffer_copy(data)
    else:  # float32
        pixel = struct.pack("<4f", r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        data = pixel * (data_size // 16)
        buf = (ctypes.c_uint8 * data_size).from_buffer_copy(data)

    cuda.memcpy(
        dst=ptr,
        src=ctypes.c_void_p(ctypes.addressof(buf)),
        count=data_size,
        kind=1,  # cudaMemcpyHostToDevice
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global _cuda_ref, _exporter_ref, _staging_ptr_ref
    # Ensure cuda_link is importable — try src/ relative to this script
    try:
        from cuda_link import CUDAIPCExporter
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
    except ImportError:
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
        src_dir = os.path.normpath(src_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            from cuda_link import CUDAIPCExporter
            from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
        except ImportError:
            print(f"[sender] ERROR: cuda_link not found. Searched: {src_dir}")
            print("[sender]   Run: pip install cuda-link  (from the project root)")
            sys.exit(1)

    cuda = get_cuda_runtime()
    _cuda_ref = cuda

    print("=" * 58)
    print("  CUDA-Link Example  —  Python → TouchDesigner Sender")
    print("=" * 58)
    print(f"  channel   : {SHM_NAME}")
    print(f"  resolution: {WIDTH}x{HEIGHT}  RGBA  {DTYPE}")
    print(f"  fps target: {TARGET_FPS}")
    print()
    print("  TD: CUDAIPCLink_from_Python  Mode=Receiver  Active=ON")
    print()

    exporter = CUDAIPCExporter(
        shm_name=SHM_NAME,
        height=HEIGHT,
        width=WIDTH,
        channels=4,
        dtype=DTYPE,
        num_slots=NUM_SLOTS,
        debug=False,
    )
    _exporter_ref = exporter

    if not exporter.initialize():
        print("[sender] ERROR: exporter.initialize() failed.")
        sys.exit(1)

    graphs_active = bool(
        getattr(exporter, "_use_graphs", False)
        and not getattr(exporter, "_graphs_disabled", False)
    )
    graphs_label = "ON" if graphs_active else "OFF"
    env_setting = os.environ.get("CUDALINK_USE_GRAPHS", "(default=1)")
    try:
        rt_version = cuda.get_runtime_version()
        rt_label = f"{rt_version // 1000}.{(rt_version % 1000) // 10}"
    except Exception:
        rt_version = 0
        rt_label = "unknown"
    print(f"[sender] cudart runtime: {rt_label} ({rt_version})")
    print(f"[sender] CUDA Graphs path: {graphs_label}  (CUDALINK_USE_GRAPHS={env_setting})")
    if not graphs_active and env_setting in ("1", "(default=1)"):
        print("[sender]   (graphs requested but disabled — see exporter logs for reason)")
    print("[sender] Initialized — waiting for TD receiver to connect ...\n")

    staging_ptr = cuda.malloc(exporter.data_size)
    _staging_ptr_ref = staging_ptr
    frame_interval = 1.0 / TARGET_FPS
    frame_count = 0
    start_time = time.perf_counter()
    last_report = start_time

    try:
        while True:
            t0 = time.perf_counter()
            color_idx = (frame_count // FRAMES_PER_COLOR) % len(_COLORS)
            color = _COLORS[color_idx]

            _fill_ctypes(cuda, staging_ptr, exporter.data_size, color)
            exporter.export_frame(
                gpu_ptr=int(staging_ptr.value),
                size=exporter.data_size,
            )
            frame_count += 1

            now = time.perf_counter()
            if frame_count % REPORT_EVERY == 0 or (now - last_report) >= 5.0:
                elapsed = now - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                export_us = (now - t0) * 1e6
                stats = exporter.get_stats()
                avg_total = stats.get("avg_total_us", 0.0)
                avg_memcpy = stats.get("avg_memcpy_us", 0.0)
                print(
                    f"  Frame {frame_count:5d} | {fps:5.1f} FPS | "
                    f"color={_COLOR_NAMES[color_idx]:<8s} | "
                    f"export={export_us:.0f} µs | "
                    f"avg_total={avg_total:.1f} µs | avg_memcpy={avg_memcpy:.1f} µs | "
                    f"graphs={graphs_label}"
                )
                last_report = now

            remaining = frame_interval - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print(f"\n[sender] Stopped after {frame_count} frames.")

    finally:
        try:
            final_stats = exporter.get_stats()
        except Exception:
            final_stats = {}
        _do_cleanup()
        total = time.perf_counter() - start_time
        avg_fps = frame_count / total if total > 0 else 0.0
        print(f"[sender] Done — {frame_count} frames in {total:.1f}s  ({avg_fps:.1f} FPS avg)", flush=True)
        if final_stats:
            print(
                f"[sender] Final stats: graphs={graphs_label}  "
                f"avg_total={final_stats.get('avg_total_us', 0.0):.1f} µs  "
                f"avg_memcpy={final_stats.get('avg_memcpy_us', 0.0):.1f} µs  "
                f"frames={final_stats.get('frame_count', 0)}",
                flush=True,
            )
        print("[sender] TD Receiver will detect shutdown on next cook.", flush=True)

        # Hold the console window open so the user can read the cleanup output —
        # but ONLY for user-initiated shutdowns. CTRL_BREAK_EVENT is also how the
        # launcher signals graceful .toe-close, so we skip the pause in that case.
        if _shutdown_via != "ctrl_break":
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input("\n[sender] Press Enter to close this window ...")


if __name__ == "__main__":
    main()
