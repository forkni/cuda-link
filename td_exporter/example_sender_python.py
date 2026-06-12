"""
CUDA-Link Example — Python Sender (subprocess target)

Sends animated solid RGBA color frames to TouchDesigner via CUDA IPC.
Run as a subprocess launched by example_sender_launcher.py (Execute DAT),
or directly from the command line:

    python td_exporter/example_sender_python.py

Pipeline:  this script  (separate OS process)
               ↓  CUDA IPC  (cudalink_ipc_Python>>TD)
           CUDAIPCLink_from_Python  (Receiver mode, in TouchDesigner)
               ↓
           Script TOP output  →  cycling solid colors

TD Setup (handled by example_sender_launcher.py Execute DAT):
    CUDAIPCLink_from_Python → Mode=Receiver, Ipcmemname=cudalink_ipc_Python>>TD, Active=ON
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
# Ensure cuda_link is importable for the console helper (may need sys.path
# patch when run without a pip install, same pattern as main() below).
# ---------------------------------------------------------------------------

try:
    from cuda_link._console import install_console_ctrl_handler, run_with_watchdog
except ImportError:
    _src = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from cuda_link._console import install_console_ctrl_handler, run_with_watchdog

# ---------------------------------------------------------------------------
# Windows console control handler — ensures GPU IPC cleanup runs even when
# the user closes the console window via the X button (CTRL_CLOSE_EVENT),
# which does NOT raise KeyboardInterrupt in Python by default.
#
# defer_close=True: CTRL_CLOSE_EVENT sets _shutdown.stop_requested so the main
# loop breaks and runs _do_cleanup() from the main thread.  This avoids the race
# between the handler thread and an in-flight cudaMemcpy in _fill_ctypes.
# ---------------------------------------------------------------------------

# Module-level refs so _do_cleanup can access them regardless of call stack.
_cuda_ref = None
_exporter_ref = None
_staging_ptr_ref = None
_cleaned_up = False


def _do_cleanup() -> None:
    """Idempotent GPU IPC cleanup — safe to call from handler thread and from finally:."""
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True

    # Under ncu kernel-replay the GPU command queue is paused inside ncu's replay
    # state. cudaFree on the staging buffer implicitly synchronises the device and
    # blocks until the queue drains — which never happens in that state, causing a
    # 30+ s hang. The 1 MB staging buffer is reclaimed by the OS on process exit.
    if _staging_ptr_ref is not None and _cuda_ref is not None:

        def _free_staging() -> None:
            try:
                _cuda_ref.free(_staging_ptr_ref)
            except Exception as exc:
                print(f"[sender] cleanup: cuda.free(staging) error: {exc}", flush=True)

        run_with_watchdog(_free_staging, timeout_s=0.5, label="cudaFree(staging)", prefix="[sender]")

    # Under ncu kernel-replay, Steps 1c/2/3 of Exporter.close()
    # (graph_exec_destroy, destroy_event, destroy_stream) can block on a
    # paused command queue.  Bound total cleanup time so main returns and ncu finalizes.
    if _exporter_ref is not None:

        def _do_exporter_cleanup() -> None:
            try:
                _exporter_ref.close()
            except Exception as exc:
                print(f"[sender] cleanup: exporter.close error: {exc}", flush=True)

        run_with_watchdog(_do_exporter_cleanup, timeout_s=3.0, label="exporter.close()", prefix="[sender]")


# _shutdown.shutdown_via tracks which event triggered shutdown (controls the
# end-of-main "Press Enter" pause). _shutdown.stop_requested is polled by the
# main loop instead of a raw global bool.
_shutdown = install_console_ctrl_handler("[sender]", _do_cleanup, defer_close=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHM_NAME = "cudalink_ipc_Python>>TD"
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
        from cuda_link import FrameSpec, GpuFrame
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
        from cuda_link.exporter import Exporter
    except ImportError:
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
        src_dir = os.path.normpath(src_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            from cuda_link import FrameSpec, GpuFrame
            from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
            from cuda_link.exporter import Exporter
        except ImportError:
            print(f"[sender] ERROR: cuda_link not found. Searched: {src_dir}")
            print("[sender]   Run: pip install cuda-link  (from the project root)")
            sys.exit(1)

    cuda = get_cuda_runtime()
    _cuda_ref = cuda

    print("=" * 58)
    print("  CUDA-Link Example  --  Python -> TouchDesigner Sender")
    print("=" * 58)
    print(f"  channel   : {SHM_NAME}")
    print(f"  resolution: {WIDTH}x{HEIGHT}  RGBA  {DTYPE}")
    print(f"  fps target: {TARGET_FPS}")
    print()
    print("  TD: CUDAIPCLink_from_Python  Mode=Receiver  Active=ON")
    print()

    try:
        exporter = Exporter.open(
            FrameSpec(
                shm_name=SHM_NAME,
                height=HEIGHT,
                width=WIDTH,
                channels=4,
                dtype=DTYPE,
                num_slots=NUM_SLOTS,
            )
        )
    except Exception as exc:
        print(f"[sender] ERROR: Exporter.open() failed: {exc}")
        sys.exit(1)
    _exporter_ref = exporter

    graphs_active = bool(exporter._policy.use_graphs and not exporter._graphs_disabled)
    graphs_label = "ON" if graphs_active else "OFF"
    profile_on = os.environ.get("CUDALINK_EXPORT_PROFILE", "0") == "1"
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
    last_report_frame = 0  # frame_count at last status line — for windowed (instantaneous) FPS

    try:
        while not _shutdown.stop_requested:
            t0 = time.perf_counter()
            color_idx = (frame_count // FRAMES_PER_COLOR) % len(_COLORS)
            color = _COLORS[color_idx]

            _fill_ctypes(cuda, staging_ptr, exporter.data_size, color)
            # _fill_ctypes uses a synchronous H2D cudaMemcpy, so the write is complete
            # before this call returns.  Pass producer_stream=0 (the CUDA legacy default
            # stream) to arm cross-stream ordering — real kernel producers MUST pass the
            # actual stream their kernels run on instead.
            exporter.export(GpuFrame(ptr=int(staging_ptr.value), size=exporter.data_size, producer_stream=0))
            frame_count += 1

            now = time.perf_counter()
            if frame_count % REPORT_EVERY == 0 or (now - last_report) >= 5.0:
                # Windowed FPS over the last report interval — reflects the CURRENT
                # rate, not the lifetime cumulative average (which only climbs
                # asymptotically and stays diluted by the pre-first-frame idle wait).
                window_dt = now - last_report
                window_frames = frame_count - last_report_frame
                fps = window_frames / window_dt if window_dt > 0 else 0.0
                export_us = (now - t0) * 1e6
                if profile_on:
                    stats = exporter.get_stats()
                    profile_suffix = (
                        f" | avg_total={stats.get('avg_total_us', 0.0):.1f} µs"
                        f" | avg_memcpy={stats.get('avg_memcpy_us', 0.0):.1f} µs"
                    )
                else:
                    profile_suffix = ""
                print(
                    f"  Frame {frame_count:5d} | {fps:5.1f} FPS | "
                    f"color={_COLOR_NAMES[color_idx]:<8s} | "
                    f"export={export_us:.0f} µs"
                    f"{profile_suffix} | "
                    f"graphs={graphs_label}"
                )
                last_report = now
                last_report_frame = frame_count

            remaining = frame_interval - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print(f"\n[sender] Stopped after {frame_count} frames.")

    finally:
        try:
            final_stats = exporter.get_stats() if profile_on else {}
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
        if _shutdown.shutdown_via not in ("ctrl_break", "ctrl_close"):
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input("\n[sender] Press Enter to close this window ...")


if __name__ == "__main__":
    main()
