"""
CUDA-Link Example — Python Receiver (subprocess target)

Receives RGBA frames from TouchDesigner via CUDA IPC.
Run as a subprocess launched by example_receiver_launcher.py (Execute DAT),
or directly from the command line:

    python td_exporter/example_receiver_python.py

Pipeline:  CUDAIPCLink_to_Python  (Sender mode, in TouchDesigner)
               ↓  CUDA IPC  (cudalink_ipc_TD>>Python)
           this script  (separate OS process)
               ↓
           Prints frame stats  →  shape, dtype, FPS, latency, get_frame µs

TD Setup (handled by example_receiver_launcher.py Execute DAT):
    CUDAIPCLink_to_Python → Mode=Sender, Ipcmemname=cudalink_ipc_TD>>Python, Active=ON

Environment variables (all optional):
    CUDALINK_RECEIVER_SHM_NAME     IPC channel name          (default: cudalink_ipc_TD>>Python)
    CUDALINK_RECEIVER_DEVICE       GPU device index           (default: 0)
    CUDALINK_RECEIVER_TIMEOUT_MS   Frame-wait timeout ms      (default: 5000)
    CUDALINK_RECEIVER_REPORT_EVERY Frames between status lines (default: 150)
    CUDALINK_RECEIVER_FRAME_MODE   numpy | torch | cupy       (default: torch)
    CUDALINK_IMPORT_PROFILE        1 = enable lib debug logging
"""

from __future__ import annotations

import contextlib
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
# Ensure cuda_link is importable for the console helper (may need sys.path
# patch when run without a pip install, same pattern as main() below).
# ---------------------------------------------------------------------------

try:
    from cuda_link._console import install_console_ctrl_handler
except ImportError:
    _src = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from cuda_link._console import install_console_ctrl_handler

# ---------------------------------------------------------------------------
# Windows console control handler — ensures CUDA IPC cleanup runs on
# console X-button close (CTRL_CLOSE_EVENT), which does NOT raise
# KeyboardInterrupt in Python by default.
# ---------------------------------------------------------------------------

# Module-level ref so _do_cleanup can access the importer regardless of call stack.
_importer_ref = None
_cleaned_up = False


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


# _shutdown.shutdown_via tracks which event triggered shutdown (controls the
# end-of-main "Press Enter" pause).
_shutdown = install_console_ctrl_handler("[receiver]", _do_cleanup)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHM_NAME = os.environ.get("CUDALINK_RECEIVER_SHM_NAME", "cudalink_ipc_TD>>Python")
DEVICE = int(os.environ.get("CUDALINK_RECEIVER_DEVICE", "0"))
TIMEOUT_MS = float(os.environ.get("CUDALINK_RECEIVER_TIMEOUT_MS", "5000"))
REPORT_EVERY = int(os.environ.get("CUDALINK_RECEIVER_REPORT_EVERY", "150"))
FRAME_MODE = os.environ.get("CUDALINK_RECEIVER_FRAME_MODE", "torch").lower()


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
    print(f"  mode      : {FRAME_MODE}")
    print()
    print("  TD: CUDAIPCLink_to_Python  Mode=Sender  Active=ON")
    print()

    spec = ImportSpec(
        shm_name=SHM_NAME,
        device=DEVICE,
        shape=None,  # auto-detected from TD sender
        dtype=None,  # auto-detected from TD sender
        timeout_ms=TIMEOUT_MS,
    )

    print("[receiver] Opening CUDA IPC channel — waiting for TD sender to publish ...\n")
    try:
        importer = Importer.open(spec)
    except Exception as exc:
        print(f"[receiver] ERROR: Importer.open() failed: {exc}")
        sys.exit(1)
    _importer_ref = importer

    # Resolve frame-fetch callable once — avoids per-iteration dispatch overhead
    # and lets the timing wrapper cover all three modes uniformly.
    if FRAME_MODE == "torch":
        get_frame = importer.get_frame
    elif FRAME_MODE == "cupy":
        get_frame = importer.get_frame_cupy
    elif FRAME_MODE == "numpy":
        get_frame = importer.get_frame_numpy
    else:
        print(f"[receiver] ERROR: unknown CUDALINK_RECEIVER_FRAME_MODE={FRAME_MODE!r} (expected: numpy, torch, cupy)")
        sys.exit(1)

    # Pre-flight: verify the requested library is importable now, before the loop starts.
    # Falls back to numpy with a warning rather than exiting mid-run on RuntimeError.
    effective_mode = FRAME_MODE
    if FRAME_MODE in ("torch", "cupy"):
        try:
            __import__(FRAME_MODE)
        except ImportError:
            print(
                f"[receiver] WARNING: {FRAME_MODE!r} not installed — falling back to numpy. "
                f"(Set CUDALINK_RECEIVER_FRAME_MODE=numpy to suppress.)",
                flush=True,
            )
            effective_mode = "numpy"
            get_frame = importer.get_frame_numpy

    profile_on = os.environ.get("CUDALINK_IMPORT_PROFILE", "0") == "1"
    frame_count = 0
    no_frame_count = 0
    start_time = time.perf_counter()
    last_report = start_time
    last_report_frame = 0  # frame_count at last status line — for windowed (instantaneous) FPS
    last_report_gf_s = 0.0  # get_frame_total_s at last status line — for windowed get_frame avg
    last_report_wait_s = 0.0  # total_wait_event_time at last status line — for windowed wait avg

    # Per-call timing accumulators — updated on NEW_FRAME only (excludes NO_FRAME sleeps).
    get_frame_total_s = 0.0
    get_frame_min_s = float("inf")
    get_frame_max_s = 0.0

    last_outcome = None  # tracks state transitions for RECONNECTING rate-limiting

    try:
        while True:
            gf_t0 = time.perf_counter()
            try:
                result = get_frame()
            except RuntimeError as exc:
                print(
                    f"[receiver] ERROR: get_frame() raised RuntimeError — is the {effective_mode!r} library installed?"
                )
                print(f"  {exc}")
                sys.exit(1)
            gf_dt = time.perf_counter() - gf_t0

            if result.outcome is ImportOutcome.NEW_FRAME:
                frame = result.frame
                assert frame is not None  # guaranteed when outcome is NEW_FRAME
                frame_count += 1
                no_frame_count = 0

                get_frame_total_s += gf_dt
                if gf_dt < get_frame_min_s:
                    get_frame_min_s = gf_dt
                if gf_dt > get_frame_max_s:
                    get_frame_max_s = gf_dt

                if last_outcome is ImportOutcome.RECONNECTING:
                    print("[receiver] Reconnected.", flush=True)

                now = time.perf_counter()
                if frame_count % REPORT_EVERY == 0 or (now - last_report) >= 5.0:
                    # Windowed FPS over the last report interval — reflects the CURRENT
                    # rate, not the lifetime cumulative average (which only climbs
                    # asymptotically and stays diluted by the pre-first-frame idle wait).
                    window_dt = now - last_report
                    window_frames = frame_count - last_report_frame
                    fps = window_frames / window_dt if window_dt > 0 else 0.0
                    latency_ms = importer.last_latency
                    # Windowed get_frame avg — reflects the CURRENT call cost over
                    # the last report interval, not the lifetime cumulative mean
                    # (which climbs asymptotically during warm-up and never resets).
                    window_gf_s = get_frame_total_s - last_report_gf_s
                    avg_gf_us = (window_gf_s / window_frames) * 1e6 if window_frames > 0 else 0.0
                    cur_wait = 0.0
                    if profile_on:
                        stats = importer.get_stats()
                        _w = stats.get("total_wait_event_time", 0.0)
                        cur_wait = _w if isinstance(_w, float) else 0.0
                        profile_suffix = f" | wait={(cur_wait - last_report_wait_s) / max(window_frames, 1):.1f} µs/f"
                    else:
                        profile_suffix = ""
                    print(
                        f"  Frame {frame_count:5d} | {fps:5.1f} FPS | "
                        f"shape={frame.shape} dtype={frame.dtype} | "
                        f"latency={latency_ms:.2f} ms | "
                        f"get_frame={avg_gf_us:.1f} µs avg"
                        f"{profile_suffix}"
                    )
                    last_report = now
                    last_report_frame = frame_count
                    last_report_gf_s = get_frame_total_s
                    last_report_wait_s = cur_wait

            elif result.outcome is ImportOutcome.NO_FRAME:
                no_frame_count += 1
                # R2 doorbell: block on the Win32 named event when enabled
                # (CUDALINK_DOORBELL=1); returns False immediately when disabled
                # or non-Windows, preserving the existing poll-sleep behaviour.
                if not importer.wait_for_doorbell(2.0):
                    time.sleep(0.001)

            elif result.outcome is ImportOutcome.SHUTDOWN:
                print("[receiver] TD sender shut down — exiting.")
                break

            elif result.outcome is ImportOutcome.RECONNECTING:
                if last_outcome is not ImportOutcome.RECONNECTING:
                    print("[receiver] Producer restarted — reconnecting ...", flush=True)

            elif result.outcome is ImportOutcome.TIMEOUT:
                print(f"[receiver] Frame wait timed out after {TIMEOUT_MS:.0f} ms — TD sender may be paused.")
                time.sleep(0.1)

            last_outcome = result.outcome

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
        if frame_count > 0:
            avg_us = (get_frame_total_s / frame_count) * 1e6
            min_us = get_frame_min_s * 1e6
            max_us = get_frame_max_s * 1e6
            print(
                f"[receiver] Perf: mode={effective_mode}  "
                f"get_frame avg={avg_us:.1f} µs  min={min_us:.1f} µs  max={max_us:.1f} µs  "
                f"(n={frame_count})",
                flush=True,
            )
        print("[receiver] TD Sender will detect consumer disconnect on next cook.", flush=True)

        if _shutdown.shutdown_via != "ctrl_break":
            with contextlib.suppress(EOFError, KeyboardInterrupt):
                input("\n[receiver] Press Enter to close this window ...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback as _tb

        _err = _tb.format_exc()
        _log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "receiver_error.log")
        try:
            with open(_log, "w", encoding="utf-8") as _f:
                _f.write(_err)
        except OSError:
            pass
        print(f"\n[receiver] FATAL — unhandled exception (log: {_log})\n{_err}", flush=True)
        with contextlib.suppress(EOFError, KeyboardInterrupt):
            input("[receiver] Press Enter to close this window ...")
        sys.exit(1)
