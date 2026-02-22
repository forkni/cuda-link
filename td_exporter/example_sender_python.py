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

import ctypes
import os
import struct
import sys
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHM_NAME         = "cudalink_output_ipc"
WIDTH            = 512
HEIGHT           = 512
DTYPE            = "uint8"       # "uint8" or "float32"
NUM_SLOTS        = 3
TARGET_FPS       = 60.0
FRAMES_PER_COLOR = 30            # Hold each solid color this many frames
REPORT_EVERY     = 150           # Print status every N frames


# ---------------------------------------------------------------------------
# Color cycle  (RGBA uint8)
# ---------------------------------------------------------------------------

_COLORS = [
    (255,   0,   0, 255),   # Red
    (  0, 255,   0, 255),   # Green
    (  0,   0, 255, 255),   # Blue
    (255, 255,   0, 255),   # Yellow
    (  0, 255, 255, 255),   # Cyan
    (255,   0, 255, 255),   # Magenta
    (255, 255, 255, 255),   # White
    ( 64,  64,  64, 255),   # Grey
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
        data  = pixel * (data_size // 4)
        buf   = (ctypes.c_uint8 * data_size).from_buffer_copy(data)
    else:  # float32
        pixel = struct.pack("<4f", r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        data  = pixel * (data_size // 16)
        buf   = (ctypes.c_uint8 * data_size).from_buffer_copy(data)

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
            print( "[sender]   Run: pip install cuda-link  (from the project root)")
            sys.exit(1)

    cuda = get_cuda_runtime()

    print("=" * 58)
    print("  CUDA-Link Example  —  Python → TouchDesigner Sender")
    print("=" * 58)
    print(f"  channel   : {SHM_NAME}")
    print(f"  resolution: {WIDTH}x{HEIGHT}  RGBA  {DTYPE}")
    print(f"  fps target: {TARGET_FPS}")
    print()
    print(f"  TD: CUDAIPCLink_from_Python  Mode=Receiver  Active=ON")
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

    if not exporter.initialize():
        print("[sender] ERROR: exporter.initialize() failed.")
        sys.exit(1)

    print(f"[sender] Initialized — waiting for TD receiver to connect ...\n")

    staging_ptr    = cuda.malloc(exporter.data_size)
    frame_interval = 1.0 / TARGET_FPS
    frame_count    = 0
    start_time     = time.perf_counter()
    last_report    = start_time

    try:
        while True:
            t0        = time.perf_counter()
            color_idx = (frame_count // FRAMES_PER_COLOR) % len(_COLORS)
            color     = _COLORS[color_idx]

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
                print(
                    f"  Frame {frame_count:5d} | {fps:5.1f} FPS | "
                    f"color={_COLOR_NAMES[color_idx]:<8s} | export={export_us:.0f} µs"
                )
                last_report = now

            remaining = frame_interval - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print(f"\n[sender] Stopped after {frame_count} frames.")

    finally:
        cuda.free(staging_ptr)
        exporter.cleanup()
        total = time.perf_counter() - start_time
        avg_fps = frame_count / total if total > 0 else 0.0
        print(f"[sender] Done — {frame_count} frames in {total:.1f}s  ({avg_fps:.1f} FPS avg)")
        print( "[sender] TD Receiver will detect shutdown on next cook.")


if __name__ == "__main__":
    main()
