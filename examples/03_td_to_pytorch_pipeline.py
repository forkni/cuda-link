"""
Example 03 — TouchDesigner → PyTorch: the zero-copy GPU pipeline.

WHAT
    Consumes frames as torch.Tensor objects that live DIRECTLY in the shared
    ring buffer on the GPU — no device-to-host copy, no per-frame allocation —
    and feeds them through a small torch "model".  This is the flagship
    cuda-link use case (docs/INTEGRATION_EXAMPLES.md Example 1).

HARDWARE
    NVIDIA GPU + CUDA-enabled PyTorch (pip install -e ".[torch]").
    A CPU-only torch wheel cannot run this example — the whole point is the
    GPU-resident tensor.  The script checks and tells you before starting.
    No TouchDesigner needed: a demo producer process stands in for TD.

HOW TO RUN
    python examples/03_td_to_pytorch_pipeline.py
    python examples/03_td_to_pytorch_pipeline.py --frames 600 --width 1280 --height 720
    # attach to a live TouchDesigner sender instead of the demo producer:
    python examples/03_td_to_pytorch_pipeline.py --no-demo-producer --shm-name td_frames

WHAT IT TEACHES
    * get_frame() returns a zero-copy VIEW of a ring slot: same data_ptr()
      values cycle forever (one per slot) — proof no copy happens.
    * The lifetime contract that comes with zero-copy: the tensor's contents
      are only guaranteed until the producer laps that slot.  Use it (or
      .clone() it) promptly; never stash raw views for later.
    * A realistic preprocessing chain on the GPU: uint8 RGBA → float NCHW.
    * Windowed FPS + importer.last_latency for live pipeline health.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make _common importable when this file runs as a script from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--frames", type=int, default=300, help="frames to process (default 300)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=float, default=60.0, help="demo producer frame rate")
    parser.add_argument("--shm-name", default=None, help="SHM name of a live producer (e.g. a TD sender)")
    parser.add_argument(
        "--no-demo-producer",
        action="store_true",
        help="do not spawn the demo producer; requires --shm-name of a live sender",
    )
    args = parser.parse_args()
    if args.no_demo_producer and not args.shm_name:
        parser.error("--no-demo-producer requires --shm-name (whom should I connect to?)")

    # ------------------------------------------------------------------
    # Preflight — fail with instructions, not a stack trace
    # ------------------------------------------------------------------
    problem = _common.probe_cuda()
    if problem is not None:
        print(f"This example needs a real NVIDIA GPU (CUDA runtime unavailable: {problem}).")
        print("Run examples/01_fake_adapter_tour.py for the GPU-free walkthrough instead.")
        return 2
    if not _common.torch_cuda_ready():
        print("This example needs CUDA-enabled PyTorch (torch.cuda.is_available() is False).")
        print('Install it with:  pip install -e ".[torch]"   (see examples/README.md)')
        return 2

    import torch

    from cuda_link import Importer, ImportOutcome, ImportSpec

    # ------------------------------------------------------------------
    # Producer: demo process by default, or a live TD sender via --shm-name
    # ------------------------------------------------------------------
    shm_name = args.shm_name or _common.unique_shm_name("cudalink_ex03")
    producer = None
    if not args.no_demo_producer:
        _common.banner("Spawn demo producer (stands in for TouchDesigner)")
        producer = _common.spawn_demo_producer(
            shm_name,
            height=args.height,
            width=args.width,
            dtype="uint8",  # TD textures are typically 8-bit RGBA
            num_frames=args.frames + 120,  # ~2s surplus so we never starve
            fps=args.fps,
        )
        print(f"producer pid={producer.pid}  shm='{shm_name}'")

    if not _common.wait_for_shm(shm_name, timeout_s=20.0):
        print(f"ERROR: no producer appeared on shm '{shm_name}' within 20 s.")
        return 1
    time.sleep(0.3)  # settle: producer finishes writing IPC handles

    # ------------------------------------------------------------------
    # Consumer: Importer.open + a synthetic torch model
    # ------------------------------------------------------------------
    _common.banner("Open Importer (zero-copy torch mode)")
    importer = Importer.open(
        # shape/dtype given explicitly; they must match the producer.  See
        # example 05 for shape=None auto-detection from SHM metadata.
        ImportSpec(shm_name=shm_name, shape=(args.height, args.width, 4), dtype="uint8"),
    )

    # Stand-in "model" — swap in your real network here.  It lives on the GPU
    # so the whole pipeline (IPC frame → preprocess → inference) never leaves
    # the device.
    model = (
        torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        .cuda()
        .eval()
    )

    slot_ptrs: set[int] = set()  # distinct device addresses seen → ring slots
    frames_done = 0
    window_start = time.perf_counter()
    window_frames = 0

    _common.banner(f"Process {args.frames} frames")
    with torch.inference_mode():
        while frames_done < args.frames:
            result = importer.get_frame()  # blocks until the producer's event fires

            if result.outcome is ImportOutcome.NEW_FRAME:
                frame = result.frame  # torch.Tensor, device='cuda', shape (H, W, 4)
                assert isinstance(frame, torch.Tensor)

                # ZERO-COPY EVIDENCE: data_ptr() is the ring slot's IPC-mapped
                # address.  Only num_slots distinct values will ever appear —
                # no tensor is allocated per frame.
                slot_ptrs.add(frame.data_ptr())

                # LIFETIME CONTRACT: this tensor VIEWS shared memory the
                # producer will overwrite once the ring wraps.  Do your GPU
                # work now (or .clone() if you must keep the pixels).
                x = frame.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)  # HWC uint8 → NCHW float
                y = model(x)

                frames_done += 1
                window_frames += 1
                now = time.perf_counter()
                if now - window_start >= 1.0:
                    fps = window_frames / (now - window_start)
                    print(
                        f"{frames_done:>4} frames | {fps:5.1f} fps | "
                        f"ipc wait {importer.last_latency:5.2f} ms | "
                        f"model out mean {y.mean().item():+.4f} | "
                        f"pixel[0,0] {int(frame[0, 0, 0])}"
                    )
                    window_start, window_frames = now, 0

            elif result.outcome is ImportOutcome.NO_FRAME:
                time.sleep(0.001)  # ring not advanced yet — yield and re-poll

            elif result.outcome is ImportOutcome.RECONNECTING:
                continue  # producer restarted; importer already reopened handles

            elif result.outcome is ImportOutcome.SHUTDOWN:
                print("producer shut down cleanly — stopping.")
                break

            elif result.outcome is ImportOutcome.TIMEOUT:
                print("ERROR: producer stopped signalling (TIMEOUT) — aborting.")
                break

    # ------------------------------------------------------------------
    # Wrap up
    # ------------------------------------------------------------------
    _common.banner("Results")
    print(f"processed {frames_done} frames")
    print(f"distinct GPU addresses seen: {len(slot_ptrs)} (= ring slots; zero-copy confirmed)")
    stats = importer.get_stats()
    print(f"importer stats: frame_count={stats['frame_count']}  shape={stats['shape']}")
    importer.close()
    if producer is not None:
        producer.join(timeout=15.0)
        if producer.is_alive():
            producer.terminate()
    return 0 if frames_done > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
