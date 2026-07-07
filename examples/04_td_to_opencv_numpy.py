"""
Example 04 — TouchDesigner → NumPy/OpenCV: the CPU output path.

WHAT
    Consumes frames as plain numpy arrays via get_frame_numpy() — cuda-link
    performs the device-to-host copy for you through pinned staging memory —
    then displays them with OpenCV (if installed) or prints pixel statistics.
    This is docs/INTEGRATION_EXAMPLES.md Example 2, runnable.

WHEN TO CHOOSE THIS PATH
    Zero-copy torch/cupy (examples 03/07) is faster, but plenty of consumers
    are CPU code: OpenCV, scikit-image, encoders, disk writers.  A D2H copy
    costs ~0.5-2 ms for a 4K RGBA frame — often perfectly fine at 30-60 fps.
    Unlike the zero-copy tensor, the numpy array is YOUR private copy: keep
    it, queue it, mutate it — no ring-buffer lifetime contract to honour.

HARDWARE
    NVIDIA GPU (falls back to FakeCUDAAdapter with zero-valued pixels if none).
    numpy required; opencv-python optional and NOT a cuda-link extra:
        pip install opencv-python

HOW TO RUN
    python examples/04_td_to_opencv_numpy.py                # cv2 window if available
    python examples/04_td_to_opencv_numpy.py --headless     # stats printout only
    python examples/04_td_to_opencv_numpy.py --no-demo-producer --shm-name td_frames

WHAT IT TEACHES
    * get_frame_numpy(): the safe, dependency-light output mode (only numpy).
    * Ownership difference vs zero-copy: the returned array is a copy.
    * RGBA→BGRA conversion — OpenCV's channel order is not TouchDesigner's.
    * Graceful degradation when optional dependencies are missing.
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
    parser.add_argument("--frames", type=int, default=300, help="frames to consume (default 300)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=60.0, help="demo producer frame rate")
    parser.add_argument("--headless", action="store_true", help="never open a cv2 window")
    parser.add_argument("--force-fake", action="store_true", help="run without a GPU (FakeCUDAAdapter)")
    parser.add_argument("--shm-name", default=None, help="SHM name of a live producer (e.g. a TD sender)")
    parser.add_argument(
        "--no-demo-producer",
        action="store_true",
        help="do not spawn the demo producer; requires --shm-name of a live sender",
    )
    args = parser.parse_args()
    if args.no_demo_producer and not args.shm_name:
        parser.error("--no-demo-producer requires --shm-name (whom should I connect to?)")

    import numpy as np

    from cuda_link import Importer, ImportOutcome, ImportPolicy, ImportSpec

    # Optional dependency ladder: cv2 window when available, stats otherwise.
    have_cv2 = _common.probe_opencv()
    use_cv2 = have_cv2 and not args.headless
    if not have_cv2 and not args.headless:
        print("opencv-python not installed — running headless (pip install opencv-python for a window).")
    cv2 = None
    if use_cv2:
        import cv2  # deferred: only pay the import when we will actually display

    # ------------------------------------------------------------------
    # Producer (demo process stands in for TD) + adapter selection
    # ------------------------------------------------------------------
    shm_name = args.shm_name or _common.unique_shm_name("cudalink_ex04")
    producer = None
    if not args.no_demo_producer:
        _common.banner("Spawn demo producer")
        producer = _common.spawn_demo_producer(
            shm_name,
            height=args.height,
            width=args.width,
            dtype="uint8",
            num_frames=args.frames + 120,
            fps=args.fps,
            force_fake=args.force_fake,
        )
        print(f"producer pid={producer.pid}  shm='{shm_name}'")

    if not _common.wait_for_shm(shm_name, timeout_s=20.0):
        print(f"ERROR: no producer appeared on shm '{shm_name}' within 20 s.")
        return 1
    time.sleep(0.3)

    adapter, is_real = _common.pick_cuda_adapter(0, force_fake=args.force_fake)
    # Landmine preflight: a CPU-only torch install would crash Importer.open()
    # even though this example only ever calls get_frame_numpy().
    for note in _common.make_importer_open_safe(using_fake=not is_real):
        print("NOTE:", note)

    _common.banner("Open Importer (numpy mode)")
    importer = Importer.open(
        ImportSpec(shm_name=shm_name, shape=(args.height, args.width, 4), dtype="uint8"),
        policy=None if is_real else ImportPolicy.for_testing(),
        cuda=adapter,
    )

    # ------------------------------------------------------------------
    # Consume: display or print stats
    # ------------------------------------------------------------------
    frames_done = 0
    window_open = False
    try:
        while frames_done < args.frames:
            result = importer.get_frame_numpy()

            if result.outcome is ImportOutcome.NEW_FRAME:
                frame = result.frame  # np.ndarray (H, W, 4) uint8 — your own copy
                assert isinstance(frame, np.ndarray)
                frames_done += 1

                if use_cv2 and cv2 is not None:
                    # TD/cuda-link deliver RGBA; OpenCV renders BGR(A).  Skip
                    # this conversion and red/blue swap on screen.
                    bgra = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                    cv2.imshow("cuda-link example 04 (q to quit)", bgra)
                    window_open = True
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("window closed by user.")
                        break
                elif frames_done <= 3 or frames_done % 30 == 0:
                    # Headless: prove data is flowing with simple statistics.
                    print(
                        f"frame {frames_done:>4}: shape={frame.shape} "
                        f"min={frame.min()} max={frame.max()} mean={frame.mean():.1f}"
                    )

            elif result.outcome is ImportOutcome.NO_FRAME:
                time.sleep(0.001)
            elif result.outcome is ImportOutcome.RECONNECTING:
                continue
            elif result.outcome is ImportOutcome.SHUTDOWN:
                print("producer shut down cleanly — stopping.")
                break
            elif result.outcome is ImportOutcome.TIMEOUT:
                print("ERROR: producer stopped signalling (TIMEOUT) — aborting.")
                break
    finally:
        importer.close()
        if window_open and cv2 is not None:
            cv2.destroyAllWindows()
        if producer is not None:
            producer.join(timeout=15.0)
            if producer.is_alive():
                producer.terminate()

    _common.banner("Results")
    print(f"consumed {frames_done} frames via get_frame_numpy()")
    if not is_real:
        print("fake mode: pixel values are zeros by design; the protocol path is what ran.")
    return 0 if frames_done > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
