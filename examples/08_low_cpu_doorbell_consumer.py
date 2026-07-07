"""
Example 08 — the doorbell: a near-zero-CPU consumer for slow or bursty streams.

WHAT
    Every previous consumer burned CPU while idle: NO_FRAME → sleep(0.001) →
    poll again, ~1000 wake-ups per second even when the producer sends 10 fps.
    This one opens both sides with doorbell=True and blocks in
    wait_for_doorbell() instead — the OS wakes it exactly when a frame lands.
    Docs/INTEGRATION_EXAMPLES.md Example 8, runnable.

HOW THE DOORBELL WORKS
    A Win32 named auto-reset event ("Local\\cudalink_db_<shm_name>"):
      * The PRODUCER creates it at Exporter.open(ExportPolicy(doorbell=True))
        and signals it after every published frame.
      * The CONSUMER opens it at Importer.open(ImportPolicy(doorbell=True))
        and blocks in wait_for_doorbell(timeout_ms).
      * BOTH sides must opt in.  If the producer never created the event,
        wait_for_doorbell() returns False immediately and you are back to
        polling — the degradation is automatic, no error.
    Auto-reset = single consumer: each signal wakes exactly ONE waiter.  Two
    consumers on one doorbell would each see every other frame — run N
    consumers as N streams (example 05), not N waiters on one event.

HARDWARE
    NVIDIA GPU; --force-fake demonstrates the identical wake-up mechanics
    without one (the doorbell is pure Win32, no CUDA involved).
    CUDA-enabled torch → frames are consumed ZERO-COPY via get_frame();
    without it the loop falls back to get_frame_numpy().
    Windows-only feature: elsewhere wait_for_doorbell() returns False and the
    script silently becomes a poll loop.  numpy required.

HOW TO RUN
    python examples/08_low_cpu_doorbell_consumer.py
    python examples/08_low_cpu_doorbell_consumer.py --fps 5 --frames 25
    python examples/08_low_cpu_doorbell_consumer.py --force-fake

WHAT IT TEACHES
    * ExportPolicy(doorbell=True) / ImportPolicy(doorbell=True) — explicit
      policy objects, not the CUDALINK_DOORBELL env var (ordering-sensitive).
    * The wait loop: poll first, and only when the poll says NO_FRAME block in
      wait_for_doorbell(2000.0).  NOTE the argument is MILLISECONDS.
    * Why it is race-free: wait_for_doorbell() is lost-wakeup-safe — it checks
      whether write_idx advanced before blocking and returns True immediately
      if a frame slipped in between your poll and your wait.
    * Why a bounded timeout beats blocking forever: a dead producer signals
      nothing, ever.  2000 ms of silence returns False and gives the loop a
      chance to notice SHUTDOWN/TIMEOUT instead of hanging.
    * The payoff, measured: polls-per-frame here vs. what the 1 ms sleep-poll
      loop of examples 02-07 would have cost over the same wall-clock time.
    * The production low-overhead consumer: wait_for_doorbell() combined with
      zero-copy get_frame() (example 03) — blocked while idle, zero copies
      when awake.  That is exactly what this script runs when CUDA torch is
      present; without it the same loop falls back to get_frame_numpy().
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

# Make _common importable when this file runs as a script from any cwd.
# Unconditional at module top: the spawn child re-imports this module too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--frames", type=int, default=50, help="frames to consume (default 50)")
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="demo producer rate (default 10 — slow on purpose, so idle time dominates)",
    )
    parser.add_argument("--force-fake", action="store_true", help="run without a GPU (FakeCUDAAdapter)")
    args = parser.parse_args()

    from cuda_link import Importer, ImportOutcome, ImportPolicy, ImportSpec

    if not args.force_fake:
        problem = _common.probe_cuda()
        if problem is not None:
            print(f"This example needs a real NVIDIA GPU (CUDA runtime unavailable: {problem}).")
            print("Re-run with --force-fake — the doorbell mechanics are identical without one.")
            return 2

    # ------------------------------------------------------------------
    # Producer with doorbell=True — it CREATES the named event and rings
    # it once per published frame.  Low fps makes the contrast stark: at
    # 10 fps the consumer is idle ~99% of the time.
    # ------------------------------------------------------------------
    _common.banner("Spawn demo producer (doorbell=True, deliberately slow)")
    shm_name = _common.unique_shm_name("cudalink_ex08")
    producer = _common.spawn_demo_producer(
        shm_name,
        height=240,
        width=320,
        dtype="uint8",
        num_frames=args.frames + 30,
        fps=args.fps,
        force_fake=args.force_fake,
        doorbell=True,
    )
    print(f"producer pid={producer.pid}  shm='{shm_name}'  rate={args.fps:.0f} fps")

    if not _common.wait_for_shm(shm_name, timeout_s=20.0):
        print("ERROR: producer never appeared.")
        return 1
    time.sleep(0.3)

    adapter, is_real = _common.pick_cuda_adapter(0, force_fake=args.force_fake)
    for note in _common.make_importer_open_safe(using_fake=not is_real):
        print("NOTE:", note)

    # ------------------------------------------------------------------
    # Consumer with doorbell=True — it OPENS the event the producer made.
    # ------------------------------------------------------------------
    _common.banner("Open Importer (doorbell=True)")
    policy = (
        # Production defaults + the doorbell opt-in.
        ImportPolicy(doorbell=True)
        if is_real
        # for_testing() strips driver-dependent machinery for the fake; the
        # doorbell rides on top (it is Win32, not CUDA).
        else dataclasses.replace(ImportPolicy.for_testing(), doorbell=True)
    )
    importer = Importer.open(
        ImportSpec(shm_name=shm_name, shape=(240, 320, 4), dtype="uint8"),
        policy=policy,
        cuda=adapter,
    )

    # Zero-copy get_frame() when CUDA torch is present (example 03), numpy
    # fallback otherwise.  Blocked while idle AND zero copies when awake —
    # that combination is the production low-overhead consumer.
    use_zero_copy = is_real and _common.torch_cuda_ready()
    print("consume path:", "zero-copy get_frame()" if use_zero_copy else "get_frame_numpy() fallback (D2H copy)")

    # ------------------------------------------------------------------
    # The doorbell loop: poll → if empty, BLOCK until rung → poll again
    # ------------------------------------------------------------------
    frames_done = 0
    polls = 0  # every get_frame*() poll
    doorbell_wakes = 0  # wait_for_doorbell() → True  (event rang / frame raced in)
    doorbell_timeouts = 0  # wait_for_doorbell() → False (silence, or doorbell unavailable)
    started = time.perf_counter()
    deadline = started + args.frames / max(args.fps, 1.0) + 30.0

    _common.banner(f"Consume {args.frames} frames (blocking between frames, not spinning)")
    while frames_done < args.frames and time.perf_counter() < deadline:
        # ALWAYS poll first.  The doorbell is a wake-up hint, not a frame
        # counter — after any wake you re-poll, and you never wait without
        # having just seen NO_FRAME.
        result = importer.get_frame() if use_zero_copy else importer.get_frame_numpy()
        polls += 1

        if result.outcome is ImportOutcome.NEW_FRAME:
            frame = result.frame  # torch.Tensor (zero-copy) or np.ndarray (fallback)
            frames_done += 1
            if frames_done % 10 == 0:
                # .item() works on both paths (tiny D2H sync for torch —
                # fine at print frequency).
                print(f"  {frames_done:>3} frames  (pixel[0,0,0]={frame[0, 0, 0].item()}, polls so far: {polls})")

        elif result.outcome is ImportOutcome.NO_FRAME:
            # Ring empty — go to sleep until the producer rings.  MILLISECONDS:
            # 2000.0 = block at most 2 s.  Lost-wakeup-safe: if a frame arrived
            # between our poll above and this call, it returns True instantly.
            if importer.wait_for_doorbell(2000.0):
                doorbell_wakes += 1  # rung (or frame already there) → re-poll now
            else:
                # False covers BOTH "2 s of real silence" (producer slow/dead —
                # the next polls will surface SHUTDOWN/TIMEOUT) and "doorbell
                # unavailable" (producer without doorbell=True, or non-Windows).
                # The tiny sleep keeps the unavailable case a civilised 1 ms
                # poll loop instead of a hot spin.
                doorbell_timeouts += 1
                time.sleep(0.001)

        elif result.outcome is ImportOutcome.RECONNECTING:
            continue  # producer restarted; importer already reopened handles

        elif result.outcome is ImportOutcome.SHUTDOWN:
            print("producer shut down cleanly — stopping.")
            break

        elif result.outcome is ImportOutcome.TIMEOUT:
            print("ERROR: producer stopped signalling (TIMEOUT) — aborting.")
            break

    elapsed = time.perf_counter() - started
    importer.close()
    producer.join(timeout=15.0)
    if producer.is_alive():
        producer.terminate()

    # ------------------------------------------------------------------
    # The receipt: what blocking saved vs. the sleep-poll loop
    # ------------------------------------------------------------------
    _common.banner("Results")
    spin_poll_estimate = int(elapsed * 1000)  # a 1 ms sleep-poll loop wakes ~1000x/s regardless of fps
    print(f"consumed {frames_done} frames in {elapsed:.1f} s at {args.fps:.0f} fps")
    print(f"  frame polls             : {polls}  (~{polls / max(frames_done, 1):.1f} per frame)")
    print(f"  doorbell wakes          : {doorbell_wakes}")
    print(f"  doorbell timeouts       : {doorbell_timeouts}")
    print(f"  a 1 ms sleep-poll loop would have made ~{spin_poll_estimate} calls over the same {elapsed:.1f} s")
    if doorbell_wakes == 0 and frames_done > 0:
        print(
            "  (doorbell never fired — non-Windows, or producer without doorbell=True: poll fallback carried the run)"
        )
    return 0 if frames_done > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
