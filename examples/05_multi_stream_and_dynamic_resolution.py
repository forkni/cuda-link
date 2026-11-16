"""
Example 05 — multiple streams and dynamic resolution (auto-detect + reconnect).

WHAT
    Part A: one consumer process reading TWO independent streams (a "main"
    video feed and a smaller float32 "control" feed) — docs Example 3.
    Part B: a producer that restarts mid-run at a NEW resolution while the
    consumer, opened with shape=None, follows along — docs Example 4.

HARDWARE
    NVIDIA GPU (--force-fake runs the same protocol without one).  numpy.
    CUDA-enabled torch → both parts consume ZERO-COPY via get_frame();
    without it they fall back to get_frame_numpy() (one D2H copy per frame).
    No TouchDesigner needed — demo producers stand in for TD senders.

HOW TO RUN
    python examples/05_multi_stream_and_dynamic_resolution.py                 # both parts
    python examples/05_multi_stream_and_dynamic_resolution.py --part a
    python examples/05_multi_stream_and_dynamic_resolution.py --part b

WHAT IT TEACHES
    * The SHM name is the ONLY routing key: N streams = N names = N Importers.
      Different geometries, dtypes, and frame rates coexist freely.
    * ImportSpec(shape=None, dtype=None): geometry is auto-detected from the
      metadata block the producer writes into SHM — your consumer keeps
      working when the TD artist changes the render resolution.
    * What a producer restart looks like from the consumer: EITHER a clean
      SHUTDOWN (you reopen the Importer) OR RECONNECTING (the importer
      reopened IPC handles itself).  Which one you see is a timing race, so
      production code must handle both — this script does.
    * The consumption path is orthogonal to routing/reconnect logic: with
      CUDA torch the loops run zero-copy get_frame() (the production path —
      example 03) and fall back to get_frame_numpy() otherwise; the outcome
      handling is identical either way.  Zero-copy even survives the part B
      resolution change — the importer rebuilds its tensor views itself.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

# Make _common importable when this file runs as a script from any cwd.
# Unconditional at module top: the spawn child re-imports this module and
# needs the path too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common

# ---------------------------------------------------------------------------
# Part B producer — restarts at a new resolution (module level: spawn pickles it)
# ---------------------------------------------------------------------------

_PHASE_GEOMETRIES = [(180, 320), (360, 640)]  # (height, width) before/after "the artist resized"


def _resizing_producer_worker(
    shm_name: str,
    frames_per_phase: int,
    fps: float,
    gap_s: float,
    force_fake: bool,
    rq: object,
) -> None:
    """Export one phase per geometry, closing and reopening in between.

    This mimics TouchDesigner when the render resolution changes: the sender
    tears its exporter down and brings it back up on the SAME shm name with
    new geometry and a bumped SHM version.
    """
    try:
        for phase, (height, width) in enumerate(_PHASE_GEOMETRIES):
            _common.demo_producer_worker(
                shm_name,
                height=height,
                width=width,
                channels=4,
                dtype="uint8",
                num_frames=frames_per_phase,
                fps=fps,
                num_slots=3,
                device=0,
                force_fake=force_fake,
                doorbell=False,
            )
            if phase < len(_PHASE_GEOMETRIES) - 1:
                time.sleep(gap_s)  # the "gap" while TD rebuilds its network
        rq.put(("OK", len(_PHASE_GEOMETRIES)))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — report ANY child failure to the parent
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Part A — two independent streams into one consumer
# ---------------------------------------------------------------------------


def run_part_a(frames_per_stream: int, force_fake: bool) -> int:
    from cuda_link import Importer, ImportOutcome, ImportPolicy, ImportSpec

    _common.banner("Part A — two streams, one consumer process")

    # Each stream is fully described by its own SHM name + geometry.  Note the
    # control stream is float32 and runs at half the rate — total independence.
    streams: dict[str, dict] = {
        "main": {
            "shm": _common.unique_shm_name("cudalink_ex05_main"),
            "h": 360,
            "w": 640,
            "dtype": "uint8",
            "fps": 60.0,
        },
        "control": {
            "shm": _common.unique_shm_name("cudalink_ex05_ctl"),
            "h": 128,
            "w": 128,
            "dtype": "float32",
            "fps": 30.0,
        },
    }

    producers = []
    for cfg in streams.values():
        producers.append(
            _common.spawn_demo_producer(
                cfg["shm"],
                height=cfg["h"],
                width=cfg["w"],
                dtype=cfg["dtype"],
                num_frames=frames_per_stream + 120,
                fps=cfg["fps"],
                force_fake=force_fake,
            )
        )
    for name, cfg in streams.items():
        if not _common.wait_for_shm(cfg["shm"], timeout_s=20.0):
            print(f"ERROR: producer for stream '{name}' never appeared.")
            return 1
    time.sleep(0.3)

    adapter_note_needed = True
    importers: dict[str, Importer] = {}
    counts: dict[str, int] = {}
    for name, cfg in streams.items():
        adapter, is_real = _common.pick_cuda_adapter(0, force_fake=force_fake)
        if adapter_note_needed:
            for note in _common.make_importer_open_safe(using_fake=not is_real):
                print("NOTE:", note)
            adapter_note_needed = False
        importers[name] = Importer.open(
            ImportSpec(shm_name=cfg["shm"], shape=(cfg["h"], cfg["w"], 4), dtype=cfg["dtype"]),
            policy=None if is_real else ImportPolicy.for_testing(),
            cuda=adapter,
        )
        counts[name] = 0
        print(f"stream '{name}': {cfg['w']}x{cfg['h']} {cfg['dtype']} @ {cfg['fps']:.0f} fps  (shm '{cfg['shm']}')")

    # Zero-copy by default: with CUDA torch, get_frame() hands back tensors
    # that LIVE in each stream's ring buffer (example 03).  numpy is the
    # fallback for CPU-only torch, no torch, or --force-fake — same loop,
    # same outcomes, one extra D2H copy per frame.
    use_zero_copy = is_real and _common.torch_cuda_ready()
    print("consume path:", "zero-copy get_frame()" if use_zero_copy else "get_frame_numpy() fallback (D2H copy)")

    # Round-robin poll.  Both get_frame() and get_frame_numpy() return
    # NO_FRAME immediately when a stream has nothing new, so one loop
    # services any number of streams regardless of the consumption path.
    live = set(streams)
    deadline = time.perf_counter() + 60.0
    while live and time.perf_counter() < deadline:
        progressed = False
        for name in list(live):
            if counts[name] >= frames_per_stream:
                live.discard(name)
                continue
            result = importers[name].get_frame() if use_zero_copy else importers[name].get_frame_numpy()
            if result.outcome is ImportOutcome.NEW_FRAME:
                frame = result.frame  # torch.Tensor (zero-copy) or np.ndarray (fallback)
                counts[name] += 1
                progressed = True
                if counts[name] % 30 == 0:
                    # .item() works on both a torch GPU tensor element and a
                    # numpy scalar (for torch it is a tiny D2H sync — fine at
                    # print frequency, wrong per-frame).
                    print(
                        f"  {name:>7}: {counts[name]:>4} frames  "
                        f"(dtype={frame.dtype}, pixel[0,0,0]={frame[0, 0, 0].item()})"
                    )
            elif result.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                print(f"  {name:>7}: {result.outcome.name} — stream ended")
                live.discard(name)
        if not progressed:
            time.sleep(0.001)  # every stream idle this pass — yield once, not per stream

    for importer in importers.values():
        importer.close()
    for proc in producers:
        proc.join(timeout=15.0)
        if proc.is_alive():
            proc.terminate()

    print(f"\nPart A result: {counts}")
    ok = all(count >= min(frames_per_stream, 30) for count in counts.values())
    print("Part A", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Part B — resolution change mid-run, shape=None auto-detect
# ---------------------------------------------------------------------------


def run_part_b(frames_per_phase: int, force_fake: bool) -> int:
    from cuda_link import Importer, ImportOutcome, ImportPolicy, ImportSpec

    _common.banner("Part B — producer restarts at a new resolution")

    shm_name = _common.unique_shm_name("cudalink_ex05_dyn")
    ctx = multiprocessing.get_context("spawn")
    rq = ctx.Queue()
    producer = ctx.Process(
        target=_resizing_producer_worker,
        args=(shm_name, frames_per_phase, 60.0, 2.0, force_fake, rq),
        daemon=True,
    )
    producer.start()
    print(f"producer pid={producer.pid}  shm='{shm_name}'  phases={_PHASE_GEOMETRIES}")

    if not _common.wait_for_shm(shm_name, timeout_s=20.0):
        print("ERROR: producer never appeared.")
        return 1
    time.sleep(0.3)

    adapter, is_real = _common.pick_cuda_adapter(0, force_fake=force_fake)
    for note in _common.make_importer_open_safe(using_fake=not is_real):
        print("NOTE:", note)

    # Zero-copy consumption when CUDA torch is present (example 03), numpy
    # fallback otherwise.  Dynamic resolution works on BOTH paths: on a
    # geometry change the importer rebuilds its torch views / numpy staging
    # itself — the consumer code below never mentions a resolution.
    use_zero_copy = is_real and _common.torch_cuda_ready()
    print("consume path:", "zero-copy get_frame()" if use_zero_copy else "get_frame_numpy() fallback (D2H copy)")

    def open_importer() -> Importer:
        # shape=None + dtype=None → geometry comes from the SHM metadata the
        # producer wrote.  The consumer never hard-codes a resolution.
        return Importer.open(
            ImportSpec(shm_name=shm_name, shape=None, dtype=None),
            policy=None if is_real else ImportPolicy.for_testing(),
            cuda=adapter,
        )

    importer = open_importer()
    shapes_seen: list[tuple[int, ...]] = []
    frames = 0
    deadline = time.perf_counter() + 90.0

    while time.perf_counter() < deadline:
        result = importer.get_frame() if use_zero_copy else importer.get_frame_numpy()

        if result.outcome is ImportOutcome.NEW_FRAME:
            frame = result.frame  # torch.Tensor (zero-copy) or np.ndarray (fallback)
            frames += 1
            shape = tuple(frame.shape)  # torch.Size and numpy shape → one comparable form
            if shape not in shapes_seen:
                shapes_seen.append(shape)
                print(f"frame {frames:>4}: geometry now {shape} — auto-detected, no consumer code changed")

        elif result.outcome is ImportOutcome.NO_FRAME:
            time.sleep(0.001)

        elif result.outcome is ImportOutcome.RECONNECTING:
            # The importer noticed the SHM version bump and reopened the IPC
            # handles itself — the next NEW_FRAME will carry the new geometry.
            print("RECONNECTING: importer is following a producer restart in place")

        elif result.outcome is ImportOutcome.SHUTDOWN:
            # Clean close ends the importer for good — reopening is on us.
            # After the LAST phase the producer is gone for good, so a failed
            # wait here is the normal end of the run.
            if len(shapes_seen) >= len(_PHASE_GEOMETRIES):
                print("SHUTDOWN after final phase — done.")
                break
            print("SHUTDOWN: producer closed (resolution change in progress) — reopening ...")
            importer.close()
            if not _common.wait_for_shm(shm_name, timeout_s=15.0):
                print("ERROR: producer never came back after shutdown.")
                break
            time.sleep(0.5)  # settle: let the new session finish writing handles
            importer = open_importer()

        elif result.outcome is ImportOutcome.TIMEOUT:
            print("TIMEOUT: producer stalled — aborting.")
            break

    importer.close()
    producer.join(timeout=20.0)
    if producer.is_alive():
        producer.terminate()
    while not rq.empty():
        status, payload = rq.get_nowait()
        if status == "ERROR":
            print("producer ERROR:", payload)

    print(f"\nPart B result: {frames} frames across geometries {shapes_seen}")
    ok = len(shapes_seen) >= len(_PHASE_GEOMETRIES)
    print("Part B", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--part", choices=["a", "b", "both"], default="both")
    parser.add_argument("--frames", type=int, default=120, help="frames per stream (A) / per phase (B)")
    parser.add_argument("--force-fake", action="store_true", help="run without a GPU (FakeCUDAAdapter)")
    args = parser.parse_args()

    rc = 0
    if args.part in ("a", "both"):
        rc |= run_part_a(args.frames, args.force_fake)
    if args.part in ("b", "both"):
        rc |= run_part_b(args.frames, args.force_fake)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
