"""
Example 06 — graceful shutdown, reconnect, and Ctrl+C: the production template.

WHAT
    A consumer that survives everything a long-running installation throws at
    it: the producer restarting (twice, scripted), Ctrl+C, and the console
    window's X button — always releasing CUDA/SHM resources exactly once.
    This is docs/INTEGRATION_EXAMPLES.md Example 5 as a runnable script, and
    the skeleton to copy for any consumer that must run unattended.

HARDWARE
    NVIDIA GPU (--force-fake runs the same choreography without one).  numpy.
    CUDA-enabled torch → frames are consumed ZERO-COPY via get_frame();
    without it the loop falls back to get_frame_numpy().
    No TouchDesigner needed — a scripted producer restarts itself so the full
    lifecycle plays out unattended in ~15 seconds.

HOW TO RUN
    python examples/06_graceful_shutdown_and_reconnect.py
    python examples/06_graceful_shutdown_and_reconnect.py --sessions 3 --frames 60
    (try Ctrl+C or closing the console mid-run — cleanup still happens)

WHAT IT TEACHES
    * The two distinct "producer went away" signals and what each demands:
        SHUTDOWN      — clean close; the Importer is now CLOSED and it is on
                        YOU to open a new one when the producer returns.
        RECONNECTING  — in-place restart; the importer already reopened its
                        IPC handles, just keep polling.
      Which one you observe is a timing race — handle both.
    * TIMEOUT as the "producer died uncleanly" signal.
    * Idempotent cleanup wired to Windows console-control events via
      cuda_link._console.install_console_ctrl_handler (no-op off Windows),
      plus KeyboardInterrupt + finally — one cleanup path, three triggers.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

# Make _common importable when this file runs as a script from any cwd.
# Unconditional at module top: the spawn child re-imports this module too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common

# ---------------------------------------------------------------------------
# Cleanup wiring — module-level because the console handler fires on a
# non-main thread and must find the importer without any closure gymnastics.
# ---------------------------------------------------------------------------

_importer_ref: object = None
_cleaned_up = False


def _do_cleanup() -> None:
    """Release the importer exactly once, no matter who calls (handler, except, finally)."""
    global _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    importer = _importer_ref
    if importer is not None:
        try:
            importer.close()  # type: ignore[attr-defined] — Importer; idempotent
        except (RuntimeError, OSError) as e:
            print(f"[example-06] cleanup error (ignored): {e}", flush=True)
    print("[example-06] cleanup complete.", flush=True)


# ---------------------------------------------------------------------------
# Scripted producer — runs N sessions with gaps, like an operator restarting TD
# ---------------------------------------------------------------------------


def _session_producer_worker(
    shm_name: str,
    sessions: int,
    frames_per_session: int,
    gap_s: float,
    fps: float,
    force_fake: bool,
    rq: object,
) -> None:
    """Each session: open exporter → export frames → clean close.  Then a gap."""
    try:
        for s in range(sessions):
            _common.demo_producer_worker(
                shm_name,
                height=240,
                width=320,
                channels=4,
                dtype="uint8",
                num_frames=frames_per_session,
                fps=fps,
                num_slots=3,
                device=0,
                force_fake=force_fake,
                doorbell=False,
            )
            if s < sessions - 1:
                time.sleep(gap_s)  # producer offline — consumer must wait this out
        rq.put(("OK", sessions))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — report ANY child failure to the parent
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The resilient consumer
# ---------------------------------------------------------------------------


def main() -> int:
    global _importer_ref

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sessions", type=int, default=2, help="producer sessions to ride through (default 2)")
    parser.add_argument("--frames", type=int, default=90, help="frames per producer session (default 90)")
    parser.add_argument("--gap", type=float, default=2.0, help="seconds the producer stays offline between sessions")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--force-fake", action="store_true", help="run without a GPU (FakeCUDAAdapter)")
    args = parser.parse_args()

    from cuda_link import Importer, ImportOutcome, ImportPolicy, ImportSpec

    # Wire cleanup to Windows console-control events (Ctrl+C, Ctrl+Break, the
    # window's X button, logoff, shutdown).  defer_close=True: the X-button
    # handler only SETS stop_requested; cleanup then runs on the main thread,
    # which cannot race an in-flight CUDA call.  No-op on non-Windows.
    from cuda_link._console import install_console_ctrl_handler

    shutdown_state = install_console_ctrl_handler("[example-06]", _do_cleanup, defer_close=True)

    _common.banner("Spawn scripted producer (restarts itself between sessions)")
    shm_name = _common.unique_shm_name("cudalink_ex06")
    ctx = multiprocessing.get_context("spawn")
    rq = ctx.Queue()
    producer = ctx.Process(
        target=_session_producer_worker,
        args=(shm_name, args.sessions, args.frames, args.gap, args.fps, args.force_fake, rq),
        daemon=True,
    )
    producer.start()
    print(f"producer pid={producer.pid}  shm='{shm_name}'  sessions={args.sessions}  gap={args.gap}s")

    adapter, is_real = _common.pick_cuda_adapter(0, force_fake=args.force_fake)
    for note in _common.make_importer_open_safe(using_fake=not is_real):
        print("NOTE:", note)

    # Zero-copy get_frame() when CUDA torch is present (the production path —
    # example 03), numpy fallback otherwise.  The lifecycle logic below is
    # identical either way — and zero-copy survives reconnects too: the
    # importer rebuilds its tensor views when a session comes back.
    use_zero_copy = is_real and _common.torch_cuda_ready()
    print("consume path:", "zero-copy get_frame()" if use_zero_copy else "get_frame_numpy() fallback (D2H copy)")

    sessions_survived = 0
    frames_total = 0
    interrupted = False

    try:
        # OUTER loop = one iteration per producer session (connect → consume →
        # producer goes away → reconnect).  This is the shape of a consumer
        # that runs for days.
        while sessions_survived < args.sessions and not shutdown_state.stop_requested:
            # -- (re)connect ---------------------------------------------
            # Cheap SHM polling, no CUDA, until the producer (re)appears.  In
            # production you would loop forever; a demo bounds the wait.
            if not _common.wait_for_shm(shm_name, timeout_s=max(20.0, args.gap + 15.0)):
                print("producer never (re)appeared — giving up.")
                break
            time.sleep(0.3)  # settle: new session finishes writing IPC handles
            importer = Importer.open(
                ImportSpec(shm_name=shm_name, shape=None, dtype=None),  # geometry from SHM metadata
                policy=None if is_real else ImportPolicy.for_testing(),
                cuda=adapter,
            )
            _importer_ref = importer
            print(f"\nconnected — session {sessions_survived + 1}")

            # -- consume until THIS session ends --------------------------
            session_frames = 0
            while not shutdown_state.stop_requested:
                result = importer.get_frame() if use_zero_copy else importer.get_frame_numpy()

                if result.outcome is ImportOutcome.NEW_FRAME:
                    session_frames += 1
                    frames_total += 1
                    if session_frames % 30 == 0:
                        print(f"  session {sessions_survived + 1}: {session_frames} frames")

                elif result.outcome is ImportOutcome.NO_FRAME:
                    time.sleep(0.001)

                elif result.outcome is ImportOutcome.RECONNECTING:
                    # In-place restart already handled by the importer — the
                    # backoff between retries is ImportPolicy.reconnect_backoff_frames.
                    print("  RECONNECTING — importer reopened IPC handles itself")

                elif result.outcome is ImportOutcome.SHUTDOWN:
                    # Clean close: the importer is now closed for good.  Count
                    # the session and let the outer loop reconnect fresh.
                    sessions_survived += 1
                    print(f"  SHUTDOWN — session {sessions_survived} ended cleanly ({session_frames} frames)")
                    break

                elif result.outcome is ImportOutcome.TIMEOUT:
                    # No clean flag, no event — the producer died or hung.
                    # Same recovery as SHUTDOWN: reconnect via the outer loop.
                    print("  TIMEOUT — producer unresponsive; attempting reconnect")
                    break

            importer.close()  # idempotent — safe even after SHUTDOWN closed it
            _importer_ref = None

    except KeyboardInterrupt:
        # install_console_ctrl_handler chains Ctrl+C to Python's default, so
        # it surfaces here — on the main thread, between CUDA calls.
        interrupted = True
        print("\n[example-06] KeyboardInterrupt — shutting down cleanly ...")
    finally:
        _do_cleanup()
        producer.join(timeout=20.0)
        if producer.is_alive():
            producer.terminate()

    while not rq.empty():
        status, payload = rq.get_nowait()
        if status == "ERROR":
            print("producer ERROR:", payload)

    _common.banner("Results")
    print(f"survived {sessions_survived}/{args.sessions} producer sessions, {frames_total} frames total")
    if interrupted or shutdown_state.stop_requested:
        print("run was interrupted by the user — cleanup verified, exiting 0")
        return 0
    return 0 if sessions_survived == args.sessions and frames_total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
