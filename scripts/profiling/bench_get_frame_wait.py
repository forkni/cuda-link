"""
bench_get_frame_wait.py — live check of Importer.get_frame() call time against a running sender.

Splits each get_frame() call into the acquire step and the event wait (_wait_for_slot), and
reports how long after publish the sender's slot event completed. Use it against a live
TouchDesigner sender (C++ Out TOP or .tox Sender); it needs no TD-side change.

Why it exists (2026-11-18): a C++ Out TOP publishes without a sync, and its slot event
completes 0.11-0.28 ms after publish. With the former 200 us spin budget the wait fell into
the ~0.5 ms timer nap on many frames (get_frame p50 0.75 ms at 1280x1280 rgba8, p95 0.77 ms
at rgba32float with 25-40 % of frames napping). With the 500 us default no frame naps
(p95 0.27-0.33 ms). A .tox Sender syncs before publish and reads 0.05 ms either way.

The check is RED (exit code 1) when get_frame p50 is above --limit-ms (default 0.3) or when
more than --max-nap-pct percent of the frames fell out of the spin phase (default 2).

Usage
-----
Start the sender in TouchDesigner (Debug off, one TD window in focus), then:

    python scripts/profiling/bench_get_frame_wait.py <shm_name> [seconds]

    # the former default, to see the regression this guards against
    set CUDALINK_WAIT_SPIN_US=200
    python scripts/profiling/bench_get_frame_wait.py <shm_name> 5

    # true completion time of the sender's event (spin long enough to never nap)
    set CUDALINK_WAIT_SPIN_US=5000
    python scripts/profiling/bench_get_frame_wait.py <shm_name> 5

Run it at the largest format in use (rgba32float): the completion time grows a little with
the frame size, so that is where a too-short spin budget shows first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from cuda_link import Importer, ImportOutcome, ImportSpec


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))], 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shm_name")
    parser.add_argument("seconds", nargs="?", type=float, default=5.0)
    parser.add_argument("--limit-ms", type=float, default=0.3, help="RED when get_frame p50 is above this")
    parser.add_argument("--max-nap-pct", type=float, default=2.0, help="RED when more frames than this napped")
    args = parser.parse_args()

    imp = Importer.open(ImportSpec(shm_name=args.shm_name, device=0, shape=None, dtype=None, timeout_ms=5000))
    cur: dict[str, float | bool | None] = {"wait": 0.0, "done": None}
    orig_wait = imp._wait_for_slot

    def timed_wait(slot: int) -> float:
        evt = imp._conn.ipc_events[slot]
        cur["done"] = bool(imp._conn.cuda.query_event(evt)) if evt else None
        t0 = time.perf_counter()
        result = orig_wait(slot)
        cur["wait"] = (time.perf_counter() - t0) * 1e3
        return result

    imp._wait_for_slot = timed_wait  # type: ignore[method-assign]

    get_frame_ms: list[float] = []
    wait_ms: list[float] = []
    completion_ms: list[float] = []  # publish -> event complete, for frames not complete at entry
    done_at_entry = 0
    shape = dtype = None
    first_seen = False
    t_end = time.perf_counter() + args.seconds + 20.0  # +20 s for the sender to appear
    try:
        while time.perf_counter() < t_end:
            cur["wait"] = 0.0
            cur["done"] = None
            t0 = time.perf_counter()
            result = imp.get_frame()
            t1 = time.perf_counter()
            if result.outcome is ImportOutcome.NEW_FRAME:
                if not first_seen:  # warm-up frame: start the timed window here
                    first_seen = True
                    t_end = t1 + args.seconds
                    shape, dtype = tuple(result.frame.shape), str(result.frame.dtype)
                    continue
                get_frame_ms.append((t1 - t0) * 1e3)
                wait_ms.append(float(cur["wait"] or 0.0))
                if cur["done"]:
                    done_at_entry += 1
                else:
                    completion_ms.append(imp.last_latency + float(cur["wait"] or 0.0))
            elif result.outcome is ImportOutcome.NO_FRAME:
                if not imp.wait_for_doorbell(0.5):
                    time.sleep(0.0005)
            elif result.outcome is ImportOutcome.SHUTDOWN:
                break
            elif result.outcome is ImportOutcome.TIMEOUT:
                time.sleep(0.05)
        spin_hits, nap_hits = imp.wait_spin_hits, imp.wait_sleep_hits
        backend = imp.wait_backend_name
        spin_us = imp._policy.wait_spin_us
    finally:
        imp.close()

    frames = len(get_frame_ms)
    nap_pct = round(100.0 * nap_hits / max(1, spin_hits + nap_hits), 1)
    p50 = _pct(get_frame_ms, 0.5)
    red = frames == 0 or p50 is None or p50 > args.limit_ms or nap_pct > args.max_nap_pct
    print(
        json.dumps(
            {
                "shm": args.shm_name,
                "shape": shape,
                "dtype": dtype,
                "frames": frames,
                "wait_spin_us": spin_us,
                "wait_backend": backend,
                "get_frame_ms_p50_p95": [p50, _pct(get_frame_ms, 0.95)],
                "event_wait_ms_p50_p95": [_pct(wait_ms, 0.5), _pct(wait_ms, 0.95)],
                "completion_after_publish_ms_p50_p95": [_pct(completion_ms, 0.5), _pct(completion_ms, 0.95)],
                "event_done_at_entry_pct": round(100.0 * done_at_entry / frames, 1) if frames else None,
                "spin_hits": spin_hits,
                "nap_hits": nap_hits,
                "nap_pct": nap_pct,
                "verdict": "RED" if red else "GREEN",
            }
        ),
        flush=True,
    )
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
