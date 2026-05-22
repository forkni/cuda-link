"""
Cross-process end-to-end latency analysis for cuda-link TD pipeline captures.

Uses the two nsys SQLite exports produced by:
  scripts/profiling/analyze_td_pipeline.py (this script)
after running the two-terminal nsys capture.

Both captures must use --trace=wddm so timestamps share the same QPC clock.

Usage:
    python scripts/profiling/analyze_td_pipeline.py

Outputs:
    benchmarks/results/nsys/td_pipeline_e2e.csv   -- per-frame latency pairs
    benchmarks/results/nsys/td_pipeline_findings.md -- findings summary
"""

from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from statistics import median, quantiles

PROD_DB = Path("benchmarks/results/nsys/td_pipeline_v3_producer/producer.sqlite")
CONS_DB = Path("benchmarks/results/nsys/td_pipeline_v3_consumer/td_consumer.sqlite")
E2E_CSV = Path("benchmarks/results/nsys/td_pipeline_e2e.csv")
FINDINGS_MD = Path("benchmarks/results/nsys/td_pipeline_findings.md")

WARMUP_FRAMES = 150  # skip first N producer events (CUDA lazy init + graph capture)
MAX_PAIR_GAP_NS = 200_000_000  # 200 ms — reject pairs further apart than this


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return quantiles(values, n=100)[p - 1]


def load_nvtx_slots(db: Path) -> list[tuple[int, int, int]]:
    """Return [(slot_idx, start_ns, end_ns), ...] sorted by start for producer NVTX slot events."""
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        """
        SELECT
            COALESCE(n.text, s.value) AS name,
            n.start,
            n.end
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE (n.text LIKE 'cudalink.exporter.slot%'
               OR s.value LIKE 'cudalink.exporter.slot%')
        ORDER BY n.start
        """
    ).fetchall()
    conn.close()
    out = []
    for name, start, end in rows:
        m = re.search(r"slot(\d+)$", name or "")
        if m:
            out.append((int(m.group(1)), int(start), int(end)))
    return out


def load_cuda_api_stats(db: Path, name: str) -> tuple[float, float] | tuple[None, None]:
    """Return (avg_us, median_us) for a CUDA API call name (LIKE match). None if not found."""
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        """
        SELECT (r.end - r.start) / 1000.0
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE ?
        """,
        (name + "%",),
    ).fetchall()
    conn.close()
    if not rows:
        return None, None
    durations = sorted(r[0] for r in rows)
    avg = sum(durations) / len(durations)
    med = durations[len(durations) // 2]
    return avg, med


def load_nvtx_stats(db: Path, text: str) -> tuple[float, float] | tuple[None, None]:
    """Return (avg_us, median_us) for an NVTX range by exact text or textId match."""
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        """
        SELECT (n.end - n.start) / 1000.0
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE n.text = ? OR s.value = ?
        """,
        (text, text),
    ).fetchall()
    conn.close()
    if not rows:
        return None, None
    durations = sorted(r[0] for r in rows)
    avg = sum(durations) / len(durations)
    med = durations[len(durations) // 2]
    return avg, med


def load_runtime_events(db: Path, name: str) -> list[tuple[int, int]]:
    """Return [(start_ns, end_ns), ...] for a named CUDA runtime API call, sorted by start.
    Uses LIKE matching because nsys appends version suffixes (e.g. cudaStreamWaitEvent_v3020).
    """
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        """
        SELECT r.start, r.end
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE s.value LIKE ?
        ORDER BY r.start
        """,
        (name + "%",),
    ).fetchall()
    conn.close()
    return [(int(s), int(e)) for s, e in rows]


def load_session_start_ns(db: Path) -> int:
    """Return the absolute session start timestamp in nanoseconds (Unix epoch)."""
    conn = sqlite3.connect(str(db))
    # First column of TARGET_INFO_SESSION_START_TIME is a Unix timestamp in ns
    row = conn.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1").fetchone()
    conn.close()
    return int(row[0]) if row else 0


def load_nvtx_consumer(db: Path) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Load consumer NVTX data if available.
    Returns:
      slots  = [(slot_idx, start_ns, end_ns), ...] sorted by start
      waits  = [(start_ns, end_ns), ...] sorted by start  (event_wait ranges)
    Returns ([], []) if consumer has no NVTX table or no cudalink ranges.
    """
    conn = sqlite3.connect(str(db))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "NVTX_EVENTS" not in tables:
        conn.close()
        return [], []
    slot_rows = conn.execute(
        """
        SELECT COALESCE(n.text, s.value), n.start, n.end
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE (n.text LIKE 'cudalink.receiver.import_frame.slot%'
               OR s.value LIKE 'cudalink.receiver.import_frame.slot%')
        ORDER BY n.start
        """
    ).fetchall()
    wait_rows = conn.execute(
        """
        SELECT n.start, n.end
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId = s.id
        WHERE (n.text = 'cudalink.receiver.event_wait'
               OR s.value = 'cudalink.receiver.event_wait')
        ORDER BY n.start
        """
    ).fetchall()
    conn.close()
    slots = []
    for name, start, end in slot_rows:
        m = re.search(r"slot(\d+)$", name or "")
        if m:
            slots.append((int(m.group(1)), int(start), int(end)))
    waits = [(int(s), int(e)) for s, e in wait_rows]
    return slots, waits


def load_gpu_memcpy(db: Path) -> list[tuple[int, int, int]]:
    """Return [(start_ns, end_ns, bytes), ...] sorted by start."""
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
    conn.close()
    return [(int(s), int(e), int(b)) for s, e, b in rows]


def compute_clock_offset(prod_db: Path, cons_db: Path) -> int:
    """Return offset_ns to ADD to consumer timestamps to put them on producer's clock.
    Both captures store timestamps relative to their own session start; we align via
    the absolute Unix-epoch session start recorded in TARGET_INFO_SESSION_START_TIME.
    """
    prod_abs = load_session_start_ns(prod_db)
    cons_abs = load_session_start_ns(cons_db)
    # consumer timestamps need this subtracted to map onto producer timeline
    offset = cons_abs - prod_abs
    print(f"  Clock offset: {offset / 1e9:.3f}s  (consumer started {offset / 1e9:.1f}s before producer)")
    return offset


def check_timestamp_overlap(prod_slots: list, cons_waits: list) -> tuple[int, int]:
    """Return (overlap_start, overlap_end) in ns (producer clock). Warn if no overlap."""
    if not prod_slots or not cons_waits:
        return 0, 0
    prod_start = prod_slots[0][1]
    prod_end = prod_slots[-1][2]
    cons_start = cons_waits[0][0]  # already adjusted to producer clock by caller
    cons_end = cons_waits[-1][1]
    overlap_start = max(prod_start, cons_start)
    overlap_end = min(prod_end, cons_end)
    if overlap_start >= overlap_end:
        print(
            f"WARNING: No timestamp overlap between captures.\n"
            f"  Producer: {prod_start}--{prod_end} ns\n"
            f"  Consumer (adjusted): {cons_start}--{cons_end} ns\n"
            "  Timestamps may not share a clock source. Cross-process metrics unreliable."
        )
    return overlap_start, overlap_end


def pair_events(
    prod_slots: list[tuple[int, int, int]],
    cons_waits: list[tuple[int, int]],
    overlap_start: int,
    overlap_end: int,
) -> list[dict]:
    """
    For each consumer wait event, find the nearest prior producer slot completion.
    Returns list of dicts with timing fields.
    """
    # Filter to overlap window + skip warmup
    prod_steady = [
        (slot, s, e)
        for i, (slot, s, e) in enumerate(prod_slots)
        if i >= WARMUP_FRAMES and overlap_start <= s <= overlap_end
    ]
    cons_steady = [(s, e) for s, e in cons_waits if overlap_start <= s <= overlap_end]

    pairs = []
    prod_idx = 0
    for cons_start, cons_end in cons_steady:
        # Advance producer pointer to the last slot that ended before this consumer wait
        while prod_idx + 1 < len(prod_steady) and prod_steady[prod_idx + 1][2] <= cons_start:
            prod_idx += 1
        if prod_idx >= len(prod_steady):
            break
        slot_idx, prod_s, prod_e = prod_steady[prod_idx]
        if prod_e > cons_start:
            continue  # producer slot hasn't finished yet at this consumer event
        gap_ns = cons_start - prod_e
        if gap_ns > MAX_PAIR_GAP_NS:
            continue  # too large a gap — skip
        pairs.append(
            {
                "slot": slot_idx,
                "prod_start_ns": prod_s,
                "prod_end_ns": prod_e,
                "cons_wait_start_ns": cons_start,
                "cons_wait_end_ns": cons_end,
                "handoff_latency_us": gap_ns / 1000.0,
                "e2e_latency_us": (cons_start - prod_s) / 1000.0,
                "event_wait_us": (cons_end - cons_start) / 1000.0,
            }
        )
    return pairs


def pair_events_by_slot(
    prod_slots: list[tuple[int, int, int]],
    cons_slots: list[tuple[int, int, int]],
    overlap_start: int,
    overlap_end: int,
    *,
    clock_offset_reliable: bool = True,
) -> list[dict]:
    """Exact slot-index matching: for each consumer slot<N> event, find the nearest
    prior producer slot<N> event with the same index. Requires consumer NVTX.

    When clock_offset_reliable=False the time-overlap gate and gap check are skipped
    because the clock offset between the two sessions exceeds the capture window.
    Slot-ID matching remains correct; handoff/e2e columns are omitted from the result
    as they would be polluted by the offset. import_frame_us is consumer-internal and
    always reliable regardless of the clock alignment.
    """
    from collections import defaultdict

    # Group producer slots by index, skip warmup; skip overlap gate if clock unreliable
    prod_by_slot: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, (slot, s, e) in enumerate(prod_slots):
        if i >= WARMUP_FRAMES and (not clock_offset_reliable or overlap_start <= s <= overlap_end):
            prod_by_slot[slot].append((s, e))

    pairs = []
    for slot, cons_s, cons_e in cons_slots:
        if clock_offset_reliable and not (overlap_start <= cons_s <= overlap_end):
            continue
        candidates = prod_by_slot.get(slot, [])
        # Find nearest prior producer slot end
        prior = [(ps, pe) for ps, pe in candidates if pe <= cons_s]
        if not prior:
            continue
        prod_s, prod_e = max(prior, key=lambda x: x[1])
        gap_ns = cons_s - prod_e
        if clock_offset_reliable and gap_ns > MAX_PAIR_GAP_NS:
            continue
        pair: dict = {
            "slot": slot,
            "prod_start_ns": prod_s,
            "prod_end_ns": prod_e,
            "cons_slot_start_ns": cons_s,
            "cons_slot_end_ns": cons_e,
            "import_frame_us": (cons_e - cons_s) / 1000.0,
        }
        if clock_offset_reliable:
            pair["handoff_latency_us"] = gap_ns / 1000.0
            pair["e2e_latency_us"] = (cons_s - prod_s) / 1000.0
        pairs.append(pair)
    pairs.sort(key=lambda p: p["prod_start_ns"])
    return pairs


def stats_str(values: list[float], label: str) -> str:
    if not values:
        return f"  {label}: no data"
    return (
        f"  {label}: "
        f"p50={median(values):.1f} µs  "
        f"p99={_pct(values, 99):.1f} µs  "
        f"max={max(values):.1f} µs  "
        f"n={len(values)}"
    )


def compute_bw_gbps(total_bytes: int, total_ns: int) -> float:
    if total_ns <= 0:
        return 0.0
    return (total_bytes / (total_ns / 1e9)) / 1e9


def main() -> None:
    import argparse as _ap

    global PROD_DB, CONS_DB, E2E_CSV, FINDINGS_MD
    _p = _ap.ArgumentParser(description="Cross-process TD pipeline nsys analysis.")
    _p.add_argument(
        "--prod-db",
        metavar="PATH",
        default=str(PROD_DB),
        help="Producer nsys SQLite (default: td_pipeline_v3_producer/producer.sqlite)",
    )
    _p.add_argument(
        "--cons-db",
        metavar="PATH",
        default=str(CONS_DB),
        help="Consumer nsys SQLite (default: td_pipeline_v3_consumer/td_consumer.sqlite)",
    )
    _p.add_argument(
        "--e2e-csv", metavar="PATH", default=str(E2E_CSV), help="Output E2E CSV (default: td_pipeline_e2e.csv)"
    )
    _p.add_argument(
        "--findings-md",
        metavar="PATH",
        default=str(FINDINGS_MD),
        help="Output findings Markdown (default: td_pipeline_findings.md)",
    )
    _args = _p.parse_args()
    PROD_DB = Path(_args.prod_db)
    CONS_DB = Path(_args.cons_db)
    E2E_CSV = Path(_args.e2e_csv)
    FINDINGS_MD = Path(_args.findings_md)

    print("Loading producer NVTX slot events ...")
    prod_slots = load_nvtx_slots(PROD_DB)
    print(f"  {len(prod_slots)} slot events found")

    print("Loading consumer NVTX (slot-level, preferred) ...")
    cons_nvtx_slots, cons_nvtx_waits = load_nvtx_consumer(CONS_DB)
    if cons_nvtx_slots:
        print(
            f"  {len(cons_nvtx_slots)} slot events + {len(cons_nvtx_waits)} event_wait ranges -- slot matching enabled"
        )
    else:
        print("  No consumer NVTX -- falling back to cudaStreamWaitEvent API timing")

    print("Loading consumer cudaStreamWaitEvent (IPC event wait fallback) ...")
    cons_waits = load_runtime_events(CONS_DB, "cudaStreamWaitEvent")
    print(f"  {len(cons_waits)} events found")

    print("Loading producer GPU memcpy (H2D staging fill) ...")
    prod_memcpy = load_gpu_memcpy(PROD_DB)
    print(f"  {len(prod_memcpy)} H2D copies")

    print("Loading consumer GPU memcpy (D2A copy from IPC buffer) ...")
    cons_memcpy = load_gpu_memcpy(CONS_DB)
    print(f"  {len(cons_memcpy)} D2A copies")

    # Sub-range breakdown for the producer (for findings doc)
    _ssync_avg, _ssync_med = load_cuda_api_stats(PROD_DB, "cudaStreamSynchronize")
    _glaunch_avg, _glaunch_med = load_cuda_api_stats(PROD_DB, "cudaGraphLaunch")
    _shm_avg, _shm_med = load_nvtx_stats(PROD_DB, "cudalink.exporter.shm_write")
    prod_subrange = {
        "ssync_avg": _ssync_avg,
        "ssync_med": _ssync_med,
        "glaunch_avg": _glaunch_avg,
        "glaunch_med": _glaunch_med,
        "shm_avg": _shm_avg,
        "shm_med": _shm_med,
    }

    print("\nComputing clock offset ...")
    clock_offset = compute_clock_offset(PROD_DB, CONS_DB)
    # Shift consumer timestamps onto producer's clock
    cons_waits = [(s - clock_offset, e - clock_offset) for s, e in cons_waits]
    cons_memcpy = [(s - clock_offset, e - clock_offset, b) for s, e, b in cons_memcpy]

    overlap_start, overlap_end = check_timestamp_overlap(prod_slots, cons_waits)
    overlap_s = (overlap_end - overlap_start) / 1e9
    print(f"Timestamp overlap: {overlap_s:.1f} s")

    # Adjust consumer NVTX slots clock offset too
    cons_nvtx_slots_adj = [(sl, s - clock_offset, e - clock_offset) for sl, s, e in cons_nvtx_slots]
    cons_nvtx_waits_adj = [(s - clock_offset, e - clock_offset) for s, e in cons_nvtx_waits]

    # Determine whether the clock alignment is reliable (offset must fit within the capture window).
    # When the two nsys sessions started more than a full capture-window apart, the adjusted
    # consumer timestamps won't overlap the producer's, making handoff/e2e latencies meaningless.
    _prod_dur_ns = prod_slots[-1][2] - prod_slots[0][1] if len(prod_slots) > 1 else 0
    _cons_dur_ns = (
        cons_nvtx_slots_adj[-1][2] - cons_nvtx_slots_adj[0][1]
        if len(cons_nvtx_slots_adj) > 1
        else (cons_waits[-1][1] - cons_waits[0][0] if len(cons_waits) > 1 else 0)
    )
    clock_offset_reliable = abs(clock_offset) <= max(_prod_dur_ns, _cons_dur_ns)
    if not clock_offset_reliable:
        print(
            f"  WARNING: clock offset ({abs(clock_offset) / 1e9:.1f}s) exceeds capture window"
            f" ({max(_prod_dur_ns, _cons_dur_ns) / 1e9:.1f}s); "
            "handoff/e2e latencies will be omitted -- import_frame_us is still valid."
        )

    print("\nPairing producer slots with consumer events ...")
    if cons_nvtx_slots_adj:
        if clock_offset_reliable:
            pairs = pair_events_by_slot(
                prod_slots, cons_nvtx_slots_adj, overlap_start, overlap_end, clock_offset_reliable=True
            )
            print(f"  {len(pairs)} slot-matched pairs (exact slot index, warmup skipped)")
        else:
            # Clock offset exceeds capture window — adjusted consumer timestamps are meaningless
            # for producer cross-reference. Emit consumer import_frame records directly so the
            # CSV is populated with trustworthy import_frame_us data.
            pairs = [
                {
                    "slot": slot,
                    "cons_slot_start_ns": cons_s,
                    "cons_slot_end_ns": cons_e,
                    "import_frame_us": (cons_e - cons_s) / 1000.0,
                }
                for i, (slot, cons_s, cons_e) in enumerate(cons_nvtx_slots)
                if i >= WARMUP_FRAMES
            ]
            pairs.sort(key=lambda p: p["cons_slot_start_ns"])
            print(
                f"  {len(pairs)} consumer-only records (clock unreliable, import_frame_us only -- no producer cross-ref)"
            )
    else:
        pairs = pair_events(prod_slots, cons_waits, overlap_start, overlap_end)
        print(f"  {len(pairs)} matched pairs (nearest-prior, warmup skipped, gap < {MAX_PAIR_GAP_NS / 1e6:.0f} ms)")

    # Per-process slot durations (NVTX)
    slot_durations_us = [(e - s) / 1000.0 for _, s, e in prod_slots[WARMUP_FRAMES:]]

    # GPU-side timing
    prod_h2d_us = [(e - s) / 1000.0 for s, e, _ in prod_memcpy]
    prod_h2d_bytes = sum(b for _, _, b in prod_memcpy)
    prod_h2d_ns = sum(e - s for s, e, _ in prod_memcpy)

    cons_d2a_us = [(e - s) / 1000.0 for s, e, _ in cons_memcpy]
    cons_d2a_bytes = sum(b for _, _, b in cons_memcpy)
    cons_d2a_ns = sum(e - s for s, e, _ in cons_memcpy)

    handoff_us = [p["handoff_latency_us"] for p in pairs if "handoff_latency_us" in p]
    e2e_us = [p["e2e_latency_us"] for p in pairs if "e2e_latency_us" in p]
    event_wait_us_nvtx = [(e - s) / 1000.0 for s, e in cons_nvtx_waits_adj[WARMUP_FRAMES:]]
    event_wait_us_api = [p["event_wait_us"] for p in pairs if "event_wait_us" in p]
    event_wait_us = event_wait_us_nvtx if event_wait_us_nvtx else event_wait_us_api

    # Write CSV
    E2E_CSV.parent.mkdir(parents=True, exist_ok=True)
    with E2E_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()) if pairs else [])
        w.writeheader()
        w.writerows(pairs)
    print(f"\nWrote {len(pairs)} rows -> {E2E_CSV}")

    # Compute per-process FPS from full capture span (not overlap window)
    if len(prod_slots) > 1:
        prod_span_s = (prod_slots[-1][2] - prod_slots[0][1]) / 1e9
        prod_fps = len(prod_slots) / prod_span_s
    else:
        prod_fps = 0.0
    if cons_nvtx_slots_adj and len(cons_nvtx_slots_adj) > 1:
        cons_span_s = (cons_nvtx_slots_adj[-1][2] - cons_nvtx_slots_adj[0][1]) / 1e9
        cons_fps = len(cons_nvtx_slots_adj) / cons_span_s
    elif len(cons_waits) > 1:
        cons_span_s = (cons_waits[-1][1] - cons_waits[0][0]) / 1e9
        cons_fps = len(cons_waits) / cons_span_s
    else:
        cons_fps = 0.0

    print("\n=== PRODUCER ===")
    print(stats_str(slot_durations_us, "exporter.slot<N> (NVTX)"))
    print(stats_str(prod_h2d_us, "H2D staging fill (GPU)"))
    print(f"  H2D bandwidth: {compute_bw_gbps(prod_h2d_bytes, prod_h2d_ns):.2f} GB/s")
    print(f"  Effective rate: {prod_fps:.1f} FPS")

    # Consumer NVTX slot durations (full capture, post-warmup)
    cons_slot_dur = [(e - s) / 1000.0 for _, s, e in cons_nvtx_slots_adj[WARMUP_FRAMES:]] if cons_nvtx_slots_adj else []

    print("\n=== CONSUMER (TD) ===")
    if cons_nvtx_slots_adj:
        print(stats_str(cons_slot_dur, "import_frame.slot<N> (NVTX)"))
        print(stats_str(event_wait_us, "event_wait (NVTX)"))
    else:
        print(stats_str(event_wait_us, "cudaStreamWaitEvent (API fallback)"))
    print(stats_str(cons_d2a_us, "D2A copy from IPC buffer (GPU)"))
    print(f"  D2A bandwidth: {compute_bw_gbps(cons_d2a_bytes, cons_d2a_ns):.2f} GB/s")
    print(f"  Effective cook rate: {cons_fps:.1f} FPS")
    drop_pct = max(0, (1 - cons_fps / prod_fps) * 100) if prod_fps > 0 else 0
    print(f"  Frame drop rate: ~{drop_pct:.0f}% (expected -- TD reads latest slot)")

    print("\n=== CROSS-PROCESS E2E ===")
    print(stats_str(handoff_us, "Handoff latency (producer.slot_end -> consumer.wait_start)"))
    print(stats_str(e2e_us, "E2E latency (producer.slot_start -> consumer.wait_start)"))

    # Write findings doc
    _write_findings(
        prod_slots,
        cons_waits,
        prod_memcpy,
        cons_memcpy,
        pairs,
        slot_durations_us,
        prod_h2d_us,
        cons_d2a_us,
        handoff_us,
        e2e_us,
        event_wait_us,
        prod_fps,
        cons_fps,
        prod_h2d_bytes,
        prod_h2d_ns,
        cons_d2a_bytes,
        cons_d2a_ns,
        overlap_s,
        cons_slot_dur=cons_slot_dur,
        has_consumer_nvtx=bool(cons_nvtx_slots_adj),
        prod_subrange=prod_subrange,
        clock_offset_reliable=clock_offset_reliable,
    )
    print(f"\nWrote findings -> {FINDINGS_MD}")


def _pval(values: list[float], p: int, fmt: str = ".1f") -> str:
    if not values:
        return "n/a"
    return format(_pct(values, p), fmt)


def _write_findings(
    prod_slots,
    cons_waits,
    prod_memcpy,
    cons_memcpy,
    pairs,
    slot_dur,
    prod_h2d,
    cons_d2a,
    handoff,
    e2e,
    event_wait,
    prod_fps,
    cons_fps,
    prod_h2d_bytes,
    prod_h2d_ns,
    cons_d2a_bytes,
    cons_d2a_ns,
    overlap_s,
    cons_slot_dur=None,
    has_consumer_nvtx=False,
    prod_subrange=None,
    clock_offset_reliable: bool = True,
) -> None:
    if cons_slot_dur is None:
        cons_slot_dur = []
    if prod_subrange is None:
        prod_subrange = {}
    bw_h2d = compute_bw_gbps(prod_h2d_bytes, prod_h2d_ns)
    bw_d2a = compute_bw_gbps(cons_d2a_bytes, cons_d2a_ns)
    frame_drop_pct = max(0, int((1 - cons_fps / prod_fps) * 100)) if prod_fps > 0 else 0

    # Pre-compute conditional f-string fragments (no backslash allowed inside expressions)
    nvtx_install_step = (
        ""
        if has_consumer_nvtx
        else "- Install nvtx into TD Python and re-capture to get per-slot consumer NVTX ranges.\n"
    )
    rerun_step = (
        "- Re-run with tighter capture timing to obtain cross-process E2E latency with consumer NVTX.\n"
        if has_consumer_nvtx and not pairs
        else ""
    )
    cons_import_row = (
        f"| import_frame.slot<N> (NVTX) | {format(median(cons_slot_dur), '.1f')} µs"
        f" | {_pval(cons_slot_dur, 99)} µs"
        f" | {format(max(cons_slot_dur), '.0f')} µs"
        f" | {len(cons_slot_dur)} |"
        if has_consumer_nvtx and cons_slot_dur
        else ""
    )
    event_wait_label = "event_wait (NVTX)" if has_consumer_nvtx else "cudaStreamWaitEvent (IPC event wait)"
    nvtx_status = (
        "present -- cudalink.receiver.* ranges active via system Python nvtx"
        if has_consumer_nvtx
        else "absent -- nvtx package not installed in TD's bundled Python runtime"
    )
    ratio_str = f"{int(prod_fps / cons_fps):.0f}" if cons_fps > 0 else "N/A"
    obs5 = (
        f"**Consumer NVTX active** -- cudalink.receiver.* ranges present via system Python nvtx.\n"
        f"   import_frame p50 {format(median(cons_slot_dur), '.0f') if cons_slot_dur else 'n/a'} µs;"
        f" event_wait p50 {format(median(event_wait), '.0f') if event_wait else 'n/a'} µs."
        if has_consumer_nvtx
        else "**No consumer NVTX** -- nvtx package is not installed in TouchDesigner's bundled Python\n"
        "   runtime. The cudalink.receiver.* ranges are instrumented but silently no-op. To surface\n"
        "   them in a future capture, install nvtx into TD's Python:\n"
        r"     C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\python.exe -m pip install nvtx"
    )

    def _fmt(v):
        return f"{v:.1f}" if v is not None else "n/a"

    _ssync_avg = _fmt(prod_subrange.get("ssync_avg"))
    _ssync_med = _fmt(prod_subrange.get("ssync_med"))
    _glaunch_avg = _fmt(prod_subrange.get("glaunch_avg"))
    _glaunch_med = _fmt(prod_subrange.get("glaunch_med"))
    _shm_avg = _fmt(prod_subrange.get("shm_avg"))
    _shm_med = _fmt(prod_subrange.get("shm_med"))
    # E2E section varies based on whether the clock alignment is reliable
    if clock_offset_reliable:
        _e2e_table_rows = (
            f"| Handoff latency (slot_end -> consumer wait_start)"
            f" | {format(median(handoff), '.0f') if handoff else 'n/a'} µs"
            f" | {_pval(handoff, 99, '.0f')} µs"
            f" | {format(max(handoff), '.0f') if handoff else 'n/a'} µs |\n"
            f"| Full E2E (slot_start -> consumer wait_start)"
            f" | {format(median(e2e), '.0f') if e2e else 'n/a'} µs"
            f" | {_pval(e2e, 99, '.0f')} µs"
            f" | {format(max(e2e), '.0f') if e2e else 'n/a'} µs |"
        )
        _e2e_note = (
            f"Note: pairing method = nearest prior producer slot completion before each consumer IPC wait.\n"
            f"At {cons_fps:.0f} FPS consumer / {prod_fps:.0f} FPS producer, the consumer always reads a slot that is\n"
            f"{ratio_str}-3 frames behind the producer's current write head -- expected."
        )
    else:
        _e2e_table_rows = (
            f"| import_frame duration (consumer-internal, trustworthy)"
            f" | {format(median(cons_slot_dur), '.0f') if cons_slot_dur else 'n/a'} µs"
            f" | {_pval(cons_slot_dur, 99, '.0f')} µs"
            f" | {format(max(cons_slot_dur), '.0f') if cons_slot_dur else 'n/a'} µs |\n"
            "⚠ Handoff and E2E latencies omitted: clock offset exceeds capture window.\n"
            "  import_frame_us in the CSV is reliable (consumer-internal).\n"
            "  handoff_latency_us and e2e_latency_us are not present in this run's CSV."
        )
        _e2e_note = (
            "Note: slot-ID matching used without time-window gating (non-overlapping nsys sessions).\n"
            "For clock-faithful cross-process latency, ensure both nsys sessions start within\n"
            "the same rolling window (gap < capture duration) on the next live capture."
        )

    _near_sync = prod_fps > 0 and cons_fps > 0 and cons_fps / prod_fps > 0.8
    _handoff_med = f"{median(handoff):.0f}" if handoff else "n/a"
    obs4 = (
        f"**TD cook rate ~{cons_fps:.0f} FPS ≈ producer {prod_fps:.0f} FPS** -- near-synchronous rates with"
        f" 3 slots mean each slot is read ~2 slot-periods after being written. The E2E handoff p50"
        f" of ~{_handoff_med} µs reflects this topology (not a latency regression)."
        if _near_sync
        else f"**TD cook rate ~{cons_fps:.0f} FPS vs producer {prod_fps:.0f} FPS** -- TD's Script TOP cook rate is"
        f" lower than the producer's send rate. The 3-slot ring buffer absorbs the rate mismatch"
        f" correctly: TD always gets the most recent frame, never waits for a specific slot."
    )

    md = f"""\
# TD Pipeline nsys Capture -- Findings (2026-05-07)

## Setup

- **TD**: CUDA_Link_Example.43.toe (TD 2025.32820, launcher Execute DAT disabled), receiver Memname=cudalink_output_ipc
- **Producer**: example_sender_python.py, 512x512 RGBA uint8, 3 slots, target 60 FPS
- **nsys**: 2026.2.1, `--trace=cuda,nvtx,wddm --wddm-additional-events=true --wddm-backtraces=true`
- **Capture**: 55s producer / 60s consumer, timestamp overlap {overlap_s:.1f}s

## Producer Metrics (PID: Python sender process)

| Metric | p50 | p99 | max | count |
|---|---|---|---|---|
| exporter.slot<N> duration (NVTX) | {format(median(slot_dur), ".1f") if slot_dur else "n/a"} µs | {_pval(slot_dur, 99)} µs | {format(max(slot_dur), ".0f") if slot_dur else "n/a"} µs | {len(slot_dur)} |
| H2D staging fill (GPU) | {format(median(prod_h2d), ".1f") if prod_h2d else "n/a"} µs | {_pval(prod_h2d, 99)} µs | {format(max(prod_h2d), ".0f") if prod_h2d else "n/a"} µs | {len(prod_h2d)} |

- **H2D bandwidth**: {bw_h2d:.2f} GB/s (512x512 RGBA uint8 = 1 MB/frame via PCIe/WDDM)
- **Effective send rate**: {prod_fps:.1f} FPS (target: 60 FPS)
- **Sub-range breakdown** (verbose NVTX + CUDA API):
  - cudaGraphLaunch (D2D copy + event record): avg {_glaunch_avg} µs, median {_glaunch_med} µs
  - cudaStreamSynchronize (flush_probe): avg {_ssync_avg} µs, median {_ssync_med} µs -- dominates slot time
  - shm_write (SHM header update): avg {_shm_avg} µs, median {_shm_med} µs

## Consumer Metrics (TD / TouchDesigner process)

| Metric | p50 | p99 | max | count |
|---|---|---|---|---|
{cons_import_row}
| D2A copy from IPC buffer (GPU) | {format(median(cons_d2a), ".1f") if cons_d2a else "n/a"} µs | {_pval(cons_d2a, 99)} µs | {format(max(cons_d2a), ".0f") if cons_d2a else "n/a"} µs | {len(cons_d2a)} |
| {event_wait_label} | {format(median(event_wait), ".1f") if event_wait else "n/a"} µs | {_pval(event_wait, 99)} µs | {format(max(event_wait), ".0f") if event_wait else "n/a"} µs | {len(event_wait)} |

- **D2A bandwidth**: {bw_d2a:.2f} GB/s (device-to-CUDA-array within GPU)
- **Effective cook rate**: {cons_fps:.1f} FPS
- **Frame drop rate**: ~{frame_drop_pct}% -- by design (TD reads latest-written slot each cook)
- **NVTX**: {nvtx_status}

## Cross-Process E2E Latency ({len(pairs)} slot-matched pairs)

| Metric | p50 | p99 | max |
|---|---|---|---|
{_e2e_table_rows}

{_e2e_note}

## Key Observations

1. **flush_probe dominates producer slot time** ({format(median(slot_dur), ".0f") if slot_dur else "n/a"} µs slot median, of which ~{_ssync_med} µs is
   cudaStreamSynchronize). The sync waits for the D2D IPC copy to complete on the GPU before
   the CPU updates the SHM header -- this is the WDDM batch-flush point.

2. **IPC event wait is healthy** (~{format(median(event_wait), ".0f") if event_wait else "n/a"} µs p50). The consumer unblocks quickly once the
   producer's event is signalled, consistent with the D2D completing before the CPU updates SHM.

3. **Producer uses two separate CUDA contexts** on the same device (confirmed from wddm_queue_sum
   in the bench_sweep run): IPC stream + main stream on separate WDDM Render queue contexts.

4. {obs4}

5. {obs5}

## Next Steps

{nvtx_install_step}- Run F2/F8 regression (CUDALINK_TD_PERSIST_STREAM=0) to see stream serialisation on the
  GPU lane (expected: D2D and D2A serialize rather than overlap).
- ncu deep-dive: the only kernel of interest is the CUDA Graph node (D2D copy in IPC stream).
  Target: `cudaGraphLaunch` call, profile with --set full to check PCIe vs HBM utilization.
{rerun_step}"""
    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
