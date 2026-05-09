"""v5b slot0 import_frame outlier diagnostic loop.

Mines the v5 consumer nsys SQLite to classify the slot0 30 ms outlier.
Distinguishes:
  H5 -- cudaMemcpy2DToArrayAsync WDDM command-buffer stall (D2A CPU-side block)
  H2 -- OS preemption / SHM poll wait (bare wall-clock gap, no CUDA API)
  H1 -- producer write-bias (slot0 gets more writes, more chances to hit stalls)

Usage:
    python scripts/profiling/v5b_slot0_outlier_mine.py [path-to-sqlite]

Defaults to:
    benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer.sqlite
"""

import sqlite3
import statistics
import sys
from pathlib import Path

DB_DEFAULT = Path("benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer.sqlite")
OUTLIER_THRESHOLD_US = 2_000  # µs — classify anything above this as an outlier

TAG = "[DEBUG-slot0]"  # grep for this to audit/remove debug prints


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        sys.exit(f"[ERROR] SQLite not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def get_cudalink_text_ids(con: sqlite3.Connection) -> dict[str, int]:
    cur = con.cursor()
    cur.execute("SELECT id, value FROM StringIds WHERE value LIKE '%cudalink%'")
    return {row["value"]: row["id"] for row in cur.fetchall()}


def get_consumer_thread(con: sqlite3.Connection, slot0_id: int) -> int:
    cur = con.cursor()
    cur.execute("SELECT globalTid FROM NVTX_EVENTS WHERE textId=? LIMIT 1", (slot0_id,))
    row = cur.fetchone()
    if row is None:
        sys.exit("[ERROR] No slot0 import_frame events found — check textId mapping")
    return row["globalTid"]


def import_frame_stats(con: sqlite3.Connection, text_ids: dict[str, int]) -> dict[str, dict]:
    """Return per-slot import_frame duration stats."""
    cur = con.cursor()
    slots = {
        "slot0": text_ids.get(":cudalink.receiver.import_frame.slot0"),
        "slot1": text_ids.get(":cudalink.receiver.import_frame.slot1"),
        "slot2": text_ids.get(":cudalink.receiver.import_frame.slot2"),
    }
    result = {}
    for name, tid_k in slots.items():
        if tid_k is None:
            # Try without the leading colon
            for k, v in text_ids.items():
                if f"import_frame.{name[-5:]}" in k:
                    tid_k = v
                    break
        if tid_k is None:
            result[name] = {"n": 0}
            continue
        cur.execute(
            "SELECT end-start AS dur FROM NVTX_EVENTS WHERE textId=? ORDER BY dur",
            (tid_k,),
        )
        durs = [r["dur"] for r in cur.fetchall()]
        if not durs:
            result[name] = {"n": 0}
            continue
        result[name] = {
            "n": len(durs),
            "p50_us": statistics.median(durs) / 1e3,
            "p99_us": durs[int(len(durs) * 0.99)] / 1e3,
            "max_us": durs[-1] / 1e3,
            "outlier_n": sum(1 for d in durs if d > OUTLIER_THRESHOLD_US * 1e3),
            "textId": tid_k,
        }
    return result


def d2a_stats_per_slot(
    con: sqlite3.Connection,
    text_ids: dict[str, int],
    consumer_tid: int,
) -> dict[str, dict]:
    """Per-slot D2A (cudaMemcpy2DToArrayAsync) duration stats, correlated via NVTX."""
    cur = con.cursor()
    # Map slot name → NVTX textId
    slot_map = {}
    for k, v in text_ids.items():
        for slot in ["slot0", "slot1", "slot2"]:
            if f"import_frame.{slot}" in k:
                slot_map[slot] = v

    cur.execute("SELECT id FROM StringIds WHERE value='cudaMemcpy2DToArrayAsync_v3020'")
    d2a_row = cur.fetchone()
    if d2a_row is None:
        return {}
    d2a_name_id = d2a_row["id"]

    result = {}
    for slot, tid_k in slot_map.items():
        cur.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE textId=? ORDER BY start",
            (tid_k,),
        )
        frames = cur.fetchall()
        durs = []
        for frame in frames:
            cur.execute(
                """
                SELECT end-start AS dur FROM CUPTI_ACTIVITY_KIND_RUNTIME
                WHERE globalTid=? AND nameId=? AND start>=? AND start<=?
                LIMIT 1
                """,
                (consumer_tid, d2a_name_id, frame["start"], frame["end"]),
            )
            r = cur.fetchone()
            if r:
                durs.append(r["dur"])
        if not durs:
            result[slot] = {"n": 0}
            continue
        durs.sort()
        result[slot] = {
            "n": len(durs),
            "p50_us": statistics.median(durs) / 1e3,
            "p99_us": durs[int(len(durs) * 0.99)] / 1e3,
            "max_us": durs[-1] / 1e3,
        }
    return result


def classify_outlier(
    con: sqlite3.Connection,
    outlier_start: int,
    outlier_end: int,
    consumer_tid: int,
    d2a_name_id: int,
    event_wait_id: int,
) -> dict:
    """Classify a single outlier event by inspecting its CUDA API profile."""
    cur = con.cursor()
    # Find the largest single CUDA API call inside the outlier window
    cur.execute(
        """
        SELECT r.start, r.end, r.end-r.start AS dur, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId=s.id
        WHERE r.globalTid=? AND r.start>=? AND r.start<=?
        ORDER BY dur DESC
        LIMIT 1
        """,
        (consumer_tid, outlier_start, outlier_end),
    )
    dominant = cur.fetchone()

    # Time from outlier start to first CUDA API call (SHM poll gap)
    cur.execute(
        """
        SELECT MIN(r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        WHERE r.globalTid=? AND r.start>=? AND r.start<=?
        """,
        (consumer_tid, outlier_start, outlier_end),
    )
    first_api_row = cur.fetchone()
    first_api_start = first_api_row[0] if first_api_row and first_api_row[0] else None
    shm_poll_gap_us = (first_api_start - outlier_start) / 1e3 if first_api_start else None

    total_dur_us = (outlier_end - outlier_start) / 1e3
    dominant_api = dominant["value"] if dominant else None
    dominant_dur_us = dominant["dur"] / 1e3 if dominant else 0.0
    dominant_frac = dominant_dur_us / total_dur_us if total_dur_us > 0 else 0.0

    # Classification
    if dominant and "cudaMemcpy2DToArrayAsync" in dominant_api and dominant_frac > 0.5:
        classification = "H5:D2A_WDDM_stall"
    elif shm_poll_gap_us and shm_poll_gap_us > 1_000:
        classification = "H2:SHM_poll_or_preemption"
    elif dominant_frac > 0.5:
        classification = f"H?:dominant={dominant_api}"
    else:
        classification = "H2:no_dominant_API_bare_gap"

    return {
        "dur_us": total_dur_us,
        "dominant_api": dominant_api,
        "dominant_dur_us": dominant_dur_us,
        "dominant_frac": dominant_frac,
        "shm_poll_gap_us": shm_poll_gap_us,
        "classification": classification,
    }


def run(db_path: Path) -> None:
    con = connect(db_path)
    text_ids = get_cudalink_text_ids(con)

    # Resolve slot0 textId (strip leading colon if present)
    slot0_id = None
    for k, v in text_ids.items():
        if "import_frame.slot0" in k:
            slot0_id = v
    if slot0_id is None:
        sys.exit("[ERROR] import_frame.slot0 label not found in StringIds")

    consumer_tid = get_consumer_thread(con, slot0_id)
    print(f"{TAG} Consumer thread globalTid: {consumer_tid}")

    # --- Per-slot import_frame stats ---
    stats = import_frame_stats(con, text_ids)
    print(f"\n{TAG} ===== import_frame duration per slot =====")
    for slot, s in stats.items():
        if s.get("n", 0) == 0:
            print(f"  {slot}: NO DATA")
        else:
            print(
                f"  {slot}: n={s['n']:5d}  p50={s['p50_us']:7.1f}µs  "
                f"p99={s['p99_us']:7.1f}µs  max={s['max_us']:9.1f}µs  "
                f"outliers(>{OUTLIER_THRESHOLD_US}µs)={s['outlier_n']}"
            )

    # --- D2A stats per slot ---
    cur = con.cursor()
    cur.execute("SELECT id FROM StringIds WHERE value='cudaMemcpy2DToArrayAsync_v3020'")
    d2a_id_row = cur.fetchone()
    event_wait_id = text_ids.get(":cudalink.receiver.event_wait") or next(
        (v for k, v in text_ids.items() if "event_wait" in k), None
    )

    d2a = d2a_stats_per_slot(con, text_ids, consumer_tid)
    print(f"\n{TAG} ===== cudaMemcpy2DToArrayAsync (D2A) per slot =====")
    for slot, s in d2a.items():
        if s.get("n", 0) == 0:
            print(f"  {slot}: NO DATA")
        else:
            print(
                f"  {slot}: n={s['n']:5d}  p50={s['p50_us']:6.1f}µs  p99={s['p99_us']:6.1f}µs  max={s['max_us']:9.1f}µs"
            )

    # --- Classify each outlier ---
    slot0_text_id = stats.get("slot0", {}).get("textId") or slot0_id
    d2a_name_id = d2a_id_row["id"] if d2a_id_row else None
    cur.execute(
        f"SELECT start, end, end-start AS dur FROM NVTX_EVENTS "
        f"WHERE textId=? AND end-start > {OUTLIER_THRESHOLD_US * 1000} "
        f"ORDER BY dur DESC",
        (slot0_text_id,),
    )
    outliers = cur.fetchall()
    print(
        f"\n{TAG} ===== slot0 outliers (>{OUTLIER_THRESHOLD_US}µs) — classified =====",
    )
    if not outliers:
        print("  None found")
    classifications = {}
    for i, o in enumerate(outliers):
        if d2a_name_id:
            cl = classify_outlier(
                con,
                o["start"],
                o["end"],
                consumer_tid,
                d2a_name_id,
                event_wait_id,
            )
        else:
            cl = {"classification": "no_D2A_nameId", "dur_us": o["dur"] / 1e3}
        classifications[cl["classification"]] = classifications.get(cl["classification"], 0) + 1
        dominant_label = cl.get("dominant_api", "none") or "none"
        gap_label = f"shm_gap={cl.get('shm_poll_gap_us', 0):.0f}µs" if cl.get("shm_poll_gap_us") else ""
        print(
            f"  [{i + 1:3d}] dur={cl['dur_us']:8.1f}µs  {cl['classification']:35s}  "
            f"dominant={dominant_label[:40]}({cl.get('dominant_frac', 0) * 100:.0f}%)  {gap_label}"
        )

    # --- Summary ---
    print(f"\n{TAG} ===== Classification summary =====")
    total = sum(classifications.values())
    for cls, n in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n}/{total} ({n / total * 100:.0f}%)")

    print(f"\n{TAG} ===== Hypothesis verdict =====")
    h5_n = sum(v for k, v in classifications.items() if "H5" in k)
    h2_n = sum(v for k, v in classifications.items() if "H2" in k)
    if h5_n > h2_n:
        print(f"  WINNER: H5 (D2A WDDM command-buffer stall) — {h5_n}/{total} outliers")
        print("  Interpretation: cudaMemcpy2DToArrayAsync CPU-side blocked waiting for")
        print("  WDDM GPU-fence; HWS=2 reduced systematic epoch gaps but did not")
        print("  eliminate all single-event stalls on the D2A copy path.")
    elif h2_n > h5_n:
        print(f"  WINNER: H2 (SHM poll / OS preemption) — {h2_n}/{total} outliers")
    else:
        print(f"  SPLIT: H5={h5_n} H2={h2_n} — mixed causes, inspect per-outlier table")

    print(f"\n{TAG} H1 (producer write-bias) note: slot0 has more writes than slot1/2")
    if stats.get("slot0", {}).get("n", 0) and stats.get("slot1", {}).get("n", 0):
        excess = stats["slot0"]["n"] - stats.get("slot1", {}).get("n", 0)
        print(
            f"  slot0 writes={stats['slot0']['n']}  slot1 writes={stats.get('slot1', {}).get('n', 0)}  "
            f"excess={excess} (+{excess / stats.get('slot1', {}).get('n', 1) * 100:.0f}%)"
        )
        print("  More writes = more chances to hit rare stall events; H1 is a")
        print("  contributing factor (write distribution), not a direct cause.")

    con.close()


if __name__ == "__main__":
    db = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer.sqlite")
    )
    run(db)
