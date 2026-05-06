"""Step D probe scanner — validates run integrity and emits a pass/fail verdict.

Usage:
    python scripts/probes/scan_step_d.py <artifact_dir>

<artifact_dir> must contain:
    producer.log       — stdout/stderr of example_sender_python.py
    td_sender.txt      — TouchDesigner textport save (optional but recommended)
    env.txt            — env snapshot written by step_d_runner.cmd

Outputs a JSON verdict to stdout and writes it to <artifact_dir>/verdict.json.

Exit codes:
    0  — valid run, verdict F9_DROPPABLE or F9_LOAD_BEARING
    1  — invalid run (env leak, F9 active during probe)
    2  — usage error or missing required file
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CYCLE_1_POST_LIMIT_MS = 5.0
CYCLE_N_POST_LIMIT_MS = 50.0
FPS_FLOOR = 58.0

# Strings that indicate F9 was active during the run (validity gate).
F9_ACTIVE_MARKERS = [
    "[ACTIVATION_BARRIER]",
]

# Strings that indicate a TDR or hard CUDA failure.
HARD_FAILURE_MARKERS = [
    "nvidia driver error",
    "cuda_error_launch_timeout",
    "cuda error",
    "tdr",
    "device lost",
    "dxgi_error_device_removed",
]

# Receiver-A non-recovery signature — these appear when the residual fails
# to self-recover (regression signal for F8/F2 drops; look for it in F9 drop).
RECEIVER_STUCK_MARKERS = [
    "sender shutdown detected",
    "retry",
    "re-attach",
    "never recovered",
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches lines like: [sender] cycle=2 post=31.4ms fps=59.6
_CYCLE_RE = re.compile(
    r"cycle=(\d+).*?post=([\d.]+)\s*ms.*?fps=([\d.]+)",
    re.IGNORECASE,
)

# Alternative: post= and fps= on separate fields anywhere in the line.
_POST_RE = re.compile(r"\bpost=([\d.]+)\s*ms", re.IGNORECASE)
_FPS_RE = re.compile(r"\bfps=([\d.]+)", re.IGNORECASE)
_CYCLE_TAG_RE = re.compile(r"\bcycle=(\d+)\b", re.IGNORECASE)


def _load(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []


def _count_markers(lines: list[str], markers: list[str]) -> int:
    total = 0
    for line in lines:
        lower = line.lower()
        for m in markers:
            if m.lower() in lower:
                total += 1
    return total


def _parse_cycles(lines: list[str]) -> list[dict]:
    """Extract per-cycle post= and fps= measurements.

    Tolerates log lines in either structured form (cycle=N post=X fps=Y) or
    lines where post= appears without an explicit cycle= tag (uses cycle order).
    """
    structured: dict[int, dict] = {}
    untagged: list[tuple[float, float]] = []

    for line in lines:
        m_cycle = _CYCLE_RE.search(line)
        if m_cycle:
            cycle_n = int(m_cycle.group(1))
            post_ms = float(m_cycle.group(2))
            fps = float(m_cycle.group(3))
            if cycle_n not in structured:
                structured[cycle_n] = {"cycle": cycle_n, "post_ms": post_ms, "fps": fps}
            continue

        m_post = _POST_RE.search(line)
        m_fps = _FPS_RE.search(line)
        m_ctag = _CYCLE_TAG_RE.search(line)
        if m_post and m_fps:
            post_ms = float(m_post.group(1))
            fps = float(m_fps.group(1))
            if m_ctag:
                cycle_n = int(m_ctag.group(1))
                if cycle_n not in structured:
                    structured[cycle_n] = {"cycle": cycle_n, "post_ms": post_ms, "fps": fps}
            else:
                untagged.append((post_ms, fps))

    if structured:
        return [structured[k] for k in sorted(structured)]

    # Fall back to order-based cycle numbering.
    return [{"cycle": i + 1, "post_ms": p, "fps": f} for i, (p, f) in enumerate(untagged)]


def _check_slot_escalation(cycles: list[dict]) -> bool:
    """Return True if slot-0 close timing escalates strictly across cycles."""
    posts = [c["post_ms"] for c in cycles if "post_ms" in c]
    if len(posts) < 2:
        return False
    return all(posts[i] < posts[i + 1] for i in range(len(posts) - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def scan(artifact_dir: Path) -> dict:
    producer_lines = _load(artifact_dir / "producer.log")
    td_lines = _load(artifact_dir / "td_sender.txt")
    env_lines = _load(artifact_dir / "env.txt")

    result: dict = {
        "artifact_dir": str(artifact_dir),
        "files_found": {
            "producer.log": (artifact_dir / "producer.log").exists(),
            "td_sender.txt": (artifact_dir / "td_sender.txt").exists(),
            "env.txt": (artifact_dir / "env.txt").exists(),
        },
    }

    # --- env audit (informational) ------------------------------------------
    f9_in_env = [
        ln
        for ln in env_lines
        if "CUDALINK_TD_ACTIVATION_BARRIER" in ln.upper() or "CUDALINK_ACTIVATION_BARRIER" in ln.upper()
    ]
    result["env_f9_vars"] = f9_in_env  # should be empty

    # --- validity gate -------------------------------------------------------
    f9_hits_producer = _count_markers(producer_lines, F9_ACTIVE_MARKERS)
    f9_hits_td = _count_markers(td_lines, F9_ACTIVE_MARKERS)
    total_f9_hits = f9_hits_producer + f9_hits_td

    if total_f9_hits > 0:
        result["valid"] = False
        result["validity_reason"] = (
            f"F9 active during run: {f9_hits_producer} hit(s) in producer.log, "
            f"{f9_hits_td} hit(s) in td_sender.txt. "
            "Env was not clean — retry in a fresh cmd.exe window."
        )
        result["verdict"] = "INVALID"
        return result

    result["valid"] = True
    result["validity_reason"] = "0 [ACTIVATION_BARRIER] hits in producer.log + td_sender.txt — F9 was off"

    # --- hard failure check --------------------------------------------------
    hard_hits_prod = _count_markers(producer_lines, HARD_FAILURE_MARKERS)
    hard_hits_td = _count_markers(td_lines, HARD_FAILURE_MARKERS)
    tdr_observed = (hard_hits_prod + hard_hits_td) > 0
    result["tdr_observed"] = tdr_observed

    # --- receiver-A recovery check ------------------------------------------
    recv_stuck_hits = _count_markers(td_lines, RECEIVER_STUCK_MARKERS)
    result["receiver_stuck_hits"] = recv_stuck_hits
    receiver_a_no_recovery = recv_stuck_hits > 3  # small residual is expected

    # --- per-cycle telemetry ------------------------------------------------
    cycles = _parse_cycles(producer_lines)
    if not cycles:
        # try TD log if producer had no cycle data
        cycles = _parse_cycles(td_lines)

    cycle_results = []
    for c in cycles:
        n = c["cycle"]
        post_ms = c.get("post_ms")
        fps = c.get("fps")
        limit = CYCLE_1_POST_LIMIT_MS if n == 1 else CYCLE_N_POST_LIMIT_MS
        post_pass = post_ms is not None and post_ms <= limit
        fps_pass = fps is not None and fps >= FPS_FLOOR
        cycle_results.append(
            {
                "cycle": n,
                "first_settle_post_ms": post_ms,
                "fps_avg": fps,
                "post_limit_ms": limit,
                "post_pass": post_pass,
                "fps_pass": fps_pass,
                "pass": post_pass and fps_pass,
            }
        )

    result["cycles"] = cycle_results
    result["slot0_escalating"] = _check_slot_escalation(cycles)

    # --- overall verdict -----------------------------------------------------
    all_cycles_pass = all(c["pass"] for c in cycle_results) if cycle_results else None
    regression = tdr_observed or receiver_a_no_recovery or (all_cycles_pass is False)

    if all_cycles_pass is None:
        result["verdict"] = "INCONCLUSIVE"
        result["verdict_reason"] = "No cycle telemetry found in logs — check log format."
    elif regression:
        result["verdict"] = "F9_LOAD_BEARING"
        reasons = []
        if tdr_observed:
            reasons.append("TDR/hard CUDA failure detected")
        if receiver_a_no_recovery:
            reasons.append(f"Receiver-A stuck ({recv_stuck_hits} retry-loop hits)")
        failing = [c for c in cycle_results if not c["pass"]]
        if failing:
            reasons.append(
                "cycle(s) failed thresholds: "
                + ", ".join(f"cycle {c['cycle']} post={c['first_settle_post_ms']}ms" for c in failing)
            )
        result["verdict_reason"] = "; ".join(reasons)
    else:
        result["verdict"] = "F9_DROPPABLE"
        result["verdict_reason"] = f"All {len(cycle_results)} cycle(s) passed — F9 can be removed from minimum stack"

    return result


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    artifact_dir = Path(sys.argv[1]).resolve()
    if not artifact_dir.is_dir():
        print(f"ERROR: {artifact_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    result = scan(artifact_dir)

    output = json.dumps(result, indent=2)
    print(output)

    verdict_path = artifact_dir / "verdict.json"
    verdict_path.write_text(output, encoding="utf-8")
    print(f"\nVerdict written to: {verdict_path}", file=sys.stderr)

    if result["verdict"] == "INVALID":
        sys.exit(1)


if __name__ == "__main__":
    main()
