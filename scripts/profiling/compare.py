"""
compare.py — Compare two profile_export.py baselines and report regressions.

Usage:
    python scripts/profiling/compare.py <baseline.json> <current.json> [--threshold 10]

Exit codes:
    0  All regions within threshold
    1  One or more regions regressed beyond threshold
    2  Bad arguments or unreadable files

Designed for CI use: run after every PR that touches src/cuda_link/*.py:

    python scripts/profiling/profile_export.py --outfile .profiling/current.json
    python scripts/profiling/compare.py .profiling/baseline.json .profiling/current.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(2)


def _delta_pct(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two cuda-link profiling baselines.")
    parser.add_argument("baseline", help="Reference baseline JSON (committed)")
    parser.add_argument("current", help="Current run JSON (just captured)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold in %% (default: 10.0)",
    )
    parser.add_argument("--metric", default="median_us", help="Metric to compare (default: median_us)")
    args = parser.parse_args()

    baseline = _load(Path(args.baseline))
    current = _load(Path(args.current))

    b_cfg = baseline.get("config", {})
    c_cfg = current.get("config", {})

    # Warn if configs differ (different frame sizes → timing not comparable)
    for key in ("width", "height", "channels", "dtype", "slot_count"):
        if b_cfg.get(key) != c_cfg.get(key):
            print(
                f"WARNING: config mismatch on '{key}': "
                f"baseline={b_cfg.get(key)!r} current={c_cfg.get(key)!r} — "
                "comparison may not be meaningful."
            )

    b_regions: dict = baseline.get("regions", {})
    c_regions: dict = current.get("regions", {})
    all_regions = sorted(set(b_regions) | set(c_regions))

    regressions: list[str] = []
    rows: list[tuple[str, float, float, float]] = []

    for region in all_regions:
        # Skip scalar-valued regions (e.g. ptr_cache_miss_per_frame); only compare stat dicts.
        if not isinstance(b_regions.get(region), dict) or not isinstance(c_regions.get(region), dict):
            continue
        if region not in b_regions:
            print(f"  NEW   {region}: no baseline (current {args.metric}={c_regions[region].get(args.metric):.1f}µs)")
            continue
        if region not in c_regions:
            print(f"  GONE  {region}: was in baseline, not in current run")
            continue

        b_val: float = b_regions[region].get(args.metric, 0.0)
        c_val: float = c_regions[region].get(args.metric, 0.0)
        delta = _delta_pct(b_val, c_val)
        rows.append((region, b_val, c_val, delta))

    print(f"\n{'Region':<35} {'Baseline':>12} {'Current':>12} {'Delta':>10}")
    print("-" * 73)

    for region, b_val, c_val, delta in rows:
        marker = ""
        if delta > args.threshold:
            marker = " *** REGRESSION"
            regressions.append(region)
        elif delta < -args.threshold:
            marker = " *** IMPROVEMENT"
        print(f"{region:<35} {b_val:>10.1f}µs {c_val:>10.1f}µs {delta:>+9.1f}%{marker}")

    print()
    if regressions:
        print(f"FAIL: {len(regressions)} region(s) regressed beyond {args.threshold:.0f}% threshold:")
        for r in regressions:
            print(f"  - {r}")
        return 1

    print(f"OK: all regions within {args.threshold:.0f}% threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
