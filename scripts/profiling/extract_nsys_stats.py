"""Extract CUDA kernel and NVTX stats from an nsys-generated SQLite file."""

from __future__ import annotations

import argparse
import sqlite3


def extract_stats(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    print(f"Tables: {tables}\n")

    # --- CUDA kernel summary ---
    kern_tables = [t for t in tables if "kern" in t.lower() or "gpu" in t.lower()]
    print(f"Kernel-related tables: {kern_tables}\n")

    # Try common nsys SQLite schema tables
    for tbl in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_KERN_SUM"):
        if tbl in tables:
            print(f"=== {tbl} (top 10 by total duration) ===")
            try:
                rows = con.execute(f'SELECT * FROM {tbl} ORDER BY "Total Time (ns)" DESC LIMIT 10').fetchall()
                if rows:
                    print("  " + " | ".join(rows[0].keys()))
                    for r in rows:
                        print("  " + " | ".join(str(v) for v in r))
            except Exception:
                # Try without quoted column
                try:
                    rows = con.execute(f"SELECT * FROM {tbl} LIMIT 10").fetchall()
                    if rows:
                        print("  cols:", list(rows[0].keys()))
                        for r in rows[:5]:
                            print("  ", dict(r))
                except Exception as e2:
                    print(f"  Error: {e2}")
            print()

    # --- NVTX ranges summary ---
    nvtx_tables = [t for t in tables if "nvtx" in t.lower()]
    print(f"NVTX tables: {nvtx_tables}\n")
    for tbl in nvtx_tables[:3]:
        print(f"=== {tbl} (first 5 rows) ===")
        try:
            rows = con.execute(f"SELECT * FROM {tbl} LIMIT 5").fetchall()
            if rows:
                print("  cols:", list(rows[0].keys()))
                for r in rows:
                    print("  ", dict(r))
        except Exception as e:
            print(f"  Error: {e}")
        print()

    # --- Raw kernel timing via CUPTI ---
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        print("=== Top 10 kernels by total duration ===")
        try:
            rows = con.execute("""
                SELECT shortName,
                       COUNT(*) as count,
                       SUM(end-start) as total_ns,
                       AVG(end-start) as avg_ns,
                       MIN(end-start) as min_ns,
                       MAX(end-start) as max_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                GROUP BY shortName
                ORDER BY total_ns DESC
                LIMIT 10
            """).fetchall()
            print(f"  {'Kernel':<60} {'count':>6} {'total_ms':>10} {'avg_us':>8} {'min_us':>7} {'max_us':>7}")
            print("  " + "-" * 100)
            for r in rows:
                print(
                    f"  {str(r[0]):<60} {r[1]:>6} {r[2] / 1e6:>10.3f} {r[3] / 1e3:>8.2f} {r[4] / 1e3:>7.2f} {r[5] / 1e3:>7.2f}"
                )
        except Exception as e:
            print(f"  Error: {e}")
        print()

    # --- CUDA memcpy summary ---
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        print("=== CUDA Memcpy operations ===")
        try:
            rows = con.execute("""
                SELECT copyKind,
                       COUNT(*) as count,
                       SUM(end-start) as total_ns,
                       AVG(end-start) as avg_ns,
                       AVG(bytes) as avg_bytes
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                GROUP BY copyKind
                ORDER BY total_ns DESC
            """).fetchall()
            for r in rows:
                kind_map = {1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "DtoD_async"}
                kind = kind_map.get(r[0], str(r[0]))
                print(
                    f"  {kind}: count={r[1]}, total={r[2] / 1e6:.2f}ms, avg={r[3] / 1e3:.1f}µs, avg_bytes={r[4] / 1024:.1f}KB"
                )
        except Exception as e:
            print(f"  Error: {e}")
        print()

    # --- CUDA API calls ---
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        print("=== Top CUDA Runtime API calls by total duration ===")
        try:
            rows = con.execute("""
                SELECT nameId,
                       COUNT(*) as count,
                       SUM(end-start) as total_ns,
                       AVG(end-start) as avg_ns
                FROM CUPTI_ACTIVITY_KIND_RUNTIME
                GROUP BY nameId
                ORDER BY total_ns DESC
                LIMIT 15
            """).fetchall()
            # Try to resolve name IDs
            name_map = {}
            try:
                names = con.execute("SELECT id, value FROM StringIds").fetchall()
                name_map = {r[0]: r[1] for r in names}
            except Exception:
                pass
            print(f"  {'API Call':<45} {'count':>6} {'total_ms':>10} {'avg_us':>8}")
            print("  " + "-" * 75)
            for r in rows:
                name = name_map.get(r[0], f"id={r[0]}")
                print(f"  {name:<45} {r[1]:>6} {r[2] / 1e6:>10.3f} {r[3] / 1e3:>8.2f}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", help="Path to .sqlite file from nsys")
    args = parser.parse_args()
    extract_stats(args.db)
