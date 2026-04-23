"""
sync_td_wrapper.py — Keep td_exporter/CUDAIPCWrapper.py in sync with the canonical source.

Canonical source: src/cuda_link/cuda_ipc_wrapper.py
Derived copy:     td_exporter/CUDAIPCWrapper.py

The two files must be byte-identical. This script is called by build_wheel.cmd and
build_tox.py before packaging so that any change to the canonical wrapper is automatically
propagated. The test suite (tests/test_wrapper_sync.py) verifies identity at CI time.

Usage:
    python scripts/sync_td_wrapper.py [--check]

    --check   Verify only; exit non-zero if files differ (used in CI).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "cuda_link" / "cuda_ipc_wrapper.py"
DST = REPO_ROOT / "td_exporter" / "CUDAIPCWrapper.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync td_exporter/CUDAIPCWrapper.py from canonical source.")
    parser.add_argument("--check", action="store_true", help="Check only; exit 1 if files differ.")
    args = parser.parse_args()

    if not SRC.exists():
        print(f"ERROR: canonical source not found: {SRC}", file=sys.stderr)
        return 1

    src_bytes = SRC.read_bytes()

    if args.check:
        if not DST.exists():
            print(f"FAIL: {DST} does not exist.", file=sys.stderr)
            return 1
        dst_bytes = DST.read_bytes()
        if src_bytes == dst_bytes:
            print(f"OK: {DST.name} is in sync with {SRC.name}")
            return 0
        else:
            print(f"FAIL: {DST.name} differs from {SRC.name}. Run scripts/sync_td_wrapper.py to fix.", file=sys.stderr)
            return 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    print(f"Synced {SRC.relative_to(REPO_ROOT)} -> {DST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
