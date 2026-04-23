"""
sync_td_wrapper.py — Keep td_exporter copies in sync with canonical sources.

Canonical → Derived pairs:
  src/cuda_link/cuda_ipc_wrapper.py → td_exporter/CUDAIPCWrapper.py
  src/cuda_link/nvml_observer.py    → td_exporter/NVMLObserver.py

Both derived files must be byte-identical to their canonical source.
This script is called by build_wheel.cmd step [1.5] and the test suite
(tests/test_wrapper_sync.py) verifies identity at CI time.

Usage:
    python scripts/sync_td_wrapper.py [--check]

    --check   Verify only; exit non-zero if any pair differs (used in CI).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAIRS: list[tuple[Path, Path]] = [
    (
        REPO_ROOT / "src" / "cuda_link" / "cuda_ipc_wrapper.py",
        REPO_ROOT / "td_exporter" / "CUDAIPCWrapper.py",
    ),
    (
        REPO_ROOT / "src" / "cuda_link" / "nvml_observer.py",
        REPO_ROOT / "td_exporter" / "NVMLObserver.py",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync td_exporter copies from canonical sources.")
    parser.add_argument("--check", action="store_true", help="Check only; exit 1 if any pair differs.")
    args = parser.parse_args()

    exit_code = 0

    for src, dst in PAIRS:
        if not src.exists():
            print(f"ERROR: canonical source not found: {src}", file=sys.stderr)
            return 1

        src_bytes = src.read_bytes()

        if args.check:
            if not dst.exists():
                print(f"FAIL: {dst} does not exist.", file=sys.stderr)
                exit_code = 1
                continue
            dst_bytes = dst.read_bytes()
            if src_bytes == dst_bytes:
                print(f"OK: {dst.name} is in sync with {src.name}")
            else:
                print(
                    f"FAIL: {dst.name} differs from {src.name}. Run scripts/sync_td_wrapper.py to fix.",
                    file=sys.stderr,
                )
                exit_code = 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Synced {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
