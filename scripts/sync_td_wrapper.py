"""
sync_td_wrapper.py — Keep td_exporter copies in sync with canonical sources.

THIS IS THE ONLY LEGITIMATE WAY TO UPDATE THE PAIRED td_exporter/ FILES.
Do not edit td_exporter/ paired files by hand — edits will be overwritten the
next time this script runs and the pre-commit sync-check hook will reject the
commit.  Edit the canonical src/cuda_link/ file, then run this script.

Canonical (src/cuda_link/)     → Derived (td_exporter/)
  cuda_ipc_wrapper.py          → CUDAIPCWrapper.py
  cuda_runtime_types.py        → CUDARuntimeTypes.py
  cuda_graphs.py               → CUDAGraphs.py
  nvml_observer.py             → NVMLObserver.py
  shm_protocol.py              → SHMProtocol.py
  activation_barrier.py        → ActivationBarrier.py

All six derived files must be byte-identical to their canonical source.
Verified by tests/test_wrapper_sync.py and by the pre-commit sync-check hook.
This script is also called by build_wheel.cmd step [1.5].

Usage:
    python scripts/sync_td_wrapper.py           # copy src → td_exporter (update)
    python scripts/sync_td_wrapper.py --check   # verify only; exit 1 if any pair differs
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
        REPO_ROOT / "src" / "cuda_link" / "cuda_runtime_types.py",
        REPO_ROOT / "td_exporter" / "CUDARuntimeTypes.py",
    ),
    (
        REPO_ROOT / "src" / "cuda_link" / "cuda_graphs.py",
        REPO_ROOT / "td_exporter" / "CUDAGraphs.py",
    ),
    (
        REPO_ROOT / "src" / "cuda_link" / "nvml_observer.py",
        REPO_ROOT / "td_exporter" / "NVMLObserver.py",
    ),
    (
        REPO_ROOT / "src" / "cuda_link" / "shm_protocol.py",
        REPO_ROOT / "td_exporter" / "SHMProtocol.py",
    ),
    (
        REPO_ROOT / "src" / "cuda_link" / "activation_barrier.py",
        REPO_ROOT / "td_exporter" / "ActivationBarrier.py",
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
