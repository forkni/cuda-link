"""
sync_td_wrapper.py — Keep td_exporter copies in sync with canonical sources.

THIS IS THE ONLY LEGITIMATE WAY TO UPDATE THE PAIRED td_exporter/ FILES.
Do not edit td_exporter/ paired files by hand — edits will be overwritten the
next time this script runs and the pre-commit sync-check hook will reject the
commit.  Edit the canonical src/cuda_link/ file, then run this script.

Two sync modes
--------------
byte_identical    — copy verbatim; canonical and derived must be byte-identical.
                    Used for modules that have no relative imports.
rewrite_relative  — rewrite ``from .X import`` → ``from DerivedX import`` and
                    ``from . import X`` → ``import DerivedX as X`` before writing.
                    Used for modules that use relative imports (incompatible with
                    TouchDesigner's flat COMP namespace where sibling Text DATs are
                    imported by bare name, not as a package).  See ADR-0002.

Pairs
-----
byte_identical (canonical → derived):
  cuda_ipc_wrapper.py      → CUDAIPCWrapper.py
  cuda_runtime_types.py    → CUDARuntimeTypes.py
  cuda_graphs.py           → CUDAGraphs.py
  nvml_observer.py         → NVMLObserver.py
  shm_protocol.py          → SHMProtocol.py
  activation_barrier.py    → ActivationBarrier.py

rewrite_relative (canonical → derived):
  _exporter_port.py        → ExporterPort.py
  _importer_port.py        → ImporterPort.py
  _exporter_adapters.py    → ExporterAdapters.py
  _cuda_adapters.py        → CudaAdapters.py
  exporter.py              → Exporter.py
  importer.py              → Importer.py

Usage:
    python scripts/sync_td_wrapper.py           # copy/rewrite src → td_exporter (update)
    python scripts/sync_td_wrapper.py --check   # verify only; exit 1 if any pair differs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Name mapping: canonical relative-import stem → derived TD module name.
# Used by _rewrite_relative_imports() for rewrite_relative pairs.
# Update this table whenever a canonical module gains a new relative-import
# dependency whose TD counterpart has a different name.
# ---------------------------------------------------------------------------

NAMES: dict[str, str] = {
    # Private support modules (drop leading underscore, PascalCase)
    "_nvtx": "NVTXShim",  # irregular (NVTXShim.py already exists)
    "_exporter_port": "ExporterPort",
    "_importer_port": "ImporterPort",
    "_exporter_adapters": "ExporterAdapters",
    "_cuda_adapters": "CudaAdapters",
    "_profile": "FrameProfile",  # FrameProfile.py already exists (byte_identical)
    # Public modules that byte_identical pairs already cover
    "activation_barrier": "ActivationBarrier",
    "cuda_runtime_types": "CUDARuntimeTypes",
    "shm_protocol": "SHMProtocol",
    "cuda_ipc_wrapper": "CUDAIPCWrapper",
    "cuda_graphs": "CUDAGraphs",
    "nvml_observer": "NVMLObserver",
}

_SRC = REPO_ROOT / "src" / "cuda_link"
_TD = REPO_ROOT / "td_exporter"

# Each entry: (canonical, derived, mode).
# The pre-commit hook and tests both consume this list.
PAIRS: list[tuple[Path, Path, Literal["byte_identical", "rewrite_relative"]]] = [
    # ---- byte_identical pairs (original 6) --------------------------------
    (_SRC / "cuda_ipc_wrapper.py", _TD / "CUDAIPCWrapper.py", "byte_identical"),
    (_SRC / "cuda_runtime_types.py", _TD / "CUDARuntimeTypes.py", "byte_identical"),
    (_SRC / "cuda_graphs.py", _TD / "CUDAGraphs.py", "byte_identical"),
    (_SRC / "nvml_observer.py", _TD / "NVMLObserver.py", "byte_identical"),
    (_SRC / "shm_protocol.py", _TD / "SHMProtocol.py", "byte_identical"),
    (_SRC / "activation_barrier.py", _TD / "ActivationBarrier.py", "byte_identical"),
    # ---- rewrite_relative pairs (new; deep modules + their dependencies) --
    (_SRC / "_exporter_port.py", _TD / "ExporterPort.py", "rewrite_relative"),
    (_SRC / "_importer_port.py", _TD / "ImporterPort.py", "rewrite_relative"),
    (_SRC / "_exporter_adapters.py", _TD / "ExporterAdapters.py", "rewrite_relative"),
    (_SRC / "_cuda_adapters.py", _TD / "CudaAdapters.py", "rewrite_relative"),
    (_SRC / "exporter.py", _TD / "Exporter.py", "rewrite_relative"),
    (_SRC / "importer.py", _TD / "Importer.py", "rewrite_relative"),
]


# ---------------------------------------------------------------------------
# Relative-import rewriter
# ---------------------------------------------------------------------------


def _resolve_name(stem: str, context_line: str) -> str:
    """Resolve a relative-import stem to its derived TD module name."""
    if stem not in NAMES:
        raise ValueError(
            f"Unknown relative import stem '{stem}' encountered in:\n"
            f"  {context_line!r}\n"
            "Add it to the NAMES mapping in scripts/sync_td_wrapper.py"
        )
    return NAMES[stem]


def rewrite_relative_imports(source: str) -> str:
    """Rewrite all relative imports in *source* to flat TD-compatible names.

    Handles:
      ``from .module import X``          → ``from Module import X``
      ``from .module import (``          → ``from Module import (``  (multi-line opener)
      ``from . import module``           → ``import Module as module``
    Rejects:
      ``from ..parent import ...``       → ValueError (upward relative)
    Non-relative lines are passed through unchanged.
    """
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        # Upward relative import — not supported
        if re.match(r"^from \.\.", line):
            raise ValueError(f"Unsupported upward relative import: {line!r}")

        # ``from . import stem`` — bare module import
        m = re.match(r"^(from \. import )([\w]+)(.*)", line)
        if m:
            stem = m.group(2)
            tail = m.group(3)
            derived = _resolve_name(stem, line)
            eol = "\n" if line.endswith("\n") else ""
            out.append(f"import {derived} as {stem}{tail}{eol}")
            continue

        # ``from .stem import ...`` — attribute import (single or multi-line opener)
        m = re.match(r"^from \.([\w]+)( import .*)", line)
        if m:
            stem = m.group(1)
            tail = m.group(2)
            derived = _resolve_name(stem, line)
            eol = "\n" if line.endswith("\n") else ""
            out.append(f"from {derived}{tail}{eol}")
            continue

        out.append(line)

    return "".join(out)


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------


def _derived_text(src_text: str, mode: str) -> str:
    if mode == "byte_identical":
        return src_text
    return rewrite_relative_imports(src_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync td_exporter copies from canonical sources.")
    parser.add_argument("--check", action="store_true", help="Check only; exit 1 if any pair differs.")
    args = parser.parse_args()

    exit_code = 0

    for src, dst, mode in PAIRS:
        if not src.exists():
            print(f"ERROR: canonical source not found: {src}", file=sys.stderr)
            return 1

        src_text = src.read_text(encoding="utf-8")
        try:
            expected = _derived_text(src_text, mode)
        except ValueError as exc:
            print(f"ERROR in {src.name}: {exc}", file=sys.stderr)
            return 1

        if args.check:
            if not dst.exists():
                print(f"FAIL [{mode}]: {dst} does not exist.", file=sys.stderr)
                exit_code = 1
                continue
            on_disk = dst.read_text(encoding="utf-8")
            if on_disk == expected:
                print(f"OK [{mode}]: {dst.name}")
            else:
                print(
                    f"FAIL [{mode}]: {dst.name} is out of sync with {src.name}. Run: python scripts/sync_td_wrapper.py",
                    file=sys.stderr,
                )
                exit_code = 1
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(expected, encoding="utf-8")
            print(f"Synced [{mode}] {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
