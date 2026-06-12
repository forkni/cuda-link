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

The authoritative pair list is PAIRS below.  Keep this docstring free of a
duplicate listing so it cannot drift.  NAMES is derived from PAIRS at module
load time — do not maintain it separately.

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

_SRC = REPO_ROOT / "src" / "cuda_link"
_TD = REPO_ROOT / "td_exporter"

# ---------------------------------------------------------------------------
# Authoritative pair list: (canonical, derived, mode).
# Consumed by the sync script, the pre-commit hook, and the test suite.
# To add a new module: add it here; NAMES and the tests update automatically.
# ---------------------------------------------------------------------------

PAIRS: list[tuple[Path, Path, Literal["byte_identical", "rewrite_relative"]]] = [
    # ---- byte_identical pairs -----------------------------------------------
    (_SRC / "_env.py", _TD / "Env.py", "byte_identical"),
    (_SRC / "_profile.py", _TD / "FrameProfile.py", "byte_identical"),
    (_SRC / "cuda_ipc_wrapper.py", _TD / "CUDAIPCWrapper.py", "byte_identical"),
    (_SRC / "cuda_runtime_types.py", _TD / "CUDARuntimeTypes.py", "byte_identical"),
    (_SRC / "cuda_graphs.py", _TD / "CUDAGraphs.py", "byte_identical"),
    (_SRC / "nvml_observer.py", _TD / "NVMLObserver.py", "byte_identical"),
    (_SRC / "shm_protocol.py", _TD / "SHMProtocol.py", "byte_identical"),
    (_SRC / "activation_barrier.py", _TD / "ActivationBarrier.py", "byte_identical"),
    (_SRC / "_doorbell.py", _TD / "Doorbell.py", "byte_identical"),
    # ---- rewrite_relative pairs (deep modules + their dependencies) ---------
    (_SRC / "_nvtx.py", _TD / "NVTXShim.py", "rewrite_relative"),
    (_SRC / "_exporter_port.py", _TD / "ExporterPort.py", "rewrite_relative"),
    (_SRC / "_importer_port.py", _TD / "ImporterPort.py", "rewrite_relative"),
    (_SRC / "_cuda_adapters.py", _TD / "CUDAAdapters.py", "rewrite_relative"),
    (_SRC / "exporter.py", _TD / "Exporter.py", "rewrite_relative"),
    (_SRC / "importer.py", _TD / "Importer.py", "rewrite_relative"),
]

# Canonical modules that intentionally have no td_exporter twin.
# tests/test_wrapper_sync.py::test_pairs_cover_all_mirrorable_modules enforces
# that every other src/cuda_link/*.py is in PAIRS.
CANONICAL_ONLY: frozenset[str] = frozenset(
    {
        "__init__.py",
        "_console.py",  # imported from cuda_link package directly by example scripts; no TD twin
        "cuda_ipc_importer.py",  # deprecation shim — TD consumers use Importer.py directly
    }
)

# Name mapping derived from PAIRS: relative-import stem → derived TD module name.
# Used by _rewrite_relative_imports() to resolve stems found inside rewrite_relative sources.
# Do not edit this dict manually — add pairs to PAIRS instead.
NAMES: dict[str, str] = {src.stem: dst.stem for src, dst, _ in PAIRS}


# ---------------------------------------------------------------------------
# Relative-import rewriter
# ---------------------------------------------------------------------------


def _resolve_name(stem: str, context_line: str) -> str:
    """Resolve a relative-import stem to its derived TD module name."""
    if stem not in NAMES:
        raise ValueError(
            f"Unknown relative import stem '{stem}' encountered in:\n"
            f"  {context_line!r}\n"
            "Add it to PAIRS in scripts/sync_td_wrapper.py"
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
        # Upward relative import — not supported (at any indentation)
        if re.match(r"^\s*from \.\.", line):
            raise ValueError(f"Unsupported upward relative import: {line!r}")

        # ``from . import stem[, stem2, ...]`` — bare module import (possibly indented or
        # comma-separated when the linter merges adjacent single-stem imports).
        m = re.match(r"^(\s*)from \. import ([\w ,]+?)(\s*)$", line)
        if m:
            indent = m.group(1)
            stems = [s.strip() for s in m.group(2).split(",") if s.strip()]
            eol = "\n" if line.endswith("\n") else ""
            # Emit one import statement per stem (black-compatible; avoids E702).
            for s in stems:
                derived = _resolve_name(s, line)
                out.append(f"{indent}import {derived} as {s}{eol}")
            continue

        # ``from .stem import ...`` — attribute import (possibly indented; single or multi-line opener)
        m = re.match(r"^(\s*)from \.([\w]+)( import .*)", line)
        if m:
            indent = m.group(1)
            stem = m.group(2)
            tail = m.group(3)
            derived = _resolve_name(stem, line)
            eol = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}from {derived}{tail}{eol}")
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
