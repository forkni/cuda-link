#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""Check the TouchDesigner Custom TOP DLLs inside built cuda-link wheels.

Rules (docs/CPP_TOP.md "Install: from the wheel"):

- A ``py3-none-any`` wheel carries no ``cuda_link/td_plugins/`` entry.
- A platform wheel carries either no DLLs, or exactly ``CudaLinkOutTOP.dll`` and
  ``CudaLinkInTOP.dll``, each stamped with the wheel's version.
- ``--require-plugins``: every platform wheel must carry both DLLs (release builds).

Usage:
    python scripts/check_wheel_plugins.py dist/cuda_link-*.whl [--require-plugins]

Exit 0 when every wheel passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

DLL_NAMES = ("CudaLinkOutTOP.dll", "CudaLinkInTOP.dll")
PLUGIN_PREFIX = "cuda_link/td_plugins/"


def wheel_tags(wheel: Path) -> tuple[str, str]:
    """(version, platform tag) from a wheel file name."""
    parts = wheel.name[: -len(".whl")].split("-")
    if len(parts) < 5:
        raise ValueError(f"not a wheel file name: {wheel.name}")
    return parts[1], parts[-1]


def check_wheel(wheel: Path, require_plugins: bool = False) -> list[str]:
    """Problems found in one wheel (empty list = pass)."""
    version, platform = wheel_tags(wheel)
    with zipfile.ZipFile(wheel) as zf:
        entries = {n[len(PLUGIN_PREFIX) :]: n for n in zf.namelist() if n.startswith(PLUGIN_PREFIX)}
        if platform == "any":
            return [f"{wheel.name}: a py3-none-any wheel must not carry {PLUGIN_PREFIX}"] if entries else []
        if not entries:
            return [f"{wheel.name}: no {PLUGIN_PREFIX} (release wheels need both DLLs)"] if require_plugins else []
        problems = []
        if sorted(entries) != sorted(DLL_NAMES):
            problems.append(f"{wheel.name}: {PLUGIN_PREFIX} holds {sorted(entries)}, expected {sorted(DLL_NAMES)}")
        stamp = b"\x00" + version.encode("ascii") + b"\x00"
        for name in DLL_NAMES:
            if name in entries and stamp not in zf.read(entries[name]):
                problems.append(f"{wheel.name}: {name} does not carry version {version}")
        return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wheels", nargs="+", type=Path)
    ap.add_argument("--require-plugins", action="store_true", help="platform wheels must carry both DLLs")
    args = ap.parse_args(argv)
    failed = False
    for wheel in args.wheels:
        problems = check_wheel(wheel, args.require_plugins)
        for p in problems:
            print(f"[FAIL] {p}")
        if problems:
            failed = True
        else:
            print(f"[OK] {wheel.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
