# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""The TouchDesigner Custom TOP DLLs: where this install has them, and the system install.

The release platform wheel (``cp311-win_amd64``) ships ``CudaLinkOutTOP.dll``,
``CudaLinkInTOP.dll`` and ``BUILD_INFO.txt`` in ``cuda_link/td_plugins/``, built from the same
tree and version as the Python package. A host that installs the wheel (for example a
TouchDesigner project with a venv) points a Custom TOP's Plugin Path at :func:`plugin_path`.
Dev builds and the ``py3-none-any`` wheel carry no DLLs; then :func:`plugin_path` returns None.

The system install copies the DLLs to ``<Documents>/Derivative/Plugins/CudaLink/`` with an
*install record*, ``VERSION.txt`` (ADR-0016)::

    cuda-link 2.0.0
    build ba5e5ab
    protocol v0.6 (magic 0x43495036)
    td header OP_CommonAPIVersion 1
    cuda toolkit 12.6
    installed 2026-11-22T10:00:00 from <source folder>
    sha256 CudaLinkOutTOP.dll <hex>
    sha256 CudaLinkInTOP.dll <hex>

The first line is the contract with the ``.tox`` reader (``CUDAIPCExtension.native_tops_status``,
which also reads ``build`` and ``cuda toolkit``). Two writers use this module:
``scripts/install_cpp_top.py`` (a local cpp_top build or ``--from-wheel``) and
``scripts/install_td_library.py`` (after the wheel install). This file has no imports from the
package, so the scripts load it by path without importing ``cuda_link``.

Loading rules are in docs/CPP_TOP.md: TouchDesigner refuses a second copy of the same plugin,
and it locks a loaded DLL, so install and uninstall need TouchDesigner closed.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PluginKind = Literal["out", "in"]

#: DLL file name per Custom TOP: "out" = CUDA Link Out (sender), "in" = CUDA Link In (receiver).
DLL_NAMES: dict[str, str] = {"out": "CudaLinkOutTOP.dll", "in": "CudaLinkInTOP.dll"}
DLL_FILES: tuple[str, str] = (DLL_NAMES["out"], DLL_NAMES["in"])

PLUGIN_DIR_NAME = "td_plugins"
#: Written by cpp_top's CMake next to the DLLs: "cuda-link <ver>", "build <sha>", "cuda toolkit <x.y>",
#: "td header OP_CommonAPIVersion <n>".
BUILD_INFO_FILE = "BUILD_INFO.txt"
#: The system install: <Plugins>/CudaLink/{DLLs, VERSION.txt}.
VENDOR_FOLDER = "CudaLink"
RECORD_FILE = "VERSION.txt"
_TD_HEADER_KEY = "td header OP_CommonAPIVersion"

Say = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# The DLLs shipped in this package
# ---------------------------------------------------------------------------


def plugin_dir() -> Path:
    """The folder that holds the shipped DLLs (it may not exist in a dev build)."""
    return Path(__file__).resolve().parent / PLUGIN_DIR_NAME


def plugin_path(kind: PluginKind) -> Path | None:
    """Absolute path of the DLL for ``kind``, or None when this install carries no DLLs.

    Raises:
        ValueError: ``kind`` is not "out" or "in".
    """
    try:
        name = DLL_NAMES[kind]
    except KeyError:
        raise ValueError(f"kind must be 'out' or 'in', not {kind!r}") from None
    path = plugin_dir() / name
    return path if path.is_file() else None


def has_plugins() -> bool:
    """True when both DLLs are present."""
    return all(plugin_path(kind) is not None for kind in ("out", "in"))


# ---------------------------------------------------------------------------
# The system Plugins folder
# ---------------------------------------------------------------------------


def documents_dir() -> Path:
    """The user's real Documents folder (FOLDERID_Documents; follows a OneDrive redirect).

    Falls back to ~/Documents off Windows or when the lookup fails.
    """
    if os.name == "nt":
        with contextlib.suppress(Exception):
            import ctypes  # noqa: PLC0415 -- Windows only
            import uuid  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415

            class _GUID(ctypes.Structure):
                _fields_ = [
                    ("d1", wintypes.DWORD),
                    ("d2", wintypes.WORD),
                    ("d3", wintypes.WORD),
                    ("d4", ctypes.c_ubyte * 8),
                ]

            u = uuid.UUID("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")  # FOLDERID_Documents
            g = _GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
            out = ctypes.c_wchar_p()
            try:
                hr = ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out))  # type: ignore[attr-defined]
                if hr == 0 and out.value:
                    return Path(out.value)
            finally:
                ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
    return Path.home() / "Documents"


def default_plugins_dir() -> Path:
    """<Documents>/Derivative/Plugins: TouchDesigner loads every subfolder of it."""
    return documents_dir() / "Derivative" / "Plugins"


def td_running() -> bool:
    """True when a TouchDesigner.exe process exists (Windows only; False elsewhere)."""
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq TouchDesigner.exe", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "touchdesigner.exe" in out.lower()


# ---------------------------------------------------------------------------
# The install record (VERSION.txt)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_build_info(dll_dir: Path) -> dict[str, str]:
    """``BUILD_INFO.txt`` next to the DLLs as {"cuda-link": .., "build": .., "cuda toolkit": ..,
    "td header OP_CommonAPIVersion": ..}.

    Empty when the file is missing (a cpp_top build from before the sidecar).
    """
    keys = ("cuda-link", "build", "cuda toolkit", _TD_HEADER_KEY)
    return _parse_fields(dll_dir / BUILD_INFO_FILE, keys)


def _parse_fields(path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in lines:
        for key in keys:
            if line.startswith(key + " ") and key not in out:
                out[key] = line[len(key) + 1 :].strip()
    return out


@dataclass(frozen=True)
class InstallRecord:
    """What is installed in <Plugins>/CudaLink and where it came from. Unknown fields: "unknown"."""

    version: str = "unknown"
    build: str = "unknown"
    protocol: str = "unknown"  # "v0.6 (magic 0x43495036)"
    td_header: str = "unknown"  # OP_CommonAPIVersion the DLLs were built against
    cuda_toolkit: str = "unknown"
    installed: str = "unknown"  # "<ISO time> from <source folder>"
    sha256: dict[str, str] = field(default_factory=dict)

    def text(self) -> str:
        lines = [
            f"cuda-link {self.version}",
            f"build {self.build}",
            f"protocol {self.protocol}",
            f"td header OP_CommonAPIVersion {self.td_header}",
            f"cuda toolkit {self.cuda_toolkit}",
            f"installed {self.installed}",
        ]
        lines += [f"sha256 {name} {digest}" for name, digest in self.sha256.items()]
        return "\n".join(lines) + "\n"

    @classmethod
    def parse(cls, text: str) -> InstallRecord:
        """Read a record. Lines it does not know are ignored; missing ones stay "unknown"."""
        values: dict[str, str] = {}
        digests: dict[str, str] = {}
        prefixes = {
            "cuda-link ": "version",
            "build ": "build",
            "protocol ": "protocol",
            "td header OP_CommonAPIVersion ": "td_header",
            "cuda toolkit ": "cuda_toolkit",
            "installed ": "installed",
        }
        for line in text.splitlines():
            if line.startswith("sha256 "):
                parts = line.split()
                if len(parts) == 3:
                    digests[parts[1]] = parts[2]
                continue
            for prefix, name in prefixes.items():
                if line.startswith(prefix) and name not in values:
                    values[name] = line[len(prefix) :].strip()
                    break
        return cls(**values, sha256=digests)


def make_record(
    dll_dir: Path,
    *,
    version: str | None = None,
    protocol: str | None = None,
    td_header: str | None = None,
    cuda_toolkit: str | None = None,
    now: _dt.datetime | None = None,
) -> InstallRecord:
    """The record for the DLLs in ``dll_dir``. ``BUILD_INFO.txt`` there gives the defaults for
    version, build, CUDA toolkit and TD header; an explicit argument wins. The DLL hashes are
    computed."""
    info = read_build_info(dll_dir)
    stamp = (now or _dt.datetime.now()).isoformat(timespec="seconds")
    return InstallRecord(
        version=version or info.get("cuda-link", "unknown"),
        build=info.get("build", "unknown"),
        protocol=protocol or "unknown",
        td_header=td_header or info.get(_TD_HEADER_KEY, "unknown"),
        cuda_toolkit=cuda_toolkit or info.get("cuda toolkit", "unknown"),
        installed=f"{stamp} from {dll_dir}",
        sha256={name: sha256_file(dll_dir / name) for name in DLL_FILES if (dll_dir / name).is_file()},
    )


def write_record(dest: Path, record: InstallRecord) -> Path:
    """Write ``record`` to dest/VERSION.txt; returns the path."""
    path = dest / RECORD_FILE
    path.write_text(record.text(), encoding="utf-8")
    return path


def read_record(plugins_dir: Path) -> InstallRecord | None:
    """The record of the system install, or None when there is none."""
    try:
        text = (plugins_dir / VENDOR_FOLDER / RECORD_FILE).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return InstallRecord.parse(text)


# ---------------------------------------------------------------------------
# install / status / uninstall (exit codes: 0 ok, 1 nothing there, 2 refused)
# ---------------------------------------------------------------------------


def install(
    plugins_dir: Path,
    dll_dir: Path,
    record: InstallRecord | None = None,
    dry_run: bool = False,
    say: Say = _print,
) -> int:
    """Copy both DLLs from ``dll_dir`` to <plugins_dir>/CudaLink and write the install record.

    TouchDesigner must be closed (the caller checks :func:`td_running`). Refuses (2) when a DLL
    is missing or the CudaLink folder holds files that are not ours.
    """
    missing = [n for n in DLL_FILES if not (dll_dir / n).is_file()]
    if missing:
        say(f"[ERROR] Not found in {dll_dir}: {', '.join(missing)}.")
        return 2
    dest = plugins_dir / VENDOR_FOLDER
    if dest.exists():
        extra = sorted(p.name for p in dest.iterdir() if p.name not in (*DLL_FILES, RECORD_FILE))
        if extra:
            say(f"[ERROR] {dest} holds other files ({', '.join(extra)}). Keep only the CUDA Link files there.")
            return 2
    say(f"Install: {dll_dir} -> {dest}")
    if dry_run:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    for n in DLL_FILES:
        shutil.copy2(dll_dir / n, dest / n)
    write_record(dest, record or make_record(dll_dir))
    say("[OK] Installed. Open TouchDesigner and approve the two plugin prompts.")
    say("     Remove any copy in a project's Plugins folder: two copies give a blocking load error.")
    return 0


def status(plugins_dir: Path, say: Say = _print) -> int:
    """Print the system install (1 when there is none)."""
    dest = plugins_dir / VENDOR_FOLDER
    if not dest.is_dir():
        say(f"Not installed: {dest} does not exist.")
        return 1
    say(f"Installed in {dest}:")
    for n in DLL_FILES:
        say(f"  {n}: {'present' if (dest / n).is_file() else 'MISSING'}")
    record_path = dest / RECORD_FILE
    say(record_path.read_text(encoding="utf-8").rstrip() if record_path.is_file() else f"  {RECORD_FILE}: MISSING")
    others = sorted(
        p.name
        for p in plugins_dir.iterdir()
        if p.is_dir() and p.name != VENDOR_FOLDER and any((p / n).is_file() for n in DLL_FILES)
    )
    if others:
        say(f"[WARN] More CUDA Link DLLs under {plugins_dir}: {', '.join(others)}. TD loads them too.")
    return 0


def uninstall(plugins_dir: Path, dry_run: bool = False, now: _dt.datetime | None = None, say: Say = _print) -> int:
    """Move <plugins_dir>/CudaLink OUT of Plugins (a rename inside it does not uninstall)."""
    dest = plugins_dir / VENDOR_FOLDER
    if not dest.is_dir():
        say(f"Nothing to uninstall: {dest} does not exist.")
        return 1
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = plugins_dir.parent / f"{VENDOR_FOLDER}.removed-{stamp}"
    say(f"Uninstall: move {dest} -> {target} (outside Plugins)")
    if dry_run:
        return 0
    shutil.move(str(dest), str(target))
    say("[OK] Removed from Plugins. Saved projects keep their nodes; they error until you install again.")
    return 0
