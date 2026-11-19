"""install_cpp_top.py — install the C++ CUDA Link TOPs into TouchDesigner's system Plugins folder.

Layout (docs/CPP_TOP.md, "Install: system Plugins folder"):

    <Documents>\\Derivative\\Plugins\\CudaLink\\
        CudaLinkOutTOP.dll
        CudaLinkInTOP.dll
        VERSION.txt

Commands:
    python scripts\\install_cpp_top.py install [--dll-dir DIR]   copy both DLLs, write VERSION.txt
    python scripts\\install_cpp_top.py status                    show the installed copy
    python scripts\\install_cpp_top.py uninstall                 move the folder OUT of Plugins

Rules the script keeps:
    * TouchDesigner must be closed: TD locks a loaded DLL (install and uninstall refuse while
      TouchDesigner.exe runs; --force skips the check).
    * Uninstall moves the folder to <Documents>\\Derivative\\CudaLink.removed-<time>. A rename
      inside Plugins does not uninstall: TD loads every subfolder.
    * Documents is resolved with the Windows known-folder API, so a OneDrive redirect is found.

Common flags:
    --plugins-dir DIR   use DIR instead of <Documents>\\Derivative\\Plugins (tests, odd setups)
    --dry-run           print what would happen, change nothing
    --force             skip the "TouchDesigner is running" check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DLL_NAMES = ("CudaLinkOutTOP.dll", "CudaLinkInTOP.dll")
VENDOR_FOLDER = "CudaLink"
VERSION_FILE = "VERSION.txt"
DEFAULT_DLL_DIR = ROOT / "cpp_top" / "build"
_INIT_PY = ROOT / "src" / "cuda_link" / "__init__.py"
_SHM_PY = ROOT / "src" / "cuda_link" / "shm_protocol.py"
_TD_HEADER = ROOT / "cpp_top" / "vendor" / "td" / "2025" / "CPlusPlus_Common.h"


def documents_dir() -> Path:
    """The user's real Documents folder (FOLDERID_Documents; follows a OneDrive redirect)."""
    if os.name == "nt":
        import ctypes
        import uuid
        from ctypes import wintypes

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
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        if shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out)) == 0:
            try:
                return Path(out.value or "")
            finally:
                ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
    return Path.home() / "Documents"


def default_plugins_dir() -> Path:
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


def _grep(path: Path, pattern: str) -> str | None:
    try:
        m = re.search(pattern, path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
    except OSError:
        return None
    return m.group(1) if m else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version_text(dll_dir: Path, now: _dt.datetime | None = None) -> str:
    """Contents of VERSION.txt for the DLLs in dll_dir."""
    now = now or _dt.datetime.now()
    cuda = _grep(dll_dir / "CMakeCache.txt", r"^CUDAToolkit_BIN_DIR:PATH=.*?[\\/]v(\d+\.\d+)[\\/]")
    version = _grep(_INIT_PY, r"^__version__\s*=\s*[\"']([^\"']+)")
    magic = _grep(_SHM_PY, r"^PROTOCOL_MAGIC[^=]*=\s*(0x[0-9A-Fa-f]+)")
    api = _grep(_TD_HEADER, r"OP_CommonAPIVersion\s*=\s*(\d+)")
    lines = [
        f"cuda-link {version or 'unknown'}",
        f"protocol v0.6 (magic {magic or 'unknown'})",
        f"td header OP_CommonAPIVersion {api or 'unknown'}",
        f"cuda toolkit {cuda or 'unknown'}",
        f"installed {now.isoformat(timespec='seconds')} from {dll_dir}",
    ]
    lines += [f"sha256 {n} {_sha256(dll_dir / n)}" for n in DLL_NAMES]
    return "\n".join(lines) + "\n"


def _say(msg: str) -> None:
    print(msg, flush=True)


def install(plugins_dir: Path, dll_dir: Path, dry_run: bool = False) -> int:
    missing = [n for n in DLL_NAMES if not (dll_dir / n).is_file()]
    if missing:
        _say(f"[ERROR] Not found in {dll_dir}: {', '.join(missing)}. Build cpp_top first or pass --dll-dir.")
        return 2
    dest = plugins_dir / VENDOR_FOLDER
    if dest.exists():
        extra = sorted(p.name for p in dest.iterdir() if p.name not in (*DLL_NAMES, VERSION_FILE))
        if extra:
            _say(f"[ERROR] {dest} holds other files ({', '.join(extra)}). Keep only the CUDA Link files there.")
            return 2
    _say(f"Install: {dll_dir} -> {dest}")
    if dry_run:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    for n in DLL_NAMES:
        shutil.copy2(dll_dir / n, dest / n)
    (dest / VERSION_FILE).write_text(version_text(dll_dir), encoding="utf-8")
    _say("[OK] Installed. Open TouchDesigner and approve the two plugin prompts.")
    _say("     Remove any copy in a project's Plugins folder: two copies give a blocking load error.")
    return 0


def status(plugins_dir: Path) -> int:
    dest = plugins_dir / VENDOR_FOLDER
    if not dest.is_dir():
        _say(f"Not installed: {dest} does not exist.")
        return 1
    _say(f"Installed in {dest}:")
    for n in DLL_NAMES:
        _say(f"  {n}: {'present' if (dest / n).is_file() else 'MISSING'}")
    vf = dest / VERSION_FILE
    _say(vf.read_text(encoding="utf-8").rstrip() if vf.is_file() else f"  {VERSION_FILE}: MISSING")
    others = sorted(
        p.name
        for p in plugins_dir.iterdir()
        if p.is_dir() and p.name != VENDOR_FOLDER and any((p / n).is_file() for n in DLL_NAMES)
    )
    if others:
        _say(f"[WARN] More CUDA Link DLLs under {plugins_dir}: {', '.join(others)}. TD loads them too.")
    return 0


def uninstall(plugins_dir: Path, dry_run: bool = False, now: _dt.datetime | None = None) -> int:
    dest = plugins_dir / VENDOR_FOLDER
    if not dest.is_dir():
        _say(f"Nothing to uninstall: {dest} does not exist.")
        return 1
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = plugins_dir.parent / f"{VENDOR_FOLDER}.removed-{stamp}"
    _say(f"Uninstall: move {dest} -> {target} (outside Plugins)")
    if dry_run:
        return 0
    shutil.move(str(dest), str(target))
    _say("[OK] Removed from Plugins. Saved projects keep their nodes; they error until you install again.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="install", choices=("install", "status", "uninstall"))
    ap.add_argument("--dll-dir", type=Path, default=DEFAULT_DLL_DIR)
    ap.add_argument("--plugins-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    plugins_dir = a.plugins_dir or default_plugins_dir()
    if a.command in ("install", "uninstall") and not a.dry_run and not a.force and td_running():
        _say("[ERROR] TouchDesigner is running. Close it first: TD locks the plugin DLLs. (--force skips this check)")
        return 3
    if a.command == "install":
        return install(plugins_dir, a.dll_dir, a.dry_run)
    if a.command == "status":
        return status(plugins_dir)
    return uninstall(plugins_dir, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
