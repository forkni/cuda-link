"""install_cpp_top.py — install the C++ CUDA Link TOPs into TouchDesigner's system Plugins folder.

Layout (docs/CPP_TOP.md, "Install: system Plugins folder"):

    <Documents>\\Derivative\\Plugins\\CudaLink\\
        CudaLinkOutTOP.dll
        CudaLinkInTOP.dll
        VERSION.txt            the install record (ADR-0016)

Commands:
    python scripts\\install_cpp_top.py install [--dll-dir DIR]   copy both DLLs, write VERSION.txt
                                    (DIR: a cpp_top build, or <site-packages>\\cuda_link\\td_plugins)
    python scripts\\install_cpp_top.py install --from-wheel      copy the DLLs of the installed wheel
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

The work is done by cuda_link.td_plugin (src/cuda_link/td_plugin.py), loaded by path so this
script needs no installed package. install_td_library.py uses the same module.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DLL_DIR = ROOT / "cpp_top" / "build"
_TD_PLUGIN_PY = ROOT / "src" / "cuda_link" / "td_plugin.py"
_INIT_PY = ROOT / "src" / "cuda_link" / "__init__.py"
_SHM_PY = ROOT / "src" / "cuda_link" / "shm_protocol.py"
_TD_HEADER = ROOT / "cpp_top" / "vendor" / "td" / "2025" / "CPlusPlus_Common.h"


def load_td_plugin(path: Path = _TD_PLUGIN_PY) -> ModuleType:
    """Load cuda_link/td_plugin.py by path (no package import, no dependencies)."""
    name = "_cuda_link_td_plugin"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec: dataclasses look the module up
    spec.loader.exec_module(module)
    return module


tp = load_td_plugin()
DLL_NAMES = tp.DLL_FILES
VENDOR_FOLDER = tp.VENDOR_FOLDER
VERSION_FILE = tp.RECORD_FILE
documents_dir = tp.documents_dir
default_plugins_dir = tp.default_plugins_dir
td_running = tp.td_running


def _grep(path: Path, pattern: str) -> str | None:
    try:
        m = re.search(pattern, path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
    except OSError:
        return None
    return m.group(1) if m else None


def protocol_text(shm_py: Path) -> str:
    """ "v0.6 (magic 0x...)" from a shm_protocol.py (the repo's, or an installed package's)."""
    magic = _grep(shm_py, r"^PROTOCOL_MAGIC[^=]*=\s*(0x[0-9A-Fa-f]+)")
    return f"v0.6 (magic {magic or 'unknown'})"


def repo_record(dll_dir: Path, now: _dt.datetime | None = None) -> object:
    """The install record for a local cpp_top build: BUILD_INFO.txt first, the repo files and
    CMakeCache.txt for what an older build has no sidecar for."""
    info = tp.read_build_info(dll_dir)
    cuda = info.get("cuda toolkit") or _grep(
        dll_dir / "CMakeCache.txt", r"^CUDAToolkit_BIN_DIR:PATH=.*?[\\/]v(\d+\.\d+)[\\/]"
    )
    return tp.make_record(
        dll_dir,
        version=info.get("cuda-link") or _grep(_INIT_PY, r"^__version__\s*=\s*[\"']([^\"']+)"),
        protocol=protocol_text(_SHM_PY),
        td_header=info.get("td header OP_CommonAPIVersion") or _grep(_TD_HEADER, r"OP_CommonAPIVersion\s*=\s*(\d+)"),
        cuda_toolkit=cuda,
        now=now,
    )


def installed_package_dir() -> Path | None:
    """The cuda_link package of the running Python's install (no import), or None."""
    spec = importlib.util.find_spec("cuda_link")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def wheel_record(package_dir: Path, now: _dt.datetime | None = None) -> object:
    """The install record for the DLLs a wheel installed in ``package_dir``."""
    dll_dir = package_dir / tp.PLUGIN_DIR_NAME
    return tp.make_record(dll_dir, protocol=protocol_text(package_dir / "shm_protocol.py"), now=now)


def install(plugins_dir: Path, dll_dir: Path, dry_run: bool = False) -> int:
    if not all((dll_dir / n).is_file() for n in DLL_NAMES):
        print(f"[ERROR] Not found in {dll_dir}: build cpp_top first or pass --dll-dir.", flush=True)
        return 2
    return tp.install(plugins_dir, dll_dir, source_record(dll_dir), dry_run=dry_run)


def source_record(dll_dir: Path) -> object:
    """The record for ``dll_dir``: an installed package's td_plugins/ folder (the wheel), or a
    cpp_top build folder."""
    package_dir = dll_dir.parent
    if dll_dir.name == tp.PLUGIN_DIR_NAME and (package_dir / "shm_protocol.py").is_file():
        return wheel_record(package_dir)
    return repo_record(dll_dir)


def install_from_wheel(plugins_dir: Path, dry_run: bool = False) -> int:
    package_dir = installed_package_dir()
    if package_dir is None:
        print("[ERROR] cuda_link is not installed in this Python.", flush=True)
        return 2
    dll_dir = package_dir / tp.PLUGIN_DIR_NAME
    if not all((dll_dir / n).is_file() for n in DLL_NAMES):
        print(f"[ERROR] The installed wheel carries no TOP DLLs ({dll_dir}): only the cp311 release wheel does.")
        return 2
    return tp.install(plugins_dir, dll_dir, wheel_record(package_dir), dry_run=dry_run)


def status(plugins_dir: Path) -> int:
    return tp.status(plugins_dir)


def uninstall(plugins_dir: Path, dry_run: bool = False, now: _dt.datetime | None = None) -> int:
    return tp.uninstall(plugins_dir, dry_run=dry_run, now=now)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="install", choices=("install", "status", "uninstall"))
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dll-dir", type=Path, default=DEFAULT_DLL_DIR)
    src.add_argument("--from-wheel", action="store_true", help="copy the DLLs of the cuda_link wheel installed here")
    ap.add_argument("--plugins-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    plugins_dir = a.plugins_dir or default_plugins_dir()
    if a.command in ("install", "uninstall") and not a.dry_run and not a.force and td_running():
        print("[ERROR] TouchDesigner is running. Close it first: TD locks the plugin DLLs. (--force skips this check)")
        return 3
    if a.command == "install":
        return (
            install_from_wheel(plugins_dir, a.dry_run) if a.from_wheel else install(plugins_dir, a.dll_dir, a.dry_run)
        )
    if a.command == "status":
        return status(plugins_dir)
    return uninstall(plugins_dir, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
