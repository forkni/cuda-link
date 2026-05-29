"""
install_td_library.py — Multi-target installer for cuda-link → TouchDesigner library mode.

Supports five installation targets so the cuda_link package is importable inside
TouchDesigner's Python environment.  The bootstrap Text DAT (CUDALinkBootstrap.py)
then uses sys.path injection to make the package available without mirror Text DATs.

Usage (interactive):
    python scripts\\install_td_library.py

Usage (non-interactive / CI):
    python scripts\\install_td_library.py --mode 1 --target "D:\\cuda_link_lib"
    python scripts\\install_td_library.py --mode 2 --venv "D:\\myproject\\.venv"
    python scripts\\install_td_library.py --mode 3 --conda base
    python scripts\\install_td_library.py --mode 4 --python "C:\\Python311\\python.exe"
    python scripts\\install_td_library.py --mode 5 --td-python "C:\\Program Files\\Derivative\\TouchDesigner\\bin\\python.exe"

Common flags:
    --non-interactive   Require all target args; skip menu and prompts.
    --dry-run           Print what would run without executing.
    --wheel <path>      Override wheel path (skips auto-detect + build).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output on Windows so box-drawing/arrow characters print correctly.
# reconfigure() is available on Python 3.7+; safe to call on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

# ─── ANSI colours (no-op on Windows consoles that don't support them) ─────────


def _c(code: str, text: str) -> str:
    if sys.stdout.isatty() and os.name != "nt":
        return f"\033[{code}m{text}\033[0m"
    return text


def _bold(t: str) -> str:
    return _c("1", t)


def _green(t: str) -> str:
    return _c("32", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _red(t: str) -> str:
    return _c("31", t)


# ─── Wheel resolution ──────────────────────────────────────────────────────────


def _find_wheel() -> Path | None:
    """Return the newest .whl in dist/, or None."""
    dist = REPO_ROOT / "dist"
    wheels = sorted(dist.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True) if dist.is_dir() else []
    return wheels[0] if wheels else None


def _build_wheel(dry_run: bool) -> Path:
    """Run build_wheel.cmd to produce a wheel; return its path."""
    print(_bold("[build] No wheel found — building now..."))
    cmd_path = REPO_ROOT / "build_wheel.cmd"
    if not cmd_path.exists():
        sys.exit(_red(f"[error] build_wheel.cmd not found at {cmd_path}"))
    if dry_run:
        print(_yellow(f"  [dry-run] Would run: {cmd_path}"))
        return REPO_ROOT / "dist" / "cuda_link-DRY_RUN.whl"
    # Use ["cmd.exe", "/c", path] instead of shell=True to avoid shell injection.
    # stdin=DEVNULL prevents build_wheel.cmd's `pause` from blocking mid-flow;
    # `pause` reads EOF immediately and passes through without waiting.
    result = subprocess.run(["cmd.exe", "/c", str(cmd_path)], cwd=REPO_ROOT, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        sys.exit(_red("[error] build_wheel.cmd failed — see output above."))
    wheel = _find_wheel()
    if not wheel:
        sys.exit(_red("[error] Build succeeded but no .whl found in dist/"))
    return wheel


def resolve_wheel(override: str | None, dry_run: bool) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(_red(f"[error] --wheel path does not exist: {p}"))
        return p
    w = _find_wheel()
    if not w:
        w = _build_wheel(dry_run)
    print(f"  Wheel: {w.name}")
    return w


# ─── pip runner ────────────────────────────────────────────────────────────────


def _run_pip(pip_args: list[str], dry_run: bool) -> None:
    cmd_str = " ".join(pip_args)
    print(f"\n  Running: {cmd_str}\n")
    if dry_run:
        print(_yellow("  [dry-run] Command not executed."))
        return
    result = subprocess.run(pip_args, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(_red("[error] pip install failed — see output above."))


def _find_site_packages_in(base: Path) -> Path | None:
    """Return the Lib/site-packages path under a Python install or venv root."""
    candidates = [
        base / "Lib" / "site-packages",  # Windows venv / installation
        base / "lib" / "site-packages",  # Unix
        base / "lib" / "python3.11" / "site-packages",
        base / "lib" / "python3.12" / "site-packages",
        base / "lib" / "python3.10" / "site-packages",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


# Python one-liner that reliably returns the site-packages directory.
# Filters site.getsitepackages() for an entry ending in 'site-packages' because some
# Windows Python 3.11 installations return the root directory as the first element instead.
_SITE_PKGS_QUERY = (
    "import site; "
    "all_paths = site.getsitepackages(); "
    "sp = next((p for p in all_paths if p.lower().endswith('site-packages')), all_paths[0]); "
    "print(sp)"
)


def _print_activation(site_packages: Path | None, label: str, td_preferences_only: bool = False) -> None:
    """Print installation confirmation and path-activation instructions.

    td_preferences_only=True (modes 2/3/4): show only the TD Preferences path instruction.
    td_preferences_only=False (mode 1 default): show CUDALINK_LIB_PATH + TD Preferences.
    """
    print()
    print(_bold("─" * 60))
    print(_bold("  INSTALL COMPLETE"))
    print(_bold("─" * 60))
    if site_packages:
        print(f"\n  {label}:")
        print(f"    {site_packages}")
        print()
        if td_preferences_only:
            print(_bold("  Add to TouchDesigner Preferences:"))
            print("       Edit → Preferences → Python 32/64 bit Module Path")
            print(_green(f"       Add:  {site_packages}"))
        else:
            print(_bold("  Activate — choose one method:"))
            print()
            print("  1. Environment variable (quickest, per-session):")
            print(_green(f"       SET CUDALINK_LIB_PATH={site_packages}"))
            print()
            print("  2. Permanent (Windows Environment Variables):")
            print("       Variable:  CUDALINK_LIB_PATH")
            print(f"       Value:     {site_packages}")
            print()
            print("  3. TD Preferences (persists per TD install, no env var needed):")
            print("       Edit → Preferences → Python 32/64 bit Module Path")
            print(f"       Add:  {site_packages}")
        print()
        print(_bold("  Then verify in the TD Textport after loading your .toe:"))
        print("    [CUDALinkBootstrap] Library mode active — cuda_link submodules aliased")
    print()


# ─── Install modes ─────────────────────────────────────────────────────────────


def mode_1_external_folder(wheel: Path, target: str | None, non_interactive: bool, dry_run: bool) -> None:
    """pip install --target <folder>  (default; CUDALINK_LIB_PATH points here)."""
    if not target:
        if non_interactive:
            sys.exit(_red("[error] --mode 1 requires --target <dir>"))
        print("\n  Install cuda_link into a standalone folder.")
        print("  Recommendations:")
        print("    D:\\cuda_link_lib")
        print(f"    {Path.home() / 'AppData' / 'Local' / 'cuda_link_lib'}")
        target = input("\n  Target folder: ").strip()
        if not target:
            sys.exit(_red("[error] No target folder specified."))

    dest = Path(target)
    dest.mkdir(parents=True, exist_ok=True)
    pip = [sys.executable, "-m", "pip", "install", "--target", str(dest), str(wheel), "--upgrade"]
    _run_pip(pip, dry_run)
    _print_activation(dest, "Install folder")


def mode_2_venv(wheel: Path, venv_dir: str | None, non_interactive: bool, dry_run: bool) -> None:
    """Install into an existing venv's site-packages."""
    if not venv_dir:
        if non_interactive:
            sys.exit(_red("[error] --mode 2 requires --venv <venv_dir>"))
        venv_dir = input("\n  Path to existing venv directory: ").strip()
        if not venv_dir:
            sys.exit(_red("[error] No venv path specified."))

    venv = Path(venv_dir)
    # Locate pip inside the venv
    pip_exe = venv / "Scripts" / "pip.exe"
    if not pip_exe.exists():
        pip_exe = venv / "bin" / "pip"
    if not pip_exe.exists():
        sys.exit(_red(f"[error] pip not found in venv at {venv / 'Scripts'} or {venv / 'bin'}"))

    pip = [str(pip_exe), "install", str(wheel), "--upgrade"]
    _run_pip(pip, dry_run)
    site_pkgs = _find_site_packages_in(venv)
    _print_activation(site_pkgs, "venv site-packages", td_preferences_only=True)


def mode_3_conda(wheel: Path, conda_env: str | None, non_interactive: bool, dry_run: bool) -> None:
    """Install into a conda environment."""
    if not conda_env:
        if non_interactive:
            sys.exit(_red("[error] --mode 3 requires --conda <env_name>"))
        conda_env = input("\n  Conda environment name (e.g. 'base' or 'myenv'): ").strip()
        if not conda_env:
            sys.exit(_red("[error] No conda environment name specified."))

    pip = ["conda", "run", "-n", conda_env, "pip", "install", str(wheel), "--upgrade"]
    _run_pip(pip, dry_run)

    # Try to find the site-packages path from conda info
    site_pkgs: Path | None = None
    if not dry_run:
        try:
            info = subprocess.run(
                ["conda", "run", "-n", conda_env, "python", "-c", _SITE_PKGS_QUERY],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            if info.returncode == 0:
                site_pkgs = Path(info.stdout.strip())
        except FileNotFoundError:
            pass

    _print_activation(site_pkgs, "conda env site-packages", td_preferences_only=True)
    if site_pkgs is None:
        print(_yellow("  Could not auto-detect conda site-packages path."))
        print('  Run: conda run -n <env> python -c "import site; print(site.getsitepackages())"')
        print("  then add the 'site-packages' entry to TD Preferences.")


def mode_4_system_python(wheel: Path, python_exe: str | None, non_interactive: bool, dry_run: bool) -> None:
    """Install into a system / parallel Python installation (the official TD docs approach)."""
    if not python_exe:
        if non_interactive:
            sys.exit(_red("[error] --mode 4 requires --python <python_exe>"))
        print("\n  Install into a parallel Python 3.11 installation.")
        print("  This is the method documented by Derivative.")
        print("  Install Python 3.11 from https://www.python.org and use its path here.")
        python_exe = input("\n  Path to python.exe (e.g. C:\\Python311\\python.exe): ").strip()
        if not python_exe:
            sys.exit(_red("[error] No Python executable specified."))

    py_path = Path(python_exe)
    # If the user pasted a directory (e.g. from Explorer), auto-append the executable name.
    if py_path.is_dir():
        exe_name = "python.exe" if sys.platform == "win32" else "python"
        py_path = py_path / exe_name
        print(_yellow(f"  (directory given — resolved to: {py_path})"))
    if not py_path.exists():
        sys.exit(_red(f"[error] Python executable not found: {py_path}"))
    if not py_path.is_file():
        sys.exit(_red(f"[error] Path is not an executable file: {py_path}"))

    pip = [str(py_path), "-m", "pip", "install", str(wheel), "--upgrade"]
    _run_pip(pip, dry_run)

    # Detect site-packages — use _SITE_PKGS_QUERY to filter for the actual site-packages
    # directory; getsitepackages()[0] can return the Python root on some installations.
    site_pkgs: Path | None = None
    if not dry_run:
        try:
            info = subprocess.run(
                [str(py_path), "-c", _SITE_PKGS_QUERY],
                capture_output=True,
                text=True,
            )
            if info.returncode == 0:
                site_pkgs = Path(info.stdout.strip())
        except (FileNotFoundError, OSError):
            pass

    _print_activation(site_pkgs, "Python site-packages", td_preferences_only=True)


def mode_5_td_python(wheel: Path, td_python_exe: str | None, non_interactive: bool, dry_run: bool) -> None:
    """Install directly into TouchDesigner's bundled Python.

    WARNING: Modifies TD's internal Python environment. Runs without needing CUDALINK_LIB_PATH.
    Obtain the path from TD Textport: print(app.pythonExecutable)
    """
    if not td_python_exe:
        if non_interactive:
            sys.exit(_red("[error] --mode 5 requires --td-python <td_python_exe>"))
        print()
        print(_bold("  Install into TouchDesigner's own Python."))
        print(_yellow("  WARNING: This modifies TD's bundled Python environment."))
        print("  Get the executable path from TD's Textport: print(app.pythonExecutable)")
        print("  Default location: C:\\Program Files\\Derivative\\TouchDesigner\\bin\\python.exe")
        td_python_exe = input("\n  TD Python executable path: ").strip()
        if not td_python_exe:
            sys.exit(_red("[error] No TD Python executable specified."))

    td_py = Path(td_python_exe)
    # If the user pasted a directory, auto-append the executable name.
    if td_py.is_dir():
        exe_name = "python.exe" if sys.platform == "win32" else "python"
        td_py = td_py / exe_name
        print(_yellow(f"  (directory given — resolved to: {td_py})"))
    if not td_py.exists():
        sys.exit(_red(f"[error] TD Python executable not found: {td_py}"))
    if not td_py.is_file():
        sys.exit(_red(f"[error] Path is not an executable file: {td_py}"))

    pip = [str(td_py), "-m", "pip", "install", str(wheel), "--upgrade"]

    print(_yellow("\n  Note: installing into TD's Python means CUDALINK_LIB_PATH is NOT needed."))
    print("  cuda_link will be importable in all TD projects automatically.")
    print("  Re-run this after upgrading TouchDesigner.\n")
    _run_pip(pip, dry_run)

    # TD's Python does not need a path variable — it's already on TD's sys.path
    print()
    print(_bold("─" * 60))
    print(_bold("  INSTALL COMPLETE (TD Python)"))
    print(_bold("─" * 60))
    print()
    print("  cuda_link is now installed into TD's Python.")
    print("  No CUDALINK_LIB_PATH or TD Preferences change needed.")
    print()
    print(_bold("  Verify in TD Textport after restarting TouchDesigner:"))
    print("    import cuda_link; print(cuda_link.__version__)")
    print()


# ─── Interactive menu ──────────────────────────────────────────────────────────

_MODE_DESCRIPTIONS = {
    1: "External folder (pip --target)   -> set CUDALINK_LIB_PATH=<folder>",
    2: "Existing venv                     -> set CUDALINK_LIB_PATH=<venv/Lib/site-packages>",
    3: "Conda environment                 -> set CUDALINK_LIB_PATH=<conda-env/Lib/site-packages>",
    4: "System / parallel Python 3.11    -> add site-packages to TD Preferences",
    5: "TouchDesigner's own Python        -> no env var needed (modifies TD's Python)",
}


def _interactive_menu() -> int:
    print()
    print(_bold("  Select installation target:"))
    print()
    for num, desc in _MODE_DESCRIPTIONS.items():
        print(f"    {_bold(str(num))}) {desc}")
    print()
    while True:
        choice = input("  Enter mode (1-5): ").strip()
        if choice in ("1", "2", "3", "4", "5"):
            return int(choice)
        print(_red("  Invalid choice — enter a number from 1 to 5."))


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install cuda-link into a Python environment accessible from TouchDesigner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=int, choices=[1, 2, 3, 4, 5], help="Installation target mode (1-5); interactive menu if omitted."
    )
    parser.add_argument("--target", metavar="DIR", help="Mode 1: destination folder for pip --target.")
    parser.add_argument("--venv", metavar="DIR", help="Mode 2: path to an existing venv.")
    parser.add_argument("--conda", metavar="ENV", help="Mode 3: conda environment name.")
    parser.add_argument("--python", metavar="EXE", help="Mode 4: path to python.exe of a parallel install.")
    parser.add_argument(
        "--td-python", metavar="EXE", help="Mode 5: path to TD's python.exe (see app.pythonExecutable in Textport)."
    )
    parser.add_argument("--wheel", metavar="PATH", help="Override wheel path (skip auto-detect + build).")
    parser.add_argument("--non-interactive", action="store_true", help="Require explicit flags; skip all prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    print()
    print(_bold("=" * 60))
    print(_bold("  cuda-link TD Library Installer"))
    print(_bold("=" * 60))
    if args.dry_run:
        print(_yellow("  [DRY-RUN MODE] — no commands will be executed."))
    print()

    # Resolve wheel
    wheel = resolve_wheel(args.wheel, args.dry_run)

    # Determine mode
    mode = args.mode
    if mode is None:
        if args.non_interactive:
            sys.exit(_red("[error] --non-interactive requires --mode."))
        mode = _interactive_menu()

    print(f"\n  Mode {mode}: {_MODE_DESCRIPTIONS[mode]}")

    # Dispatch
    if mode == 1:
        mode_1_external_folder(wheel, args.target, args.non_interactive, args.dry_run)
    elif mode == 2:
        mode_2_venv(wheel, args.venv, args.non_interactive, args.dry_run)
    elif mode == 3:
        mode_3_conda(wheel, args.conda, args.non_interactive, args.dry_run)
    elif mode == 4:
        mode_4_system_python(wheel, args.python, args.non_interactive, args.dry_run)
    elif mode == 5:
        mode_5_td_python(wheel, args.td_python, args.non_interactive, args.dry_run)


if __name__ == "__main__":
    main()
