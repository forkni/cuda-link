"""
install_td_library.py — Multi-target installer for cuda-link → TouchDesigner library mode.

Supports five installation targets so the cuda_link package is importable inside
TouchDesigner's Python environment.  The bootstrap Text DAT (CUDALinkBootstrap.py)
then uses sys.path injection to make the package available without mirror Text DATs.

Pass --spout to also install the cuda-link-spout Spout bridge into the same target.
The spout wheel is auto-built on demand via utils\\build_spout_wheel.cmd when missing
(requires the Spout2 SDK, CUDA 12.x/13.x, and a C++17 compiler; see spout/README.md).

The cuda-link-native wait-backend accelerator installs BY DEFAULT (pass --no-native
to skip it) — auto-built on demand via utils\\build_native_wheel.cmd when missing
(requires only a C++17 compiler; no CUDA Toolkit, no SDK; see native/README.md). If
the build fails (e.g. no MSVC found), the installer warns and continues with just
the core wheel — this is a soft default, never a hard install failure.

Usage (interactive):
    python scripts\\install_td_library.py

Usage (non-interactive / CI):
    python scripts\\install_td_library.py --mode 1 --target "D:\\cuda_link_lib"
    python scripts\\install_td_library.py --mode 2 --venv "D:\\myproject\\.venv"
    python scripts\\install_td_library.py --mode 3 --conda base
    python scripts\\install_td_library.py --mode 4 --python "C:\\Python311\\python.exe"
    python scripts\\install_td_library.py --mode 5 --td-python "C:\\Program Files\\Derivative\\TouchDesigner\\bin\\python.exe"

Native wait backend (on by default, auto-builds on demand):
    python scripts\\install_td_library.py --mode 5 --td-python "..."             # native installs automatically
    python scripts\\install_td_library.py --mode 5 --td-python "..." --no-native # skip it

Spout bridge (optional, auto-builds on demand):
    python scripts\\install_td_library.py --mode 5 --td-python "..." --spout
    python scripts\\install_td_library.py --mode 5 --td-python "..." --no-spout   # suppress interactive prompt

Common flags:
    --non-interactive   Require all target args; skip menu and prompts.
    --dry-run           Print what would run without executing.
    --wheel <path>      Override core wheel path (skips auto-detect + build).
    --native-wheel <path>  Override native wheel path (implies --native).
    --spout-wheel <path>  Override spout wheel path (implies --spout).

Environment variables (persisted via `setx`, current user, on by default):
    CUDALINK_DOORBELL=1 is set after ANY successful install (all modes).
    CUDALINK_RECEIVER_PYTHON_EXE=<python_exe> is also set after modes 2 (venv)
    and 4 (system/parallel Python) — whichever interpreter
    example_receiver_launcher.py's standalone receiver subprocess should use.
    Pass --no-set-env to skip both. Note: setx only affects NEW processes —
    restart your terminal / TouchDesigner before the change applies.
"""

from __future__ import annotations

import argparse
import os
import re
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
    """Return the newest cuda_link-*.whl (core) in dist/, or None."""
    dist = REPO_ROOT / "dist"
    # Scope to core wheels only: "cuda_link-*.whl" never matches "cuda_link_spout-*.whl"
    # because the latter has "_spout" immediately after "cuda_link" (underscore, not dash).
    wheels = (
        sorted(dist.glob("cuda_link-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True) if dist.is_dir() else []
    )
    return wheels[0] if wheels else None


def _find_spout_wheel() -> Path | None:
    """Return the newest cuda_link_spout-*.whl in dist/, or None."""
    dist = REPO_ROOT / "dist"
    wheels = (
        sorted(dist.glob("cuda_link_spout-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if dist.is_dir()
        else []
    )
    return wheels[0] if wheels else None


def _find_native_wheel() -> Path | None:
    """Return the newest cuda_link_native-*.whl in dist/, or None."""
    dist = REPO_ROOT / "dist"
    wheels = (
        sorted(dist.glob("cuda_link_native-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if dist.is_dir()
        else []
    )
    return wheels[0] if wheels else None


# Repo-relative source roots whose changes should invalidate each wheel.
_CORE_SOURCE_ROOTS = ("src/cuda_link", "pyproject.toml")
_SPOUT_SOURCE_ROOTS = ("spout/src", "spout/CMakeLists.txt", "spout/pyproject.toml")
_NATIVE_SOURCE_ROOTS = ("native/src", "native/CMakeLists.txt", "native/pyproject.toml")

# Build artifact dirs to skip when walking source roots for mtimes — a wheel must
# never be able to count itself (or a stale build/ copy) as its own source.
_ARTIFACT_DIRS = {"build", "build_ninja", "dist", "__pycache__"}


def _newest_source_mtime(roots: tuple[str, ...], _root: Path = REPO_ROOT) -> float:
    """Newest mtime (epoch secs) across source files under the given repo-relative roots.

    Directories are walked recursively; build artifacts (build/, build_ninja/, dist/,
    __pycache__/, *.egg-info/) are skipped. Returns 0.0 when nothing matches, which
    callers treat as "not newer than the wheel" — i.e. no spurious rebuild.

    `_root` overrides the repo root; only tests should pass it (production callers
    always resolve against the real REPO_ROOT).
    """
    newest = 0.0
    for rel in roots:
        p = _root / rel
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
        elif p.is_dir():
            for f in p.rglob("*"):
                if not f.is_file():
                    continue
                parts = set(f.relative_to(p).parts)
                if parts & _ARTIFACT_DIRS or f.parent.name.endswith(".egg-info"):
                    continue
                newest = max(newest, f.stat().st_mtime)
    return newest


def _wheel_is_stale(wheel: Path, roots: tuple[str, ...], _root: Path = REPO_ROOT) -> bool:
    """True if any source file under `roots` is newer than the built `wheel`.

    mtime is a pragmatic signal, not a content hash: a `git checkout`/clone can
    rewrite source mtimes and trigger a spurious rebuild, but that is safe
    (correct-but-slower) — never a wrong install. Use the --wheel/--spout-wheel/
    --native-wheel override flags to force-reuse a specific wheel unconditionally.

    `_root` overrides the repo root; only tests should pass it.
    """
    return _newest_source_mtime(roots, _root=_root) > wheel.stat().st_mtime


def _build_wheel(dry_run: bool) -> Path:
    """Run build_wheel.cmd to produce a wheel; return its path."""
    print(_bold("[build] No wheel found — building now..."))
    cmd_path = REPO_ROOT / "utils" / "build_wheel.cmd"
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


def _build_spout_wheel(dry_run: bool) -> Path:
    """Run build_spout_wheel.cmd to produce a spout wheel; return its path."""
    print(_bold("[build] No spout wheel found — building now..."))
    cmd_path = REPO_ROOT / "utils" / "build_spout_wheel.cmd"
    if not cmd_path.exists():
        sys.exit(_red(f"[error] build_spout_wheel.cmd not found at {cmd_path}"))
    if dry_run:
        print(_yellow(f"  [dry-run] Would run: {cmd_path}"))
        return REPO_ROOT / "dist" / "cuda_link_spout-DRY_RUN.whl"
    # Use ["cmd.exe", "/c", path] instead of shell=True to avoid shell injection.
    # stdin=DEVNULL prevents build_spout_wheel.cmd's `pause` from blocking mid-flow;
    # `pause` reads EOF immediately and passes through without waiting.
    result = subprocess.run(["cmd.exe", "/c", str(cmd_path)], cwd=REPO_ROOT, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        sys.exit(
            _red(
                "[error] Spout wheel build failed.\n"
                "        Common causes: missing Spout2 SDK, CUDA Toolkit, or C++17 compiler.\n"
                "        See spout/README.md for prerequisites:\n"
                "          git clone --depth 1 https://github.com/leadedge/Spout2 C:\\src\\Spout2\n"
                "        Then retry — or run utils\\build_spout_wheel.cmd directly for full output."
            )
        )
    wheel = _find_spout_wheel()
    if not wheel:
        sys.exit(_red("[error] Build succeeded but no cuda_link_spout-*.whl found in dist/"))
    return wheel


def _build_native_wheel(dry_run: bool) -> Path | None:
    """Run build_native_wheel.cmd to produce a native wheel; return its path, or None on failure.

    Unlike _build_spout_wheel (which sys.exit()s on failure — spout is an explicit
    opt-in), this degrades gracefully: native installs by default, so a build
    failure here (e.g. no MSVC found) must not block the core install. The caller
    prints a warning and continues with the core wheel only.
    """
    print(_bold("[build] No native wheel found — building now..."))
    cmd_path = REPO_ROOT / "utils" / "build_native_wheel.cmd"
    if not cmd_path.exists():
        print(_yellow(f"  [warn] build_native_wheel.cmd not found at {cmd_path} — skipping native backend."))
        return None
    if dry_run:
        print(_yellow(f"  [dry-run] Would run: {cmd_path}"))
        return REPO_ROOT / "dist" / "cuda_link_native-DRY_RUN.whl"
    # Use ["cmd.exe", "/c", path] instead of shell=True to avoid shell injection.
    # stdin=DEVNULL prevents build_native_wheel.cmd's `pause` from blocking mid-flow;
    # `pause` reads EOF immediately and passes through without waiting.
    result = subprocess.run(["cmd.exe", "/c", str(cmd_path)], cwd=REPO_ROOT, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        print(
            _yellow(
                "  [warn] Native wheel build failed — continuing without the native wait backend.\n"
                "         Common cause: no C++17 compiler (MSVC) found.\n"
                "         See native/README.md, or run utils\\build_native_wheel.cmd directly for full output.\n"
                "         The core wheel alone still works — Importer falls back to its Python wait path."
            )
        )
        return None
    wheel = _find_native_wheel()
    if not wheel:
        print(_yellow("  [warn] Build reported success but no cuda_link_native-*.whl found in dist/ — skipping."))
        return None
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
    elif _wheel_is_stale(w, _CORE_SOURCE_ROOTS):
        print(_yellow(f"  [stale] {w.name} predates src/cuda_link changes — rebuilding..."))
        w = _build_wheel(dry_run)
    print(f"  Wheel: {w.name}")
    return w


def resolve_spout_wheel(override: str | None, dry_run: bool) -> Path:
    """Locate or auto-build the spout wheel; return its path."""
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(_red(f"[error] --spout-wheel path does not exist: {p}"))
        return p
    w = _find_spout_wheel()
    if not w:
        w = _build_spout_wheel(dry_run)
    elif _wheel_is_stale(w, _SPOUT_SOURCE_ROOTS):
        print(_yellow(f"  [stale] {w.name} predates spout/ changes — rebuilding..."))
        w = _build_spout_wheel(dry_run)
    print(f"  Spout wheel: {w.name}")
    return w


def resolve_native_wheel(override: str | None, dry_run: bool) -> Path | None:
    """Locate or auto-build the native wheel; return its path, or None to skip gracefully.

    Unlike resolve_spout_wheel (fatal on failure — spout is explicit opt-in), a
    missing/failed native wheel here is not fatal: native is a soft default.
    """
    if override:
        p = Path(override)
        if not p.exists():
            print(_yellow(f"  [warn] --native-wheel path does not exist: {p} — skipping native backend."))
            return None
        return p
    w = _find_native_wheel()
    if not w:
        w = _build_native_wheel(dry_run)
    elif _wheel_is_stale(w, _NATIVE_SOURCE_ROOTS):
        print(_yellow(f"  [stale] {w.name} predates native/ changes — rebuilding..."))
        w = _build_native_wheel(dry_run) or w  # keep the stale wheel if rebuild fails
    if w:
        print(f"  Native wheel: {w.name}")
    return w


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    """Prompt the user for a yes/no answer; returns default on EOF or blank input."""
    hint = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {hint} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


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


def _setx_user(name: str, value: str, dry_run: bool) -> bool:
    """Persist a current-user Windows environment variable via `setx` (no admin needed).

    Soft-failure like the native wheel auto-build: a setx problem (e.g. the
    combined-length limit, or setx being unavailable on a non-Windows dev
    environment) warns and returns False rather than aborting the whole install.

    NOTE: setx only affects NEW processes started after it runs — the current
    shell/TD session will not see the new value until restarted. Callers are
    responsible for telling the user this (see _print_env_vars_set below).
    """
    cmd_str = f'setx {name} "{value}"'
    print(f"  Running: {cmd_str}")
    if dry_run:
        print(_yellow("    [dry-run] Command not executed."))
        return True
    if sys.platform != "win32":
        print(_yellow(f"    [skip] setx is Windows-only; not setting {name} on this platform."))
        return False
    try:
        result = subprocess.run(["setx", name, value], capture_output=True, text=True)
    except OSError as e:
        print(_yellow(f"    [warn] Could not run setx for {name}: {e}"))
        return False
    if result.returncode != 0:
        print(_yellow(f"    [warn] setx {name} failed (exit {result.returncode}): {result.stderr.strip()}"))
        return False
    return True


def _print_env_vars_set(set_vars: dict[str, str]) -> None:
    """Print a clearly-labeled confirmation block for env vars just persisted via setx.

    Deliberately separate from _print_activation's CUDALINK_LIB_PATH instructions
    so it isn't mistaken for them — these are a different mechanism (already
    done, not something the user needs to do) for a different purpose.
    """
    if not set_vars:
        return
    print()
    print(_bold("  Environment variables set (current user):"))
    for name, value in set_vars.items():
        print(_green(f"    {name}={value}"))
    print(_yellow("  Note: these only take effect in NEW processes — restart your terminal"))
    print(_yellow("  and/or TouchDesigner before they apply. Use --no-set-env to skip this."))


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


def _discover_system_pythons() -> list[tuple[str, Path]]:
    """Return registered Python installations via the Windows `py` launcher.

    Uses `py --list-paths` which outputs lines like:
        -V:3.11 *        C:\\...\\Python311\\python.exe
    Returns [(version_label, exe_path), ...], newest/default first.
    Falls back to [] if the launcher is unavailable (non-Windows or py not installed).
    """
    try:
        out = subprocess.run(["py", "--list-paths"], capture_output=True, text=True, timeout=5)
        results: list[tuple[str, Path]] = []
        for line in out.stdout.splitlines():
            m = re.match(r"\s*-V:([\d.]+)\s*(\*)?\s+(.*python(?:\.exe)?)", line, re.IGNORECASE)
            if m:
                ver = m.group(1) + (" (default)" if m.group(2) else "")
                exe = Path(m.group(3).strip())
                if exe.is_file():
                    results.append((ver, exe))
        return results
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []


def _discover_td_pythons() -> list[tuple[str, Path]]:
    """Return python.exe paths found inside TouchDesigner installations.

    Globs C:/Program Files/Derivative/TouchDesigner*/bin/python.exe.
    Returns [(td_dir_name, exe_path), ...] sorted newest-first by directory mtime.
    Falls back to [] if the Derivative directory does not exist.
    """
    base = Path("C:/Program Files/Derivative")
    if not base.is_dir():
        return []
    results: list[tuple[str, Path]] = []
    for td_dir in sorted(base.glob("TouchDesigner*"), key=lambda p: p.stat().st_mtime, reverse=True):
        py = td_dir / "bin" / "python.exe"
        if py.is_file():
            results.append((td_dir.name, py))
    return results


def _pick_from_list(
    items: list[tuple[str, Path]],
    prompt_header: str,
    item_label: str,
) -> Path | None:
    """Print a numbered list and return the chosen Path, or None to enter manually.

    Returns None if the list is empty or if the user picks option 0 (manual entry).
    Auto-selects without prompting when exactly one item is found.
    """
    if not items:
        return None
    print(f"\n  {prompt_header}")
    for i, (label, exe) in enumerate(items, start=1):
        print(f"    {i}) {label:<30}  {exe}")
    print(f"    0) Enter {item_label} path manually")

    if len(items) == 1:
        print(f"\n  (only one found — auto-selecting: {items[0][1]})")
        return items[0][1]

    while True:
        try:
            choice = input("\n  Select (default 1): ").strip() or "1"
        except EOFError:
            choice = "1"
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice) - 1][1]
        print(_red(f"  Invalid — enter a number from 0 to {len(items)}."))


# Python one-liner that reliably returns the site-packages directory.
# Filters site.getsitepackages() for an entry ending in 'site-packages' because some
# Windows Python 3.11 installations return the root directory as the first element instead.
_SITE_PKGS_QUERY = (
    "import site; "
    "all_paths = site.getsitepackages(); "
    "sp = next((p for p in all_paths if p.lower().endswith('site-packages')), all_paths[0]); "
    "print(sp)"
)


def _extra_packages_from_wheels(wheels: list[Path]) -> list[str]:
    """Map installed wheel filenames to their extra (non-core) import names.

    "Extra" excludes the core cuda_link-*.whl (always present, never listed
    separately) — used to build verification/activation hints that scale to
    any combination of optional packages (native, spout, future additions),
    rather than a single with_spout-style boolean that stopped generalizing
    once a third optional wheel (native) was added.
    """
    names = []
    for w in wheels:
        if w.name.startswith("cuda_link_native-"):
            names.append("cuda_link_native")
        elif w.name.startswith("cuda_link_spout-"):
            names.append("cuda_link_spout")
    return names


def _print_activation(
    site_packages: Path | None,
    label: str,
    td_preferences_only: bool = False,
    extra_packages: list[str] | None = None,
) -> None:
    """Print installation confirmation and path-activation instructions.

    td_preferences_only=True (modes 2/3/4): show only the TD Preferences path instruction.
    td_preferences_only=False (mode 1 default): show CUDALINK_LIB_PATH + TD Preferences.
    extra_packages: optional-package import names installed alongside cuda_link
    (e.g. ["cuda_link_native"]) — each gets its own verify-import hint line.
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
        for pkg in extra_packages or []:
            print(f"    import {pkg}; print({pkg}.__version__)")
    print()


# ─── Install modes ─────────────────────────────────────────────────────────────


def mode_1_external_folder(wheels: list[Path], target: str | None, non_interactive: bool, dry_run: bool) -> None:
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
    pip = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(dest),
        *[str(w) for w in wheels],
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
    ]
    _run_pip(pip, dry_run)
    _print_activation(dest, "Install folder", extra_packages=_extra_packages_from_wheels(wheels))


def mode_2_venv(
    wheels: list[Path], venv_dir: str | None, non_interactive: bool, dry_run: bool, set_env: bool = True
) -> None:
    """Install into an existing venv's site-packages."""
    if not venv_dir:
        if non_interactive:
            sys.exit(_red("[error] --mode 2 requires --venv <venv_dir>"))
        venv_dir = input("\n  Path to existing venv directory: ").strip()
        if not venv_dir:
            sys.exit(_red("[error] No venv path specified."))

    venv = Path(venv_dir)
    # Locate pip (and python) inside the venv
    pip_exe = venv / "Scripts" / "pip.exe"
    python_exe = venv / "Scripts" / "python.exe"
    if not pip_exe.exists():
        pip_exe = venv / "bin" / "pip"
        python_exe = venv / "bin" / "python"
    if not pip_exe.exists():
        sys.exit(_red(f"[error] pip not found in venv at {venv / 'Scripts'} or {venv / 'bin'}"))

    pip = [str(pip_exe), "install", *[str(w) for w in wheels], "--upgrade", "--force-reinstall", "--no-deps"]
    _run_pip(pip, dry_run)
    site_pkgs = _find_site_packages_in(venv)
    _print_activation(
        site_pkgs, "venv site-packages", td_preferences_only=True, extra_packages=_extra_packages_from_wheels(wheels)
    )

    if set_env and python_exe.exists():
        # Same rationale as mode 4: this venv's python.exe is the interpreter
        # example_receiver_launcher.py's standalone receiver subprocess should
        # use, since cuda_link was just installed into it.
        print()
        print(_bold("  Persisting environment variable for the standalone receiver:"))
        set_vars: dict[str, str] = {}
        if _setx_user("CUDALINK_RECEIVER_PYTHON_EXE", str(python_exe), dry_run):
            set_vars["CUDALINK_RECEIVER_PYTHON_EXE"] = str(python_exe)
        _print_env_vars_set(set_vars)


def mode_3_conda(wheels: list[Path], conda_env: str | None, non_interactive: bool, dry_run: bool) -> None:
    """Install into a conda environment."""
    if not conda_env:
        if non_interactive:
            sys.exit(_red("[error] --mode 3 requires --conda <env_name>"))
        conda_env = input("\n  Conda environment name (e.g. 'base' or 'myenv'): ").strip()
        if not conda_env:
            sys.exit(_red("[error] No conda environment name specified."))

    pip = [
        "conda",
        "run",
        "-n",
        conda_env,
        "pip",
        "install",
        *[str(w) for w in wheels],
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
    ]
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

    _print_activation(
        site_pkgs,
        "conda env site-packages",
        td_preferences_only=True,
        extra_packages=_extra_packages_from_wheels(wheels),
    )
    if site_pkgs is None:
        print(_yellow("  Could not auto-detect conda site-packages path."))
        print('  Run: conda run -n <env> python -c "import site; print(site.getsitepackages())"')
        print("  then add the 'site-packages' entry to TD Preferences.")


def mode_4_system_python(
    wheels: list[Path], python_exe: str | None, non_interactive: bool, dry_run: bool, set_env: bool = True
) -> None:
    """Install into a system / parallel Python installation (the official TD docs approach)."""
    if not python_exe:
        if non_interactive:
            sys.exit(_red("[error] --mode 4 requires --python <python_exe>"))
        print("\n  Install into a system / parallel Python installation.")
        print("  This is the method documented by Derivative (TD uses Python 3.11).")
        discovered = _discover_system_pythons()
        picked = _pick_from_list(discovered, "Found Python installations:", "Python")
        if picked is not None:
            python_exe = str(picked)
        else:
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

    pip = [
        str(py_path),
        "-m",
        "pip",
        "install",
        *[str(w) for w in wheels],
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
    ]
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

    _print_activation(
        site_pkgs, "Python site-packages", td_preferences_only=True, extra_packages=_extra_packages_from_wheels(wheels)
    )

    if set_env:
        # CUDALINK_RECEIVER_PYTHON_EXE: the interpreter example_receiver_launcher.py's
        # standalone receiver subprocess should use — this IS that interpreter, since
        # mode 4 just installed cuda_link into it. Without this, the launcher falls
        # back to `py -3` resolution, which may pick a DIFFERENT Python 3 install.
        # (CUDALINK_DOORBELL is set centrally in main() after any mode succeeds.)
        print()
        print(_bold("  Persisting environment variable for the standalone receiver:"))
        set_vars: dict[str, str] = {}
        if _setx_user("CUDALINK_RECEIVER_PYTHON_EXE", str(py_path), dry_run):
            set_vars["CUDALINK_RECEIVER_PYTHON_EXE"] = str(py_path)
        _print_env_vars_set(set_vars)


def mode_5_td_python(wheels: list[Path], td_python_exe: str | None, non_interactive: bool, dry_run: bool) -> None:
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
        print("  (Re-run this after upgrading TouchDesigner.)")
        discovered = _discover_td_pythons()
        picked = _pick_from_list(discovered, "Found TouchDesigner installations:", "TD Python")
        if picked is not None:
            td_python_exe = str(picked)
        else:
            print("  Tip: get the exact path from TD's Textport: print(app.pythonExecutable)")
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

    pip = [str(td_py), "-m", "pip", "install", *[str(w) for w in wheels], "--upgrade", "--force-reinstall", "--no-deps"]

    extra_pkgs = _extra_packages_from_wheels(wheels)
    all_pkgs = ["cuda_link", *extra_pkgs]
    plural = len(all_pkgs) > 1
    pkgs_label = " + ".join(all_pkgs)
    print(_yellow(f"\n  Note: installing {pkgs_label} into TD's Python means CUDALINK_LIB_PATH is NOT needed."))
    print(f"  {pkgs_label} will be importable in all TD projects automatically.")
    print("  Re-run this after upgrading TouchDesigner.\n")
    _run_pip(pip, dry_run)

    # TD's Python does not need a path variable — it's already on TD's sys.path
    verify_stmt = f"import {', '.join(all_pkgs)}; print(cuda_link.__version__)"
    print()
    print(_bold("─" * 60))
    print(_bold("  INSTALL COMPLETE (TD Python)"))
    print(_bold("─" * 60))
    print()
    print(f"  {pkgs_label} {'are' if plural else 'is'} now installed into TD's Python.")
    print("  No CUDALINK_LIB_PATH or TD Preferences change needed.")
    print()
    print(_bold("  Verify in TD Textport after restarting TouchDesigner:"))
    print(f"    {verify_stmt}")
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
        description="Install cuda-link (+ optional Spout bridge) into a Python environment accessible from TouchDesigner.",
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
    parser.add_argument("--wheel", metavar="PATH", help="Override core wheel path (skip auto-detect + build).")
    native_grp = parser.add_mutually_exclusive_group()
    native_grp.add_argument(
        "--native",
        action="store_true",
        default=False,
        help="Install the cuda-link-native wait-backend accelerator (default: on; this flag is "
        "only needed to force it back on after --no-native, or to be explicit).",
    )
    native_grp.add_argument(
        "--no-native",
        action="store_true",
        default=False,
        help="Skip the native wait-backend accelerator (it installs by default otherwise).",
    )
    parser.add_argument(
        "--native-wheel",
        metavar="PATH",
        help="Override native wheel path (implies --native; skip auto-detect).",
    )
    spout_grp = parser.add_mutually_exclusive_group()
    spout_grp.add_argument(
        "--spout",
        action="store_true",
        default=False,
        help="Also install the cuda-link-spout Spout bridge into the same target (requires dist/cuda_link_spout-*.whl).",
    )
    spout_grp.add_argument(
        "--no-spout",
        action="store_true",
        default=False,
        help="Skip the Spout bridge even when prompted interactively.",
    )
    parser.add_argument(
        "--spout-wheel",
        metavar="PATH",
        help="Override spout wheel path (implies --spout; skip auto-detect).",
    )
    parser.add_argument("--non-interactive", action="store_true", help="Require explicit flags; skip all prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--no-set-env",
        action="store_true",
        default=False,
        help="Don't persist CUDALINK_DOORBELL (any mode) / CUDALINK_RECEIVER_PYTHON_EXE (modes 2, 4) as "
        "current-user Windows environment variables via setx (set by default otherwise).",
    )
    args = parser.parse_args()

    # --spout-wheel implies --spout
    if args.spout_wheel:
        args.spout = True
    # --native-wheel implies --native
    if args.native_wheel:
        args.native = True

    print()
    print(_bold("=" * 60))
    print(_bold("  cuda-link TD Library Installer"))
    print(_bold("=" * 60))
    if args.dry_run:
        print(_yellow("  [DRY-RUN MODE] — no commands will be executed."))
    print()

    # Resolve core wheel
    core_wheel = resolve_wheel(args.wheel, args.dry_run)

    # Determine mode
    mode = args.mode
    if mode is None:
        if args.non_interactive:
            sys.exit(_red("[error] --non-interactive requires --mode."))
        mode = _interactive_menu()

    print(f"\n  Mode {mode}: {_MODE_DESCRIPTIONS[mode]}")

    # Decide whether to also install the native wait-backend accelerator.
    # Soft default ON — unlike Spout, no interactive prompt: --no-native is the
    # only way to skip it, and a build/resolve failure degrades gracefully
    # (resolve_native_wheel returns None) rather than aborting the install.
    install_native = not args.no_native

    # Decide whether to also install the Spout bridge
    if args.spout:
        install_spout = True
    elif args.no_spout or args.non_interactive:
        install_spout = False
    else:
        install_spout = _prompt_yes_no("\n  Also install the Spout bridge (cuda_link_spout)?", default=False)

    wheels: list[Path] = [core_wheel]
    if install_native:
        native_wheel = resolve_native_wheel(args.native_wheel, args.dry_run)
        if native_wheel:
            wheels.append(native_wheel)
    if install_spout:
        wheels.append(resolve_spout_wheel(args.spout_wheel, args.dry_run))

    # Set unconditionally by --no-set-env: True unless the user opted out.
    set_env = not args.no_set_env

    # Dispatch
    if mode == 1:
        mode_1_external_folder(wheels, args.target, args.non_interactive, args.dry_run)
    elif mode == 2:
        mode_2_venv(wheels, args.venv, args.non_interactive, args.dry_run, set_env)
    elif mode == 3:
        mode_3_conda(wheels, args.conda, args.non_interactive, args.dry_run)
    elif mode == 4:
        mode_4_system_python(wheels, args.python, args.non_interactive, args.dry_run, set_env)
    elif mode == 5:
        mode_5_td_python(wheels, args.td_python, args.non_interactive, args.dry_run)

    # CUDALINK_DOORBELL: persisted here (not a library code default) after ANY
    # successful mode above — a mode function that hit sys.exit() on error never
    # reaches this point, so a failed install correctly skips it too.
    if set_env:
        print()
        print(_bold("  Persisting environment variable for the R2 doorbell:"))
        if _setx_user("CUDALINK_DOORBELL", "1", args.dry_run):
            _print_env_vars_set({"CUDALINK_DOORBELL": "1"})


if __name__ == "__main__":
    main()
