"""
install_td_library.py — Multi-target installer for cuda-link → TouchDesigner library mode.

Supports five installation targets so the cuda_link package is importable inside
TouchDesigner's Python environment.  The bootstrap Text DAT (CUDALinkBootstrap.py)
then uses sys.path injection to make the package available without mirror Text DATs.

Wheel resolution (prebuilt-first — see docs/adr/0013-prebuilt-wheel-distribution.md):
cuda-link ships as a compiled cp311 wheel (carrying the native _native_waiter
wait-backend accelerator) plus a universal py3-none-any fallback for every other
interpreter. This installer never compiles anything on an end-user machine. For
each install target it resolves a wheel — matched to THAT target's Python version,
not the installer's own — in this order:
    1. --wheel <path>                     explicit override
    2. dist\\cuda_link-*-<tag>.whl          a matching prebuilt wheel already present
    3. auto-download from GitHub Releases  matching the installed __version__
    4. --build (dev-only, requires MSVC)   compile locally via utils\\build_wheel.cmd
Two supported scenarios in practice: a system Python 3.11 install (mode 4), or a
StreamDiffusionTD-style venv pinned to Python 3.11.9 (mode 2) — both resolve the
native cp311 wheel automatically. Any other interpreter version gets the
py3-none-any fallback; the native accelerator is a marginal (<1-5%) latency win,
so the fallback loses almost nothing functionally.

Usage (interactive):
    python scripts\\install_td_library.py

Usage (non-interactive / CI):
    python scripts\\install_td_library.py --mode 1 --target "D:\\cuda_link_lib"
    python scripts\\install_td_library.py --mode 2 --venv "D:\\myproject\\.venv"
    python scripts\\install_td_library.py --mode 3 --conda base
    python scripts\\install_td_library.py --mode 4 --python "C:\\Python311\\python.exe"
    python scripts\\install_td_library.py --mode 5 --td-python "C:\\Program Files\\Derivative\\TouchDesigner\\bin\\python.exe"
        (mode 5 is deprecated — prefer mode 2 or 4; see docs/adr/0013-prebuilt-wheel-distribution.md)

Common flags:
    --non-interactive   Require all target args; skip menu and prompts.
    --dry-run           Print what would run without executing.
    --wheel <path>      Override wheel path (skips resolution entirely).
    --build             Allow local compilation via utils\\build_wheel.cmd when no
                        prebuilt or downloadable wheel is found. Dev-only —
                        requires MSVC on Windows for the native cp311 wheel. End
                        users should not need this; a prebuilt wheel or GitHub
                        Release download covers every supported target.

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
import urllib.request
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

# GitHub Release asset base URL; assets are named cuda_link-<version>-<tag>.whl
# and attached to each release.yml `v<version>` tag run — see
# .github/workflows/release.yml and docs/adr/0013-prebuilt-wheel-distribution.md.
_GITHUB_RELEASES_BASE = "https://github.com/forkni/cuda-link/releases/download"

# Only cp311 has a compiled native wheel today (CI's setup-python matrix — see
# .github/workflows/release.yml). Any other target version, or one that could
# not be determined at all, gets the universal fallback so install never fails.
_NATIVE_WHEEL_TAG = "cp311-cp311-win_amd64"
_FALLBACK_WHEEL_TAG = "py3-none-any"


def _wheel_tag_for_version(version: tuple[int, int] | None) -> str:
    """Return the wheel filename-tag substring to prefer for a target Python version."""
    if version == (3, 11):
        return _NATIVE_WHEEL_TAG
    return _FALLBACK_WHEEL_TAG


def _find_wheel(tag: str) -> Path | None:
    """Return the newest cuda_link-*.whl in dist/ whose filename contains `tag`, or None."""
    dist = REPO_ROOT / "dist"
    if not dist.is_dir():
        return None
    wheels = sorted(
        (p for p in dist.glob("cuda_link-*.whl") if tag in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return wheels[0] if wheels else None


# Repo-relative source roots whose changes should invalidate the wheel.
_CORE_SOURCE_ROOTS = ("src/cuda_link", "pyproject.toml", "CMakeLists.txt")

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
    (correct-but-slower) — never a wrong install. Use the --wheel override flag
    to force-reuse a specific wheel unconditionally.

    `_root` overrides the repo root; only tests should pass it.
    """
    return _newest_source_mtime(roots, _root=_root) > wheel.stat().st_mtime


def _installed_version() -> str:
    """Read cuda_link.__version__ from source, without importing the package.

    Importing would pull in the package's own import graph just to read a
    string; this installer may run from an interpreter that doesn't have
    cuda_link's runtime deps installed yet (that's the whole point of it).
    """
    init_py = REPO_ROOT / "src" / "cuda_link" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        sys.exit(_red(f"[error] Could not find __version__ in {init_py}"))
    return m.group(1)


def _download_release_wheel(version: str, tag: str, dry_run: bool) -> Path | None:
    """Auto-download the matching wheel asset from the GitHub Release for `version`.

    Public asset URL — no `gh` auth needed. Returns None (with a warning printing
    the exact URL + --wheel instructions) on any failure: offline, no matching
    release, asset not yet uploaded, etc. Never raises — this is one link in
    resolve_wheel()'s fallback chain, not the only one.
    """
    filename = f"cuda_link-{version}-{tag}.whl"
    url = f"{_GITHUB_RELEASES_BASE}/v{version}/{filename}"
    dest = REPO_ROOT / "dist" / filename

    print(_yellow(f"  [download] No local wheel found — fetching {url}"))
    if dry_run:
        print(_yellow(f"    [dry-run] Would download to {dest}"))
        return dest

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https:// GitHub Releases URL
            data = resp.read()
    except OSError as e:
        reason = f"HTTP {e.code}" if hasattr(e, "code") else str(e)
        print(_yellow(f"    [warn] Auto-download failed ({reason})."))
        print(_yellow(f"    Manual fallback: download {url}"))
        print(_yellow("    then re-run with --wheel <downloaded path>"))
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(_green(f"    Downloaded {dest.name} ({len(data) // 1024} KB)"))
    return dest


def _build_wheel(tag: str, dry_run: bool) -> Path | None:
    """Run build_wheel.cmd to produce a wheel locally; return its path.

    Dev-only: only reached from resolve_wheel() when the caller passed --build.
    End-user installs never take this path — see the module docstring.
    """
    print(_bold("[build] --build passed and no prebuilt/downloadable wheel found — building locally..."))
    cmd_path = REPO_ROOT / "utils" / "build_wheel.cmd"
    if not cmd_path.exists():
        sys.exit(_red(f"[error] build_wheel.cmd not found at {cmd_path}"))

    cmd_args = ["cmd.exe", "/c", str(cmd_path)]
    if tag == _FALLBACK_WHEEL_TAG:
        cmd_args.append("nowaiter")

    if dry_run:
        print(_yellow(f"  [dry-run] Would run: {' '.join(cmd_args)}"))
        return REPO_ROOT / "dist" / "cuda_link-DRY_RUN.whl"

    # Use a list of args instead of shell=True to avoid shell injection.
    # stdin=DEVNULL prevents build_wheel.cmd's `pause` from blocking mid-flow;
    # `pause` reads EOF immediately and passes through without waiting.
    result = subprocess.run(cmd_args, cwd=REPO_ROOT, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        sys.exit(_red("[error] build_wheel.cmd failed — see output above."))
    wheel = _find_wheel(tag)
    if not wheel:
        sys.exit(_red(f"[error] Build succeeded but no matching wheel (tag={tag}) found in dist/"))
    return wheel


def _target_python_version(probe_cmd: list[str]) -> tuple[int, int] | None:
    """Query the target interpreter for its (major, minor) version.

    `probe_cmd` is the target's invocation prefix (e.g. ["C:/venv/Scripts/python.exe"]
    or ["conda", "run", "-n", "myenv", "python"]); a "-c <query>" is appended. This
    is a read-only probe (no installs, no side effects), so it always runs — even
    under --dry-run — which is what lets a dry-run against an existing venv/env
    still report which wheel it would actually pick.

    Returns None if the probe fails for any reason (interpreter not found, env
    doesn't exist yet, timeout, unparsable output). Callers fall back to the
    universal py3-none-any wheel in that case, so an unresolved version never
    blocks the install — it only means the native accelerator isn't selected.
    """
    try:
        result = subprocess.run(
            [*probe_cmd, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    m = re.match(r"^(\d+)\.(\d+)$", result.stdout.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def resolve_wheel(
    target_version: tuple[int, int] | None,
    override: str | None,
    dry_run: bool,
    allow_build: bool,
) -> Path:
    """Resolve the wheel to install for a target interpreter version.

    Source order: --wheel override -> a tag-matched prebuilt wheel already in
    dist/ -> auto-download the matching GitHub Release asset -> (only with
    --build) compile locally. End-user machines are expected to stop at the
    download step; see docs/adr/0013-prebuilt-wheel-distribution.md.
    """
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(_red(f"[error] --wheel path does not exist: {p}"))
        return p

    tag = _wheel_tag_for_version(target_version)
    if target_version is None:
        print(_yellow("  [warn] Could not determine target Python version — defaulting to py3-none-any."))
    elif tag == _FALLBACK_WHEEL_TAG:
        print(
            _yellow(
                f"  [info] Target is Python {target_version[0]}.{target_version[1]} — "
                "no native wheel for this version; using py3-none-any."
            )
        )

    w = _find_wheel(tag)
    # Staleness (mtime predates local src/cuda_link changes) is a dev-workflow
    # signal only: a downloaded release wheel's mtime is its download time, not
    # its content's age, so this check is meaningless — and could false-positive
    # — for an end-user install. Only consult it when --build makes a rebuild
    # possible in the first place.
    if w and allow_build and _wheel_is_stale(w, _CORE_SOURCE_ROOTS):
        print(_yellow(f"  [stale] {w.name} predates src/cuda_link changes — re-resolving..."))
        w = None

    if not w:
        w = _download_release_wheel(_installed_version(), tag, dry_run)

    if not w and allow_build:
        w = _build_wheel(tag, dry_run)

    if not w:
        sys.exit(
            _red(
                "[error] No wheel available for this target.\n"
                f"        Looked for a prebuilt dist\\cuda_link-*-{tag}.whl, then tried to auto-download\n"
                "        the matching asset from https://github.com/forkni/cuda-link/releases.\n"
                "        Fixes:\n"
                "          - Download the wheel manually from Releases and pass --wheel <path>\n"
                "          - On a Windows box with MSVC installed, re-run with --build to compile\n"
                "            locally (dev-only — see docs/adr/0013-prebuilt-wheel-distribution.md)"
            )
        )
    print(f"  Wheel: {w.name}")
    return w


def _resolve_wheel_for(probe_cmd: list[str], override: str | None, dry_run: bool, allow_build: bool) -> Path:
    """Convenience wrapper: probe a target interpreter, then resolve_wheel() for it."""
    version = _target_python_version(probe_cmd)
    return resolve_wheel(version, override, dry_run, allow_build)


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

    Soft-failure: a setx problem (e.g. the combined-length limit, or setx
    being unavailable on a non-Windows dev environment) warns and returns
    False rather than aborting the whole install.

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


def _print_activation(
    site_packages: Path | None,
    label: str,
    td_preferences_only: bool = False,
) -> None:
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


def mode_1_external_folder(
    target: str | None, wheel_override: str | None, allow_build: bool, non_interactive: bool, dry_run: bool
) -> None:
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

    # This mode installs using the installer's own interpreter (sys.executable
    # runs the pip below), so that's the target whose version picks the wheel.
    wheel = _resolve_wheel_for([sys.executable], wheel_override, dry_run, allow_build)

    pip = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(dest),
        str(wheel),
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
    ]
    _run_pip(pip, dry_run)
    _print_activation(dest, "Install folder")


def mode_2_venv(
    venv_dir: str | None,
    wheel_override: str | None,
    allow_build: bool,
    non_interactive: bool,
    dry_run: bool,
    set_env: bool = True,
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

    wheel = _resolve_wheel_for([str(python_exe)], wheel_override, dry_run, allow_build)

    pip = [str(pip_exe), "install", str(wheel), "--upgrade", "--force-reinstall", "--no-deps"]
    _run_pip(pip, dry_run)
    site_pkgs = _find_site_packages_in(venv)
    _print_activation(site_pkgs, "venv site-packages", td_preferences_only=True)

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


def mode_3_conda(
    conda_env: str | None, wheel_override: str | None, allow_build: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Install into a conda environment."""
    if not conda_env:
        if non_interactive:
            sys.exit(_red("[error] --mode 3 requires --conda <env_name>"))
        conda_env = input("\n  Conda environment name (e.g. 'base' or 'myenv'): ").strip()
        if not conda_env:
            sys.exit(_red("[error] No conda environment name specified."))

    wheel = _resolve_wheel_for(["conda", "run", "-n", conda_env, "python"], wheel_override, dry_run, allow_build)

    pip = [
        "conda",
        "run",
        "-n",
        conda_env,
        "pip",
        "install",
        str(wheel),
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

    _print_activation(site_pkgs, "conda env site-packages", td_preferences_only=True)
    if site_pkgs is None:
        print(_yellow("  Could not auto-detect conda site-packages path."))
        print('  Run: conda run -n <env> python -c "import site; print(site.getsitepackages())"')
        print("  then add the 'site-packages' entry to TD Preferences.")


def mode_4_system_python(
    python_exe: str | None,
    wheel_override: str | None,
    allow_build: bool,
    non_interactive: bool,
    dry_run: bool,
    set_env: bool = True,
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

    wheel = _resolve_wheel_for([str(py_path)], wheel_override, dry_run, allow_build)

    pip = [
        str(py_path),
        "-m",
        "pip",
        "install",
        str(wheel),
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

    _print_activation(site_pkgs, "Python site-packages", td_preferences_only=True)

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


def mode_5_td_python(
    td_python_exe: str | None, wheel_override: str | None, allow_build: bool, non_interactive: bool, dry_run: bool
) -> None:
    """Install directly into TouchDesigner's bundled Python.

    DEPRECATED — pip-installing into TD's own interpreter is a known anti-pattern
    (see docs/adr/0013-prebuilt-wheel-distribution.md). Prefer mode 2 (venv, the
    StreamDiffusionTD default) or mode 4 (system Python 3.11). Kept functional
    for existing workflows that depend on it.

    WARNING: Modifies TD's internal Python environment. Runs without needing CUDALINK_LIB_PATH.
    Obtain the path from TD Textport: print(app.pythonExecutable)
    """
    if not td_python_exe:
        if non_interactive:
            sys.exit(_red("[error] --mode 5 requires --td-python <td_python_exe>"))
        print()
        print(_bold("  Install into TouchDesigner's own Python."))
        print(_yellow("  DEPRECATED: pip-installing into TD's bundled interpreter is discouraged."))
        print(_yellow("  Prefer mode 2 (venv) or mode 4 (system Python 3.11) instead."))
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

    wheel = _resolve_wheel_for([str(td_py)], wheel_override, dry_run, allow_build)

    pip = [str(td_py), "-m", "pip", "install", str(wheel), "--upgrade", "--force-reinstall", "--no-deps"]

    print(_yellow("\n  Note: installing cuda_link into TD's Python means CUDALINK_LIB_PATH is NOT needed."))
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
    5: "TouchDesigner's own Python        -> no env var needed (DEPRECATED — see mode 2/4)",
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
        "--td-python",
        metavar="EXE",
        help="Mode 5 (deprecated): path to TD's python.exe (see app.pythonExecutable in Textport).",
    )
    parser.add_argument("--wheel", metavar="PATH", help="Override wheel path (skip resolution entirely).")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Allow local compilation via utils\\build_wheel.cmd when no prebuilt or downloadable "
        "wheel is found. Dev-only — requires MSVC on Windows for the native cp311 wheel.",
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

    print()
    print(_bold("=" * 60))
    print(_bold("  cuda-link TD Library Installer"))
    print(_bold("=" * 60))
    if args.dry_run:
        print(_yellow("  [DRY-RUN MODE] — no commands will be executed."))
    print()

    # Determine mode
    mode = args.mode
    if mode is None:
        if args.non_interactive:
            sys.exit(_red("[error] --non-interactive requires --mode."))
        mode = _interactive_menu()

    if mode == 5:
        print()
        print(_yellow("  [DEPRECATED] Mode 5 (TouchDesigner's own Python) is discouraged — pip-installing"))
        print(_yellow("  into TD's bundled interpreter is a known anti-pattern. Prefer mode 2 (venv, the"))
        print(_yellow("  StreamDiffusionTD default) or mode 4 (system Python 3.11)."))

    print(f"\n  Mode {mode}: {_MODE_DESCRIPTIONS[mode]}")

    # Set unconditionally by --no-set-env: True unless the user opted out.
    set_env = not args.no_set_env

    # Dispatch — each mode resolves its own wheel once it knows its target
    # interpreter (see _resolve_wheel_for), rather than a single upfront
    # resolution: the right wheel depends on THAT target's Python version.
    if mode == 1:
        mode_1_external_folder(args.target, args.wheel, args.build, args.non_interactive, args.dry_run)
    elif mode == 2:
        mode_2_venv(args.venv, args.wheel, args.build, args.non_interactive, args.dry_run, set_env)
    elif mode == 3:
        mode_3_conda(args.conda, args.wheel, args.build, args.non_interactive, args.dry_run)
    elif mode == 4:
        mode_4_system_python(args.python, args.wheel, args.build, args.non_interactive, args.dry_run, set_env)
    elif mode == 5:
        mode_5_td_python(args.td_python, args.wheel, args.build, args.non_interactive, args.dry_run)

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
