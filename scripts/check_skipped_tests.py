#!/usr/bin/env python3
"""
Check Skipped Tests - Fail the build when disabled tests have piled up.

Implements the skipped/xfail-test ratchet from Core Rule 11 (see
references/test-suite-drift.md). A test suite can rot to "green" one skip/xfail at a time --
coverage gates don't catch this, because a skipped test still shows up as covered-or-not-
applicable rather than as a red flag. Continuous Delivery (Humble & Farley), Ch3: "you could
even fail the build if the number of commented tests exceeds some threshold, maybe 2% of the
total." This script generalizes that to `@pytest.mark.skip`, `@pytest.mark.xfail`,
`@pytest.mark.quarantine` (the known-flaky marker), `unittest.skip`, and (best-effort, via a
comment regex) tests commented out of existence entirely.

Stdlib-only (ast + tomllib, no third-party deps) so it runs anywhere without an install step.

What does NOT count as "disabled" here (this is the part the book glosses over and that this
skill's own CUDA/Windows templates need): a test marked `@pytest.mark.skipif(...)` whose
condition is environment/platform-gating (references `sys.platform`, `requires_cuda`,
`requires_native`, or similar) is *deselection*, not suite rot -- excluding it keeps this gate
from being permanently red on exactly the CUDA-on-Windows projects this skill targets, where a
large fraction of the suite is legitimately `skipif`'d off the hosted runner.

Two gating modes (mirrors pyrefly's --baseline pattern -- see references/quality-gates.md and
the "ratcheting" discussion it shares with this script):

    --max-ratio FLOAT   fail if disabled/total exceeds this ratio (default 0.02, the book's
                         number -- see [tool.check_skipped_tests] in pyproject-snippets.toml)
    --baseline PATH      fail only if the disabled *count* has increased since the last time
                         the baseline file was written; use --update-baseline to snapshot the
                         current count as the new baseline (never lower it silently)

Usage:
    python check_skipped_tests.py tests/
    python check_skipped_tests.py tests/ --max-ratio 0.02
    python check_skipped_tests.py tests/ --baseline .skip-baseline.json
    python check_skipped_tests.py tests/ --baseline .skip-baseline.json --update-baseline
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".mypy_cache"}

# Substrings that mark a skipif condition as environment/platform gating rather than suite rot.
# Checked case-insensitively against the unparsed condition expression.
PLATFORM_GATE_HINTS = (
    "sys.platform",
    "platform.system",
    "platform.machine",
    "requires_cuda",
    "requires_native",
    "requires_gpu",
    "cuda",
    "gpu_available",
)

# Matches a commented-out test definition, e.g. "# def test_foo():" or "#    async def test_x(...)".
COMMENTED_TEST_RE = re.compile(r"^\s*#+\s*(async\s+)?def\s+test_\w+")


def iter_test_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def decorator_dotted_name(dec: ast.expr) -> str:
    """Best-effort dotted name for a decorator, e.g. `@pytest.mark.skip` -> 'pytest.mark.skip'."""
    node = dec.func if isinstance(dec, ast.Call) else dec
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def decorator_condition_text(dec: ast.expr) -> str:
    """Unparsed text of a skipif's condition (first positional arg), for platform-gate sniffing."""
    if isinstance(dec, ast.Call) and dec.args:
        try:
            return ast.unparse(dec.args[0])
        except Exception:
            return ""
    return ""


def classify_function(node) -> dict:
    """Inspect one test function's decorators. Returns counts to add to the running totals."""
    result = {"collected": 1, "disabled": 0, "platform_gated": 0}
    for dec in node.decorator_list:
        name = decorator_dotted_name(dec)
        short = name.rsplit(".", 1)[-1] if name else ""
        if short == "skipif":
            condition = decorator_condition_text(dec).lower()
            if any(hint in condition for hint in PLATFORM_GATE_HINTS):
                result["platform_gated"] += 1
            else:
                result["disabled"] += 1
        elif short in ("skip", "xfail", "quarantine"):
            # `quarantine` = known-flaky test pulled out of the PR gate; it must count here or the
            # marker becomes a way to hide failures (references/test-suite-drift.md).
            result["disabled"] += 1
    return result


def scan_file(path: Path) -> dict:
    totals = {"collected": 0, "disabled": 0, "platform_gated": 0, "commented": 0}
    text = ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return totals

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return totals

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            counts = classify_function(node)
            totals["collected"] += counts["collected"]
            totals["disabled"] += counts["disabled"]
            totals["platform_gated"] += counts["platform_gated"]

    for line in text.splitlines():
        if COMMENTED_TEST_RE.match(line):
            totals["commented"] += 1

    return totals


def load_config(root: Path) -> dict:
    """Read [tool.check_skipped_tests] from pyproject.toml, if present. Never required."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        return {}
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get("tool", {}).get("check_skipped_tests", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test_dir", nargs="?", default="tests", help="Directory to scan (default: tests)")
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        help="Fail if disabled/total exceeds this ratio (default 0.02, or "
        "[tool.check_skipped_tests].max_ratio from pyproject.toml)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to a baseline JSON file; fail only if the disabled count "
        "has increased since it was last written (ratchet mode)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current disabled count to --baseline instead of gating on it",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir).resolve()
    if not test_dir.is_dir():
        print(f"error: not a directory: {test_dir}", file=sys.stderr)
        return 2

    # Config file default, unless the caller was explicit on the CLI.
    config = load_config(test_dir.parent if test_dir.name in ("tests", "test") else test_dir)
    max_ratio = args.max_ratio
    if max_ratio is None:
        max_ratio = config.get("max_ratio", 0.02)

    totals = {"collected": 0, "disabled": 0, "platform_gated": 0, "commented": 0}
    per_file = {}
    for path in iter_test_files(test_dir):
        file_totals = scan_file(path)
        if any(file_totals.values()):
            per_file[str(path.relative_to(test_dir))] = file_totals
        for key in totals:
            totals[key] += file_totals[key]

    # "Total" includes commented-out tests, per the book's own phrasing ("exceeds some
    # threshold, maybe 2% of the total") -- a suite that comments tests out of existence
    # shrinks its own denominator otherwise, which would let the ratio hide the rot it's
    # meant to catch.
    disabled = totals["disabled"] + totals["commented"]
    total = totals["collected"] + totals["commented"]
    ratio = (disabled / total) if total else 0.0

    print(f"Skipped/xfail-test drift check: {test_dir}")
    print("=" * 70)
    print(f"Collected tests       : {totals['collected']}")
    print(f"skip/skipif/xfail      : {totals['disabled']}")
    print(f"Commented-out tests    : {totals['commented']}")
    print(f"Platform-gated (excluded from count): {totals['platform_gated']}")
    print(f"Disabled ratio         : {ratio:.1%} ({disabled}/{total})")
    if per_file:
        print()
        print("Files with disabled or commented tests:")
        for name, counts in sorted(per_file.items()):
            flagged = counts["disabled"] + counts["commented"]
            if flagged:
                print(
                    f"  {name}: disabled={counts['disabled']} commented={counts['commented']} "
                    f"platform_gated={counts['platform_gated']}"
                )

    if args.baseline is not None:
        if args.update_baseline:
            args.baseline.write_text(json.dumps({"disabled": disabled}, indent=2) + "\n", encoding="utf-8")
            print(f"\nBaseline updated: {args.baseline} (disabled={disabled})")
            return 0
        if not args.baseline.exists():
            args.baseline.write_text(json.dumps({"disabled": disabled}, indent=2) + "\n", encoding="utf-8")
            print(
                f"\nNo baseline found; created {args.baseline} with disabled={disabled}. "
                "Nothing to compare against on this run."
            )
            return 0
        try:
            baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
            baseline_count = baseline_data["disabled"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"\nerror: could not read baseline {args.baseline}: {exc}", file=sys.stderr)
            return 2
        print(f"\nBaseline disabled count: {baseline_count}")
        if disabled > baseline_count:
            print(
                f"FAIL: disabled test count increased ({baseline_count} -> {disabled}). "
                "Fix or remove the new skip/xfail/commented tests, or rerun with "
                "--update-baseline if the increase was deliberate and reviewed."
            )
            return 1
        print("OK: disabled test count did not increase.")
        return 0

    print(f"\nThreshold: {max_ratio:.1%}")
    if ratio > max_ratio:
        print(f"FAIL: disabled ratio {ratio:.1%} exceeds threshold {max_ratio:.1%}.")
        return 1
    print("OK: disabled ratio within threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
