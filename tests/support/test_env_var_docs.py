"""Env-var documentation invariant (Arch item A).

Every CUDALINK_* environment variable referenced by env_bool/env_int/env_str
in src/cuda_link/*.py must appear in the Performance Tuning table in README.md.

This makes the README env-var table a hard invariant, not a best-effort doc
that silently drifts when new knobs are added.

Precedent: tests/support/test_wrapper_sync.py (ADR-0002).
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src" / "cuda_link"
_README = _PROJECT_ROOT / "README.md"

# Regex that matches env_bool/env_int/env_str call sites.
# Captures the first positional argument (the variable name string).
_ENV_CALL_RE = re.compile(r'env_(?:bool|int|str)\(\s*"([A-Z0-9_]+)"')

# Heading that marks the beginning of the performance tuning section.
_SECTION_HEADING = "### Performance Tuning (env vars)"


def _collect_env_vars() -> set[str]:
    """Return all CUDALINK_* variable names referenced in src/cuda_link/*.py."""
    found: set[str] = set()
    for path in _SRC_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for match in _ENV_CALL_RE.finditer(source):
            name = match.group(1)
            if name.startswith("CUDALINK_"):
                found.add(name)
    return found


def _extract_perf_tuning_section(readme: str) -> str:
    """Return the README slice from the perf-tuning heading to the next real heading.

    Fence-aware: a ``#`` at column 0 inside a fenced code block is NOT a heading.
    """
    lines = readme.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == _SECTION_HEADING]
    assert starts, f"Heading {_SECTION_HEADING!r} not found in README.md"
    start = starts[0]
    in_fence = False
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and re.match(r"#{1,3} ", lines[i]):
            return "\n".join(lines[start:i])
    return "\n".join(lines[start:])


def test_all_env_vars_documented_in_readme() -> None:
    """Every CUDALINK_* env var in src/ must appear in README's Performance Tuning section."""
    readme_text = _README.read_text(encoding="utf-8")
    perf_section = _extract_perf_tuning_section(readme_text)

    env_vars = _collect_env_vars()
    assert env_vars, "No CUDALINK_* env vars found in src/cuda_link — regex may be wrong"

    missing = sorted(name for name in env_vars if name not in perf_section)

    assert not missing, (
        "The following CUDALINK_* env vars are used in src/cuda_link/*.py but are NOT "
        "documented in the '### Performance Tuning (env vars)' section of README.md:\n"
        + "".join(f"  {name}\n" for name in missing)
        + "\nAdd a row for each variable to the env-var table in README.md."
    )
