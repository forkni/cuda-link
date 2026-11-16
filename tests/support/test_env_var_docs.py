"""Env-var documentation invariant (Arch item A).

Every CUDALINK_* environment variable referenced by env_bool/env_int/env_str
in src/cuda_link/*.py must appear in docs/ENV_VARS.md, the complete env-var
reference. README.md's Performance Tuning section only carries a curated
subset (the six variables most consumers reach for) with a pointer to
docs/ENV_VARS.md for the rest — see the "De-duplicate install instructions" /
env-var relocation change in the README structural trim.

This makes docs/ENV_VARS.md a hard invariant, not a best-effort doc that
silently drifts when new knobs are added.

Precedent: tests/support/test_wrapper_sync.py (ADR-0002).
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src" / "cuda_link"
_ENV_VARS_DOC = _PROJECT_ROOT / "docs" / "ENV_VARS.md"

# Regex that matches env_bool/env_int/env_str call sites.
# Captures the first positional argument (the variable name string).
_ENV_CALL_RE = re.compile(r'env_(?:bool|int|str)\(\s*"([A-Z0-9_]+)"')


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


def test_all_env_vars_documented_in_readme() -> None:
    """Every CUDALINK_* env var in src/ must appear in docs/ENV_VARS.md."""
    doc_text = _ENV_VARS_DOC.read_text(encoding="utf-8")

    env_vars = _collect_env_vars()
    assert env_vars, "No CUDALINK_* env vars found in src/cuda_link — regex may be wrong"

    missing = sorted(name for name in env_vars if name not in doc_text)

    assert not missing, (
        "The following CUDALINK_* env vars are used in src/cuda_link/*.py but are NOT "
        "documented in docs/ENV_VARS.md:\n"
        + "".join(f"  {name}\n" for name in missing)
        + "\nAdd a row for each variable to the table in docs/ENV_VARS.md."
    )
