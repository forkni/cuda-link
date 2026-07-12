"""
Tests for cuda_link.cuda_runtime_types ABI guards.

Background
----------
The module-level struct-size guards used to be plain ``assert`` statements. Asserts are
stripped when Python runs with the ``-O`` (optimize) flag, so a ctypes struct-layout drift
(e.g. from a platform/compiler ABI change) would import silently instead of failing loudly.
They were replaced with an explicit ``_abi_guard()`` helper that raises ``RuntimeError``
unconditionally, regardless of ``-O``.

These tests are GPU-free — they only touch ctypes struct definitions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The repo's pyproject.toml adds "src" to pytest's `pythonpath` ini option, so imports work
# in-process without an editable install. A bare subprocess doesn't get that for free — it
# would silently fall back to whatever `cuda_link` happens to be installed in site-packages
# (which may be stale). Explicitly prepend src/ to PYTHONPATH so the subprocess imports the
# same source tree these tests are validating.
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC_DIR if not existing else f"{_SRC_DIR}{os.pathsep}{existing}"
    return env


def test_abi_guard_passes_for_correct_size() -> None:
    """_abi_guard() is a no-op when actual == expected."""
    from cuda_link.cuda_runtime_types import _abi_guard

    _abi_guard(64, 64, "some_struct")  # must not raise


def test_abi_guard_raises_runtime_error_on_mismatch() -> None:
    """_abi_guard() raises RuntimeError (not AssertionError) on a size mismatch."""
    from cuda_link.cuda_runtime_types import _abi_guard

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        _abi_guard(63, 64, "some_struct")


def test_module_imports_cleanly_under_dash_o() -> None:
    """Regression: the ABI guards must not be assert statements.

    Under `python -O`, `assert` statements are stripped entirely. If the guards were still
    asserts, a real struct-size drift would import silently instead of raising. This test
    can't directly force a drift (the structs are hardcoded correctly), so it instead proves
    the guard mechanism survives -O by checking the module imports cleanly and _abi_guard is
    still callable and still raises on a synthetic mismatch, all inside a `python -O` subprocess.
    """
    code = "import cuda_link.cuda_runtime_types as m\nm._abi_guard(1, 2, 'synthetic')\n"
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode != 0, "expected _abi_guard to raise under -O, but process exited cleanly"
    assert "RuntimeError" in result.stderr
    assert "ABI mismatch" in result.stderr


def test_module_imports_cleanly_under_dash_o_with_real_structs() -> None:
    """The real module-level guards (correct sizes) must import cleanly under -O."""
    result = subprocess.run(
        [sys.executable, "-O", "-c", "import cuda_link.cuda_runtime_types"],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, f"import failed under -O:\n{result.stderr}"
