"""The package version has one source: ``__version__`` in src/cuda_link/__init__.py.

pyproject.toml (wheel metadata), scripts/sync_td_wrapper.py (the MIRROR_VERSION stamp) and
cpp_top/CMakeLists.txt (the DLL Info header) must all read that one line. A second literal
let the wheel, the ``.tox`` stamp and the DLLs disagree after a release bump.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INIT_PY = REPO_ROOT / "src" / "cuda_link" / "__init__.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"
CPP_CMAKE = REPO_ROOT / "cpp_top" / "CMakeLists.txt"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_td_wrapper  # noqa: E402


def _project_table(text: str) -> str:
    """The ``[project]`` table of pyproject.toml as text (py3.9 has no tomllib)."""
    match = re.search(r"^\[project\]\n(.*?)(?=^\[)", text, re.MULTILINE | re.DOTALL)
    assert match, "pyproject.toml has no [project] table"
    return match.group(1)


def test_pyproject_has_no_static_version() -> None:
    project = _project_table(PYPROJECT.read_text(encoding="utf-8"))
    assert not re.search(r"^version\s*=", project, re.MULTILINE), (
        "pyproject.toml [project] must not hold a version literal; edit __version__ in src/cuda_link/__init__.py"
    )
    assert re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', project, re.MULTILINE)


def test_pyproject_reads_version_from_init() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^\[tool\.scikit-build\.metadata\.version\]\n(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, "pyproject.toml has no [tool.scikit-build.metadata.version] table"
    table = match.group(1)
    assert 'provider = "scikit_build_core.metadata.regex"' in table
    assert 'input = "src/cuda_link/__init__.py"' in table


def test_init_version_line_matches_every_reader() -> None:
    """One ``__version__ = "x"`` line, in the exact shape each reader's regex needs."""
    text = INIT_PY.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("__version__")]
    assert len(lines) == 1, lines
    # cpp_top/CMakeLists.txt: file(STRINGS ... REGEX "^__version__ = \"[^\"]+\"")
    cmake_match = re.match(r'^__version__ = "([^"]+)"', lines[0])
    assert cmake_match, f'CMake cannot read {lines[0]!r}: keep the form __version__ = "x.y.z"'
    assert sync_td_wrapper.read_package_version() == cmake_match.group(1)


def test_cmake_reads_version_from_init() -> None:
    lines = CPP_CMAKE.read_text(encoding="utf-8").splitlines()
    # Comment lines name both files in prose: only the commands count.
    code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    read = re.search(
        r'file\(STRINGS\s+"\$\{CMAKE_CURRENT_SOURCE_DIR\}/\.\./src/cuda_link/__init__\.py"\s+(\w+)'
        r'\s+REGEX\s+"\^__version__ = ',
        code,
    )
    assert read, "cpp_top must read the version line from src/cuda_link/__init__.py"
    # CUDALINK_VERSION comes from that line, not from a literal.
    assert re.search(
        r"string\(REGEX REPLACE\s+\S.*\sCUDALINK_VERSION\s+\"\$\{" + read.group(1) + r"\}\"\)",
        code,
    ), "CUDALINK_VERSION must come from the __init__.py line"
    assert not re.search(r'set\(CUDALINK_VERSION\s+"\d', code), "cpp_top must not hold a version literal"
    assert "pyproject.toml" not in code, "cpp_top must not read the version from pyproject.toml"
