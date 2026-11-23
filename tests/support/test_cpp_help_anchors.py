"""The C++ TOPs' operator Help button opens docs/CPP_TOP.md at a heading anchor.

Each ``FillTOPPluginInfo`` passes an anchor to ``setOpHelpUrl``
(cpp_top/src/common/info_page.h). A renamed heading in docs/CPP_TOP.md leaves the button
opening a section that no longer exists, with every other test green, so this test checks
each anchor against the GitHub heading slugs of that file.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_SRC = REPO_ROOT / "cpp_top" / "src"
DOC = REPO_ROOT / "docs" / "CPP_TOP.md"

# One call per TOP: file (relative to cpp_top/src) -> expected anchor.
EXPECTED = {
    "in_top/CudaLinkInTOP.cpp": "in-top-parameters",
    "out_top/CudaLinkOutTOP.cpp": "out-top-parameters",
}

_CALL = re.compile(r'setOpHelpUrl\(\s*\w+\s*,\s*"([^"]*)"\s*\)')


def _github_slug(heading: str) -> str:
    """GitHub's anchor for a Markdown heading: lower case, punctuation dropped, spaces to '-'."""
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _doc_slugs() -> set[str]:
    slugs: set[str] = set()
    in_fence = False
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        match = re.match(r"^#{1,6}\s+(.*)$", line)
        if match and not in_fence:
            slugs.add(_github_slug(match.group(1)))
    return slugs


def _calls() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(CPP_SRC.rglob("*.cpp")):
        anchors = _CALL.findall(path.read_text(encoding="utf-8"))
        if anchors:
            found[path.relative_to(CPP_SRC).as_posix()] = anchors
    return found


def test_each_top_sets_its_help_anchor() -> None:
    assert _calls() == {name: [anchor] for name, anchor in EXPECTED.items()}


def test_help_anchors_match_doc_headings() -> None:
    slugs = _doc_slugs()
    missing = sorted(a for anchors in _calls().values() for a in anchors if a not in slugs)
    assert not missing, f"docs/CPP_TOP.md has no heading for help anchor(s) {missing}"


def test_slug_rule() -> None:
    assert _github_slug("Out TOP parameters") == "out-top-parameters"
    assert _github_slug("Live matrix 2.0.0.dev0: 9 cells × 6 formats (2026-11-18)") == (
        "live-matrix-200dev0-9-cells--6-formats-2026-11-18"
    )
