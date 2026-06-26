"""Make ``cuda_link_spout`` importable from spout/src without installing the package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# Source-first for pure-Python dev/CI — BUT defer to an installed build that carries the
# compiled _spout_bridge extension. Otherwise this src shadow hides the .pyd (which lives in
# site-packages, not the source tree) and the native smoke tests can never see it.
# `find_spec` locates the package without importing it, so no sys.modules contamination.
_spec = importlib.util.find_spec("cuda_link_spout")
_has_native = bool(
    _spec
    and _spec.submodule_search_locations
    and any(list(Path(loc).glob("_spout_bridge*.pyd")) for loc in _spec.submodule_search_locations)
)
if not _has_native and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
