"""Make ``cuda_link_spout`` importable from spout/src without installing the package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# 1. Find native extension directories BEFORE modifying sys.path — find_spec here
#    resolves to the installed package (if any) because local src is not yet on sys.path.
_spec = importlib.util.find_spec("cuda_link_spout")
_native_dirs: list[str] = []
if _spec and _spec.submodule_search_locations:
    for _loc in _spec.submodule_search_locations:
        if list(Path(_loc).glob("_spout_bridge*.pyd")):
            _native_dirs.append(_loc)

# 2. Always prepend local src so working-tree Python changes (bridge.py, _format.py, …)
#    are what the tests actually exercise.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 3. If native extension lives in the installed package, extend the local package's
#    __path__ so `from cuda_link_spout import _spout_bridge` / `load_native_backend()`
#    still finds the compiled .pyd.  cuda_link_spout.__init__ does NOT eagerly import
#    _spout_bridge, so this manipulation is safe after the package is first imported.
if _native_dirs:
    import cuda_link_spout as _csp  # triggers import from local src (now first in path)

    for _nd in _native_dirs:
        if _nd not in _csp.__path__:
            _csp.__path__.append(_nd)
