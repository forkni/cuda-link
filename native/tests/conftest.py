"""Make ``cuda_link_native`` importable from native/src without installing the package."""

from __future__ import annotations

import site
import sys
import sysconfig
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# 1. Locate any INSTALLED package's native extension directly via sysconfig/site
#    paths — independent of sys.path timing/order. Deliberately NOT
#    importlib.util.find_spec("cuda_link_native"): pytest's own `pythonpath` ini
#    option (see native/pyproject.toml) inserts _SRC into sys.path as part of its
#    bootstrap, which happens BEFORE this conftest module body runs — so a
#    find_spec-based check here would already resolve to the local source tree
#    (no compiled extension) instead of the installed package, silently defeating
#    the whole point of this file. Probing known site-packages roots directly
#    sidesteps that ordering entirely.
_native_dirs: list[str] = []
_candidate_roots = {sysconfig.get_path("purelib"), sysconfig.get_path("platlib"), *site.getsitepackages()}
for _root in _candidate_roots:
    _pkg_dir = Path(_root) / "cuda_link_native"
    if list(_pkg_dir.glob("_native_waiter*.pyd")):
        _native_dirs.append(str(_pkg_dir))

# 2. Always prepend local src so working-tree Python changes (_backend.py, _native.py, …)
#    are what the tests actually exercise.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 3. If native extension lives in an installed package, extend the local package's
#    __path__ so `from cuda_link_native import _native_waiter` / `load_native_backend()`
#    still finds the compiled .pyd.  cuda_link_native.__init__ does NOT eagerly import
#    _native_waiter, so this manipulation is safe after the package is first imported.
if _native_dirs:
    import cuda_link_native as _cln  # triggers import from local src (now first in path)

    for _nd in _native_dirs:
        if _nd not in _cln.__path__:
            _cln.__path__.append(_nd)
