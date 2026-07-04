"""The native loader must fail with actionable guidance when _native_waiter is absent.

In CI / on non-Windows machines the compiled extension does not exist; loading it
should raise a clear RuntimeError (not a bare ImportError).
"""

from __future__ import annotations

import pytest
from cuda_link_native._native import load_native_backend


def _native_module_present() -> bool:
    try:
        from cuda_link_native import _native_waiter  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(_native_module_present(), reason="native _native_waiter is built")
def test_load_native_backend_raises_actionable_error():
    with pytest.raises(RuntimeError, match="native module"):
        load_native_backend()
