"""The native loader must fail with actionable guidance when _native_waiter is
absent, or when the caller has no cudaEventQuery pointer to hand in.

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
def test_load_native_backend_raises_actionable_error_when_module_absent():
    with pytest.raises(RuntimeError, match="native module"):
        load_native_backend(0)


@pytest.mark.skipif(not _native_module_present(), reason="requires the compiled native module")
def test_load_native_backend_raises_when_fn_ptr_is_zero():
    """A built module with no cudart pointer (caller has no CUDA runtime loaded yet).

    Does not require a live GPU — fn_ptr=0 means the native module never
    dereferences a cudart function pointer at all.
    """
    with pytest.raises(RuntimeError, match="cuda_event_query_fn_ptr was 0"):
        load_native_backend(0)
