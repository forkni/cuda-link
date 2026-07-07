"""The native loader must fail with actionable guidance when _native_waiter is
absent, or when the caller has no cudaEventQuery pointer to hand in.

In CI / on non-Windows machines the compiled extension does not exist; loading it
should raise a clear RuntimeError (not a bare ImportError).
"""

from __future__ import annotations

import pytest

from cuda_link._native_loader import load_native_backend


def _native_module_present() -> bool:
    try:
        from cuda_link import _native_waiter  # noqa: F401

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


@pytest.mark.skipif(not _native_module_present(), reason="requires the compiled native module")
def test_load_native_backend_refuses_hosts_without_hr_timers(monkeypatch):
    """A host that cannot create high-resolution waitable timers (Windows 10
    pre-1803) must be refused at load time — activating there would reintroduce
    the ~15.6ms-quantized block-phase naps fixed on 2026-07-06 — so the caller
    falls back to the python wait path.

    Does not require a live GPU: the sentinel fn_ptr just has to be non-zero to
    get past the cudart_resolved() gate; it is never dereferenced because the
    hr_nap_supported() check raises first. The pointer is reset to 0 afterwards
    so no other test can observe the sentinel as a resolved cudart.
    """
    from cuda_link import _native_waiter

    monkeypatch.setattr(_native_waiter, "hr_nap_supported", lambda: False)
    try:
        with pytest.raises(RuntimeError, match="high-resolution waitable timers"):
            load_native_backend(0xDEAD)
    finally:
        _native_waiter.set_cuda_event_query(0)
