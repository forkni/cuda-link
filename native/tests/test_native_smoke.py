"""
First on-hardware smoke tests for the _native_waiter native module.

These tests are the first exercises of the real cudart resolution + spin/block
state machine that is otherwise only covered by pure-Python unit tests against
FakeWaitBackend. They require a Windows box with a CUDA runtime already loaded
in-process and the native module compiled via:

    pip install ./native

Run:
    pytest native/tests/test_native_smoke.py -m requires_native -v

All tests skip automatically when the native module is absent so the file is
safe to collect in any environment.
"""

from __future__ import annotations

import pytest
from cuda_link_native._backend import WaitStatus
from cuda_link_native._native import load_native_backend


def _native_module_present() -> bool:
    """Return True if the compiled _native_waiter extension resolves cudart.

    Goes through load_native_backend() so the os.add_dll_directory() call in
    _native.py runs first, and so a built-but-cudart-unresolved module (no CUDA
    runtime loaded yet in this process) is also treated as "not usable" here.
    """
    try:
        load_native_backend()
        return True
    except (ImportError, RuntimeError):
        return False


_skip_no_native = pytest.mark.skipif(
    not _native_module_present(),
    reason="native _native_waiter not built or no cudart loaded — see native/README.md",
)

pytestmark = [pytest.mark.requires_native, _skip_no_native]


def _make_event():
    """Create a real CUDA event via cuda_link's runtime wrapper; return (runtime, event)."""
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime  # noqa: PLC0415

    rt = get_cuda_runtime()
    evt = rt.create_event()
    return rt, evt


def test_wait_slot_returns_ready_spin_for_already_signaled_event() -> None:
    """An event that is already complete should be caught in the spin phase."""
    backend = load_native_backend()
    rt, evt = _make_event()
    try:
        rt.record_event(evt)  # record on the default/null stream -> completes immediately
        rt.synchronize()
        result = backend.wait_slot(int(evt), 0, 0, 0, spin_us=200, timeout_ms=1000)
        assert result.status is WaitStatus.READY_SPIN
        assert result.waited_us < 1000  # well under the full timeout
    finally:
        rt.destroy_event(evt)


def test_wait_slot_times_out_on_a_never_signaled_event() -> None:
    """An event that is never recorded should time out at the requested deadline."""
    backend = load_native_backend()
    rt, evt = _make_event()
    try:
        result = backend.wait_slot(int(evt), 0, 0, 0, spin_us=200, timeout_ms=50)
        assert result.status is WaitStatus.TIMEOUT
        assert result.waited_us >= 50 * 1000 * 0.8  # allow scheduling slack
    finally:
        rt.destroy_event(evt)
