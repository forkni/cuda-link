"""The native loader must fail with actionable guidance when _native_waiter is
absent, or when the caller has no cudaEventQuery pointer to hand in.

In CI / on non-Windows machines the compiled extension does not exist; loading it
should raise a clear RuntimeError (not a bare ImportError).

The "module present" success path (load_native_backend() actually returning a
working _NativeWaitBackend, and that adapter's wait_slot() translating the
compiled module's raw tuple into a WaitResult) can only run for real against a
Windows-built _native_waiter.pyd — see test_native_smoke.py / test_state_machine.py
(requires_native, skipped here). The tests below substitute a fake in-process
_native_waiter module (types.ModuleType, monkeypatched onto the cuda_link_native
package) so that same success path is still exercised on every platform/CI leg.
"""

from __future__ import annotations

import types

import cuda_link_native
import pytest
from cuda_link_native import WaitResult, WaitStatus
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


def _fake_native_waiter(*, cudart_resolved: bool, wait_slot_return: tuple[int, float, int] = (0, 0.0, 0)):
    """Build an in-process stand-in for the compiled _native_waiter extension.

    Records every call so tests can assert on what load_native_backend() /
    _NativeWaitBackend.wait_slot() forwarded to it.
    """
    mod = types.ModuleType("cuda_link_native._native_waiter")
    mod.calls = []  # type: ignore[attr-defined]

    def set_cuda_event_query(fn_ptr: int) -> None:
        mod.calls.append(("set_cuda_event_query", fn_ptr))  # type: ignore[attr-defined]

    def _cudart_resolved() -> bool:
        return cudart_resolved

    def wait_slot(*args: int) -> tuple[int, float, int]:
        mod.calls.append(("wait_slot", args))  # type: ignore[attr-defined]
        return wait_slot_return

    mod.set_cuda_event_query = set_cuda_event_query  # type: ignore[attr-defined]
    mod.cudart_resolved = _cudart_resolved  # type: ignore[attr-defined]
    mod.wait_slot = wait_slot  # type: ignore[attr-defined]
    return mod


def test_load_native_backend_success_with_mocked_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """A "built" module that resolves cudart yields a working _NativeWaitBackend."""
    fake = _fake_native_waiter(cudart_resolved=True)
    monkeypatch.setattr(cuda_link_native, "_native_waiter", fake, raising=False)

    backend = load_native_backend(0x1234)

    assert fake.calls == [("set_cuda_event_query", 0x1234)]
    # White-box: confirm the adapter wraps the exact module instance we injected.
    assert backend._mod is fake  # type: ignore[attr-defined]


def test_load_native_backend_raises_when_module_reports_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocked equivalent of test_load_native_backend_raises_when_fn_ptr_is_zero above,
    exercised without needing the compiled extension to be present."""
    fake = _fake_native_waiter(cudart_resolved=False)
    monkeypatch.setattr(cuda_link_native, "_native_waiter", fake, raising=False)

    with pytest.raises(RuntimeError, match="cuda_event_query_fn_ptr was 0"):
        load_native_backend(0x1234)


def test_native_wait_backend_wait_slot_translates_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """_NativeWaitBackend.wait_slot() forwards args 1:1 and translates the compiled
    module's (status_int, waited_us, method) tuple into a WaitResult."""
    fake = _fake_native_waiter(cudart_resolved=True, wait_slot_return=(1, 123.5, 7))
    monkeypatch.setattr(cuda_link_native, "_native_waiter", fake, raising=False)
    backend = load_native_backend(0x1234)

    result = backend.wait_slot(
        event_ptr=0x1,
        doorbell_handle=0x2,
        write_idx_addr=0x3,
        last_write_idx=5,
        spin_us=10,
        timeout_ms=100,
    )

    assert result == WaitResult(status=WaitStatus.READY_DOORBELL, waited_us=123.5, method=7)
    assert fake.calls[-1] == ("wait_slot", (0x1, 0x2, 0x3, 5, 10, 100))
