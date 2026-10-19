"""Coverage-focused tests for cuda_link._native_loader.

Exercises the success path of load_native_backend() and the two RuntimeError
guard branches (cudart not resolved, hr naps unsupported), plus the
_NativeWaitBackend adapter body directly — all via a fake ``_native_waiter``
module injected into sys.modules. None of this requires the compiled native
extension or a real GPU.
"""

from __future__ import annotations

import sys
import types

import pytest

from cuda_link._native_loader import _NativeWaitBackend, load_native_backend
from cuda_link._wait_backend import WaitStatus


def _install_fake_native_waiter(
    monkeypatch: pytest.MonkeyPatch, *, cudart_resolved: bool, hr_nap_supported: bool
) -> types.ModuleType:
    """Inject a fake cuda_link._native_waiter module via monkeypatch.

    Mirrors the sys.modules[name] = None ImportError-forcing trick already used
    in tests/cuda/test_nvml_observer.py, but here we inject a *working* fake
    module object so `from . import _native_waiter` succeeds via the import
    system's fromlist fallback (module.__dict__[x] = sys.modules[from_]).
    """
    import cuda_link

    fake = types.ModuleType("cuda_link._native_waiter")
    fake.set_cuda_event_query = lambda ptr: None
    fake.cudart_resolved = lambda: cudart_resolved
    fake.hr_nap_supported = lambda: hr_nap_supported
    fake.wait_slot = lambda *a: (0, 42, "spin")
    monkeypatch.setitem(sys.modules, "cuda_link._native_waiter", fake)
    monkeypatch.setattr(cuda_link, "_native_waiter", fake, raising=False)
    return fake


def test_load_native_backend_raises_when_cudart_not_resolved(monkeypatch: pytest.MonkeyPatch):
    _install_fake_native_waiter(monkeypatch, cudart_resolved=False, hr_nap_supported=True)
    with pytest.raises(RuntimeError, match="cuda_event_query_fn_ptr was 0"):
        load_native_backend(0)


def test_load_native_backend_raises_when_hr_nap_unsupported(monkeypatch: pytest.MonkeyPatch):
    _install_fake_native_waiter(monkeypatch, cudart_resolved=True, hr_nap_supported=False)
    with pytest.raises(RuntimeError, match="high-resolution waitable timers"):
        load_native_backend(0xDEAD)


@pytest.mark.parametrize(
    ("fn_ptr", "hr_nap_supported"),
    [(0, True), (0xDEAD, False)],
    ids=["zero_fn_ptr", "hr_nap_unsupported"],
)
def test_refused_load_leaves_global_event_query_untouched(
    monkeypatch: pytest.MonkeyPatch, fn_ptr: int, hr_nap_supported: bool
):
    """g_cudaEventQuery is shared by every live _NativeWaitBackend; a refused load must not overwrite it."""
    fake = _install_fake_native_waiter(monkeypatch, cudart_resolved=True, hr_nap_supported=hr_nap_supported)
    set_calls: list[int] = []
    fake.set_cuda_event_query = set_calls.append
    with pytest.raises(RuntimeError):
        load_native_backend(fn_ptr)
    assert set_calls == []


def test_load_native_backend_success_returns_native_wait_backend(monkeypatch: pytest.MonkeyPatch):
    _install_fake_native_waiter(monkeypatch, cudart_resolved=True, hr_nap_supported=True)
    backend = load_native_backend(0xDEAD)
    assert isinstance(backend, _NativeWaitBackend)


def test_native_wait_backend_init_stores_module():
    fake_mod = types.SimpleNamespace(wait_slot=lambda *a: (0, 1, "spin"))
    backend = _NativeWaitBackend(fake_mod)
    assert backend._mod is fake_mod


def test_native_wait_backend_wait_slot_delegates_and_maps_status():
    calls = []

    def fake_wait_slot(event_ptr, doorbell_handle, write_idx_addr, last_write_idx, spin_us, timeout_ms):
        calls.append((event_ptr, doorbell_handle, write_idx_addr, last_write_idx, spin_us, timeout_ms))
        return (1, 123, "doorbell")

    fake_mod = types.SimpleNamespace(wait_slot=fake_wait_slot)
    backend = _NativeWaitBackend(fake_mod)

    result = backend.wait_slot(
        event_ptr=0x1000,
        doorbell_handle=0x2000,
        write_idx_addr=0x3000,
        last_write_idx=5,
        spin_us=200,
        timeout_ms=1000,
    )

    assert calls == [(0x1000, 0x2000, 0x3000, 5, 200, 1000)]
    assert result.status is WaitStatus.READY_DOORBELL
    assert result.waited_us == 123
    assert result.method == "doorbell"


def test_native_wait_backend_cuda_error_raises_runtime_error():
    """Status 5 (CUDA_ERROR) must raise with the cudaError_t code, never pass as a result."""
    fake_mod = types.SimpleNamespace(wait_slot=lambda *a: (5, 10.0, 700))
    backend = _NativeWaitBackend(fake_mod)
    with pytest.raises(RuntimeError, match="cudaEventQuery failed: 700"):
        backend.wait_slot(0x1000, 0x2000, 0x3000, 5, 200, 1000)


def test_native_wait_backend_unknown_status_raises_runtime_error():
    fake_mod = types.SimpleNamespace(wait_slot=lambda *a: (99, 0.0, 0))
    backend = _NativeWaitBackend(fake_mod)
    with pytest.raises(RuntimeError, match="unknown status 99"):
        backend.wait_slot(0x1000, 0x2000, 0x3000, 5, 200, 1000)
