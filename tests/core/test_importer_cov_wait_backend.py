"""Coverage tests for Importer._resolve_wait_backend() — the R5 native-wait seam.

Exercises all four resolution outcomes (ImportError, AttributeError, RuntimeError,
success) by monkeypatching sys.modules["cuda_link._native_loader"] directly, never
touching the real compiled _native_waiter extension (which may not be built on
this host/platform at all -- see tests/core/test_wait_backend_seam.py docstring
for the same rationale).
"""

from __future__ import annotations

import sys
import types

import pytest
from fakes import make_connected_importer


def test_resolve_wait_backend_returns_none_when_native_loader_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """import of cuda_link._native_loader fails -> ImportError -> None (Python fallback)."""
    monkeypatch.setitem(sys.modules, "cuda_link._native_loader", None)
    imp = make_connected_importer(write_idx=1)

    assert imp._resolve_wait_backend() is None


def test_resolve_wait_backend_returns_none_without_fn_ptr_method() -> None:
    """FakeCUDAAdapter has no cudart_event_query_fn_ptr() -> AttributeError -> None."""
    imp = make_connected_importer(write_idx=1)
    assert not hasattr(imp._cuda, "cudart_event_query_fn_ptr")

    assert imp._resolve_wait_backend() is None


def test_resolve_wait_backend_returns_none_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_native_backend() raising RuntimeError (extension not built) -> None."""
    imp = make_connected_importer(write_idx=1)
    monkeypatch.setattr(imp._cuda, "cudart_event_query_fn_ptr", lambda: 0x1234, raising=False)

    fake_loader = types.ModuleType("cuda_link._native_loader")

    def _raise(_fn_ptr: int) -> None:
        raise RuntimeError("simulated: native module not built")

    fake_loader.load_native_backend = _raise
    monkeypatch.setitem(sys.modules, "cuda_link._native_loader", fake_loader)

    assert imp._resolve_wait_backend() is None


def test_resolve_wait_backend_returns_backend_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full success path: fn_ptr resolves, native loader returns a backend instance."""
    imp = make_connected_importer(write_idx=1)
    monkeypatch.setattr(imp._cuda, "cudart_event_query_fn_ptr", lambda: 0x1234, raising=False)

    sentinel_backend = object()
    fake_loader = types.ModuleType("cuda_link._native_loader")
    fake_loader.load_native_backend = lambda fn_ptr: sentinel_backend  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "cuda_link._native_loader", fake_loader)

    assert imp._resolve_wait_backend() is sentinel_backend
