"""
Port contract tests for ImporterCudaPort Protocol, value objects, and FakeCudaAdapter.

These tests run entirely without a GPU. Mirrors tests/test_exporter_port.py.
"""

from __future__ import annotations

import pytest

from cuda_link._cuda_adapters import CTypesCudaAdapter, FakeCudaAdapter
from cuda_link._importer_port import (
    ImporterCudaPort,
    ImportOutcome,
    ImportPolicy,
    ImportResult,
    ImportSpec,
)

# ---------------------------------------------------------------------------
# ImportSpec
# ---------------------------------------------------------------------------


def test_import_spec_frozen() -> None:
    spec = ImportSpec(shm_name="test")
    with pytest.raises((AttributeError, TypeError)):
        spec.shm_name = "other"  # type: ignore[misc]


def test_import_spec_defaults() -> None:
    spec = ImportSpec(shm_name="x")
    assert spec.device == 0
    assert spec.shape is None
    assert spec.dtype is None
    assert spec.timeout_ms == 5000.0


def test_import_spec_explicit_values() -> None:
    spec = ImportSpec(shm_name="ch", device=1, shape=(512, 512, 4), dtype="float32", timeout_ms=1000.0)
    assert spec.device == 1
    assert spec.shape == (512, 512, 4)
    assert spec.dtype == "float32"
    assert spec.timeout_ms == 1000.0


# ---------------------------------------------------------------------------
# ImportPolicy
# ---------------------------------------------------------------------------


def test_import_policy_frozen() -> None:
    pol = ImportPolicy()
    with pytest.raises((AttributeError, TypeError)):
        pol.wait_spin_us = 999  # type: ignore[misc]


def test_import_policy_defaults() -> None:
    pol = ImportPolicy()
    assert pol.wait_spin_us == 200
    assert pol.d2h_num_streams == 1
    assert pol.d2h_stream_high_priority is False
    assert pol.allow_pageable_fallback is False
    assert pol.debug is False


def test_import_policy_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDALINK_WAIT_SPIN_US", "1000")
    monkeypatch.setenv("CUDALINK_D2H_STREAMS", "2")
    monkeypatch.setenv("CUDALINK_D2H_STREAM_PRIO", "high")
    monkeypatch.setenv("CUDALINK_ALLOW_PAGEABLE_FALLBACK", "1")

    pol = ImportPolicy.from_env()
    assert pol.wait_spin_us == 1000
    assert pol.d2h_num_streams == 2
    assert pol.d2h_stream_high_priority is True
    assert pol.allow_pageable_fallback is True
    assert pol.debug is False


def test_import_policy_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CUDALINK_WAIT_SPIN_US",
        "CUDALINK_D2H_STREAMS",
        "CUDALINK_D2H_STREAM_PRIO",
        "CUDALINK_ALLOW_PAGEABLE_FALLBACK",
    ):
        monkeypatch.delenv(var, raising=False)

    pol = ImportPolicy.from_env()
    assert pol.wait_spin_us == 200
    assert pol.d2h_num_streams == 1
    assert pol.d2h_stream_high_priority is False
    assert pol.allow_pageable_fallback is False


def test_import_policy_for_testing() -> None:
    pol = ImportPolicy.for_testing()
    assert pol.wait_spin_us == 0
    assert pol.d2h_num_streams == 1
    assert pol.d2h_stream_high_priority is False
    assert pol.allow_pageable_fallback is True


def test_import_policy_low_latency() -> None:
    pol = ImportPolicy.low_latency()
    assert pol.wait_spin_us == 2000
    assert pol.d2h_num_streams == 2
    assert pol.d2h_stream_high_priority is True
    assert pol.allow_pageable_fallback is False


# ---------------------------------------------------------------------------
# ImportOutcome
# ---------------------------------------------------------------------------


def test_import_outcome_members() -> None:
    assert ImportOutcome.NEW_FRAME != ImportOutcome.NO_FRAME
    assert ImportOutcome.SHUTDOWN != ImportOutcome.RECONNECTING
    assert ImportOutcome.TIMEOUT != ImportOutcome.NO_FRAME
    assert len(ImportOutcome) == 5


# ---------------------------------------------------------------------------
# ImportResult
# ---------------------------------------------------------------------------


def test_import_result_frozen() -> None:
    result = ImportResult(outcome=ImportOutcome.NEW_FRAME, frame=object())
    with pytest.raises((AttributeError, TypeError)):
        result.outcome = ImportOutcome.NO_FRAME  # type: ignore[misc]


def test_import_result_no_frame_default() -> None:
    result = ImportResult(outcome=ImportOutcome.NO_FRAME)
    assert result.frame is None


def test_import_result_new_frame_carries_payload() -> None:
    sentinel = object()
    result = ImportResult(outcome=ImportOutcome.NEW_FRAME, frame=sentinel)
    assert result.outcome is ImportOutcome.NEW_FRAME
    assert result.frame is sentinel


# ---------------------------------------------------------------------------
# FakeCudaAdapter — Protocol conformance
# ---------------------------------------------------------------------------


def test_fake_adapter_satisfies_importer_protocol() -> None:
    fake = FakeCudaAdapter()
    assert isinstance(fake, ImporterCudaPort)


# ---------------------------------------------------------------------------
# FakeCudaAdapter — importer-side IPC memory
# ---------------------------------------------------------------------------


def test_fake_adapter_ipc_open_mem_handle_tracks_handle() -> None:
    from cuda_link._cuda_adapters import _FakeIpcHandle

    fake = FakeCudaAdapter()
    handle = _FakeIpcHandle("test_mem")
    ptr = fake.ipc_open_mem_handle(handle, flags=1)
    assert ptr is not None
    assert ptr.value in fake.opened_mem_handles


def test_fake_adapter_ipc_close_mem_handle_removes_entry() -> None:
    from cuda_link._cuda_adapters import _FakeIpcHandle

    fake = FakeCudaAdapter()
    handle = _FakeIpcHandle("test_close")
    ptr = fake.ipc_open_mem_handle(handle)
    fake.ipc_close_mem_handle(ptr)
    assert ptr.value not in fake.opened_mem_handles


def test_fake_adapter_ipc_open_event_handle_returns_handle() -> None:
    from cuda_link._cuda_adapters import _FakeIpcHandle

    fake = FakeCudaAdapter()
    event_handle = _FakeIpcHandle("test_evt")
    evt = fake.ipc_open_event_handle(event_handle)
    assert evt is not None


def test_fake_adapter_query_event_always_true() -> None:
    fake = FakeCudaAdapter()
    evt = fake.create_sync_event()
    assert fake.query_event(evt) is True


def test_fake_adapter_synchronize_noop() -> None:
    fake = FakeCudaAdapter()
    fake.synchronize()  # must not raise


# ---------------------------------------------------------------------------
# FakeCudaAdapter — importer-side host memory
# ---------------------------------------------------------------------------


def test_fake_adapter_malloc_host_alloc_returns_valid_ptr() -> None:
    fake = FakeCudaAdapter()
    ptr = fake.malloc_host_alloc(1024, flags=0x01)
    assert ptr is not None
    assert ptr.value != 0


def test_fake_adapter_free_host_releases_allocation() -> None:
    fake = FakeCudaAdapter()
    ptr = fake.malloc_host_alloc(256)
    fake.free_host(ptr)
    # After free, the ptr should not be in _host_allocs
    ptr_int = ptr.value
    assert ptr_int not in fake._host_allocs


def test_fake_adapter_malloc_host_alloc_backed_by_real_buffer() -> None:
    import ctypes

    import numpy as np

    fake = FakeCudaAdapter()
    nbytes = 128
    ptr = fake.malloc_host_alloc(nbytes)
    # The allocation is a real ctypes buffer — np.frombuffer must succeed
    buf = (ctypes.c_ubyte * nbytes).from_address(ptr.value)
    arr = np.frombuffer(buf, dtype=np.uint8)
    assert arr.shape == (nbytes,)


def test_fake_adapter_host_register_noop() -> None:
    fake = FakeCudaAdapter()
    fake.host_register(0x1000, 4096)  # must not raise


def test_fake_adapter_host_unregister_noop() -> None:
    fake = FakeCudaAdapter()
    fake.host_unregister(0x1000)  # must not raise


# ---------------------------------------------------------------------------
# CTypesCudaAdapter — Protocol conformance (requires GPU)
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_ctypes_adapter_satisfies_importer_protocol() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    assert isinstance(adapter, ImporterCudaPort)
