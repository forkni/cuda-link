"""Coverage-focused tests for cuda_link._cuda_adapters.

No existing test file targets this module directly (its classes are used
pervasively as fixtures elsewhere, but a few specific code paths — __getattr__
delegation/caching, direct _FakeIpcHandle construction, the
create_stream_with_priority failure injection, and record_event_external —
are never exercised). All tests here are pure Python/ctypes-free (CTypesCUDAAdapter
is exercised with a plain stand-in api object, not a real CUDARuntimeAPI).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuda_link._cuda_adapters import CTypesCUDAAdapter, FakeCUDAAdapter, _FakeHandle, _FakeIpcHandle

# ---------------------------------------------------------------------------
# CTypesCUDAAdapter.__getattr__ — delegation + caching
# ---------------------------------------------------------------------------


def test_getattr_delegates_to_underlying_api():
    calls = []

    class _FakeApi:
        def some_method(self, x):
            calls.append(x)
            return x * 2

    adapter = CTypesCUDAAdapter(_FakeApi())
    result = adapter.some_method(21)

    assert result == 42
    assert calls == [21]


def test_getattr_caches_resolved_attribute_on_instance_dict():
    api = SimpleNamespace(value=99)
    adapter = CTypesCUDAAdapter(api)

    first = adapter.value
    # After the first lookup, "value" should be present in the adapter's own
    # __dict__ (cached via object.__setattr__), so it no longer round-trips
    # through __getattr__.
    assert "value" in adapter.__dict__
    assert adapter.__dict__["value"] == 99

    second = adapter.value
    assert first == second == 99


def test_getattr_raises_attributeerror_for_missing_attribute_on_api():
    adapter = CTypesCUDAAdapter(SimpleNamespace())
    with pytest.raises(AttributeError):
        _ = adapter.definitely_not_a_real_attribute


# ---------------------------------------------------------------------------
# _FakeIpcHandle — direct construction (not reached via the module-level
# ipc_get_mem_handle/ipc_get_event_handle indirection in existing tests)
# ---------------------------------------------------------------------------


def test_fake_ipc_handle_direct_construction():
    handle = _FakeIpcHandle("my_tag")
    assert handle._tag == "my_tag"
    assert handle.internal == bytes(64)
    assert handle.reserved == bytes(64)
    assert "my_tag" in repr(handle)


def test_fake_cuda_adapter_ipc_get_mem_handle_and_event_handle():
    adapter = FakeCUDAAdapter()
    mem_handle = adapter.ipc_get_mem_handle(dev_ptr=0x1234)
    event_handle = adapter.ipc_get_event_handle(event=_FakeHandle("evt"))

    assert isinstance(mem_handle, _FakeIpcHandle)
    assert isinstance(event_handle, _FakeIpcHandle)
    assert mem_handle.internal == bytes(64)
    assert event_handle.reserved == bytes(64)


# ---------------------------------------------------------------------------
# FakeCUDAAdapter.create_stream_with_priority — failure injection branch
# ---------------------------------------------------------------------------


def test_create_stream_with_priority_returns_handle_normally():
    adapter = FakeCUDAAdapter()
    stream = adapter.create_stream_with_priority(flags=0x01, priority=5)
    assert isinstance(stream, _FakeHandle)


def test_create_stream_with_priority_raises_when_injected():
    adapter = FakeCUDAAdapter()
    adapter.fail_on_stream_create = True
    with pytest.raises(RuntimeError, match="injected stream creation failure"):
        adapter.create_stream_with_priority(priority=5)


# ---------------------------------------------------------------------------
# FakeCUDAAdapter.record_event_external — never called by any existing test
# ---------------------------------------------------------------------------


def test_record_event_external_tracks_call():
    adapter = FakeCUDAAdapter()
    event = _FakeHandle("evt")
    stream = _FakeHandle("strm")

    adapter.record_event_external(event, stream)

    assert adapter.recorded_events == [("evt", "strm")]


def test_record_event_external_uses_str_fallback_for_untagged_objects():
    adapter = FakeCUDAAdapter()
    event = object()
    stream = object()

    adapter.record_event_external(event, stream)

    # Exact fallback content, not just "something got recorded" — catches a swapped
    # event/stream order or a wrong/constant fallback value.
    assert adapter.recorded_events == [(str(event), str(stream))]


# ---------------------------------------------------------------------------
# FakeCUDAAdapter.check_ipc_capability — failure injection branch
# ---------------------------------------------------------------------------


def test_check_ipc_capability_returns_none_by_default():
    adapter = FakeCUDAAdapter()
    assert adapter.check_ipc_capability() is None


def test_check_ipc_capability_raises_when_injected():
    adapter = FakeCUDAAdapter()
    adapter.fail_ipc_capability = True
    with pytest.raises(RuntimeError, match="simulated cudaDevAttrIpcEventSupport=0"):
        adapter.check_ipc_capability()


def test_check_ipc_capability_message_preserves_explicit_device_zero():
    # device=0 is falsy in Python — `device or self.device` would silently substitute
    # self.device (1) here instead. Regression guard for that truthiness bug.
    adapter = FakeCUDAAdapter(device=1)
    adapter.fail_ipc_capability = True
    with pytest.raises(RuntimeError, match="for device 0"):
        adapter.check_ipc_capability(device=0)
