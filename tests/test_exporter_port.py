"""
Adapter contract tests for CudaPort Protocol, value objects, and FakeCudaAdapter.

These tests run entirely without a GPU. The same contract fixture is designed to
run against CTypesCudaAdapter too (marked @pytest.mark.requires_cuda) once a GPU
is available, ensuring both adapters satisfy the same behavioural contract.
"""

from __future__ import annotations

import pytest

from cuda_link._cuda_adapters import CTypesCudaAdapter, FakeCudaAdapter
from cuda_link._exporter_port import (
    CudaPort,
    ExportPolicy,
    FrameOutcome,
    FrameSpec,
    GpuFrame,
)

# ---------------------------------------------------------------------------
# Value-object tests (no adapter needed)
# ---------------------------------------------------------------------------


def test_frame_spec_frozen() -> None:
    spec = FrameSpec(shm_name="test", height=512, width=512)
    with pytest.raises((AttributeError, TypeError)):
        spec.height = 1024  # type: ignore[misc]


def test_frame_spec_defaults() -> None:
    spec = FrameSpec(shm_name="x", height=100, width=200)
    assert spec.channels == 4
    assert spec.dtype == "uint8"
    assert spec.num_slots == 3
    assert spec.device == 0


def test_export_policy_frozen() -> None:
    pol = ExportPolicy()
    with pytest.raises((AttributeError, TypeError)):
        pol.export_sync = False  # type: ignore[misc]


def test_export_policy_defaults() -> None:
    pol = ExportPolicy()
    assert pol.export_sync is True
    assert pol.use_graphs is True
    assert pol.flush_probe is True
    assert pol.strict_device is False
    assert pol.barrier_enabled is True


def test_export_policy_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDALINK_EXPORT_SYNC", "0")
    monkeypatch.setenv("CUDALINK_USE_GRAPHS", "0")
    monkeypatch.setenv("CUDALINK_EXPORT_FLUSH_PROBE", "0")
    monkeypatch.setenv("CUDALINK_STRICT_DEVICE", "1")
    monkeypatch.setenv("CUDALINK_ACTIVATION_BARRIER", "0")
    monkeypatch.setenv("CUDALINK_LIB_STREAM_PRIO", "normal")
    monkeypatch.setenv("CUDALINK_EXPORT_PROFILE", "1")

    pol = ExportPolicy.from_env()
    assert pol.export_sync is False
    assert pol.use_graphs is False
    assert pol.flush_probe is False
    assert pol.strict_device is True
    assert pol.barrier_enabled is False
    assert pol.high_priority_stream is False
    assert pol.export_profile is True


def test_export_policy_for_testing() -> None:
    pol = ExportPolicy.for_testing()
    assert pol.use_graphs is False
    assert pol.export_sync is False
    assert pol.barrier_enabled is False
    assert pol.flush_probe is False


def test_export_policy_low_latency() -> None:
    pol = ExportPolicy.low_latency()
    assert pol.export_sync is False
    assert pol.flush_probe is False
    assert pol.use_graphs is True
    assert pol.high_priority_stream is True


def test_gpu_frame_frozen() -> None:
    frame = GpuFrame(ptr=0x1000, size=1024)
    with pytest.raises((AttributeError, TypeError)):
        frame.ptr = 0x2000  # type: ignore[misc]


def test_gpu_frame_defaults() -> None:
    frame = GpuFrame(ptr=0x1000, size=1024)
    assert frame.producer_stream is None


def test_frame_outcome_members() -> None:
    assert FrameOutcome.PUBLISHED != FrameOutcome.FAILED
    assert FrameOutcome.SKIPPED_BARRIER != FrameOutcome.SKIPPED_NOT_READY
    assert len(FrameOutcome) == 4


# ---------------------------------------------------------------------------
# FakeCudaAdapter — Protocol conformance
# ---------------------------------------------------------------------------


def test_fake_adapter_satisfies_protocol() -> None:
    fake = FakeCudaAdapter()
    assert isinstance(fake, CudaPort)


# ---------------------------------------------------------------------------
# FakeCudaAdapter — allocation tracking
# ---------------------------------------------------------------------------


def test_fake_adapter_malloc_free_lifecycle() -> None:
    fake = FakeCudaAdapter()
    ptr = fake.malloc(1024)
    assert ptr.value in fake.allocations
    fake.free(ptr)
    assert ptr.value not in fake.allocations
    assert ptr.value in fake.freed


def test_fake_adapter_malloc_tracks_size() -> None:
    fake = FakeCudaAdapter()
    ptr = fake.malloc(512)
    assert fake.allocations[ptr.value] == 512


def test_fake_adapter_multiple_allocations() -> None:
    fake = FakeCudaAdapter()
    ptrs = [fake.malloc(100) for _ in range(5)]
    assert len(fake.allocations) == 5
    for ptr in ptrs:
        fake.free(ptr)
    assert len(fake.allocations) == 0
    assert len(fake.freed) == 5


def test_fake_adapter_malloc_failure_injection() -> None:
    fake = FakeCudaAdapter()
    fake.fail_on_malloc_count = 2
    fake.malloc(1024)  # call 1 — succeeds
    with pytest.raises(RuntimeError, match="injected malloc failure"):
        fake.malloc(1024)  # call 2 — fails


def test_fake_adapter_no_leak_after_free() -> None:
    fake = FakeCudaAdapter()
    for _ in range(10):
        ptr = fake.malloc(256)
        fake.free(ptr)
    assert len(fake.allocations) == 0


# ---------------------------------------------------------------------------
# FakeCudaAdapter — streams
# ---------------------------------------------------------------------------


def test_fake_adapter_create_stream() -> None:
    fake = FakeCudaAdapter()
    stream = fake.create_stream()
    assert stream is not None
    fake.destroy_stream(stream)  # no-op, no error


def test_fake_adapter_create_stream_with_priority() -> None:
    fake = FakeCudaAdapter()
    stream = fake.create_stream_with_priority(flags=0x01)
    assert stream is not None


def test_fake_adapter_stream_query_always_ready() -> None:
    fake = FakeCudaAdapter()
    stream = fake.create_stream()
    assert fake.stream_query(stream) is True


def test_fake_adapter_stream_failure_injection() -> None:
    fake = FakeCudaAdapter()
    fake.fail_on_stream_create = True
    with pytest.raises(RuntimeError, match="stream creation failure"):
        fake.create_stream()


# ---------------------------------------------------------------------------
# FakeCudaAdapter — events
# ---------------------------------------------------------------------------


def test_fake_adapter_create_ipc_event() -> None:
    fake = FakeCudaAdapter()
    event = fake.create_ipc_event()
    assert event is not None
    fake.destroy_event(event)


def test_fake_adapter_event_failure_injection() -> None:
    fake = FakeCudaAdapter()
    fake.fail_on_event_create = True
    with pytest.raises(RuntimeError, match="event creation failure"):
        fake.create_ipc_event()


def test_fake_adapter_record_event_tracked() -> None:
    fake = FakeCudaAdapter()
    event = fake.create_ipc_event()
    stream = fake.create_stream()
    fake.record_event(event, stream)
    assert len(fake.recorded_events) == 1


def test_fake_adapter_ipc_get_event_handle() -> None:
    fake = FakeCudaAdapter()
    event = fake.create_ipc_event()
    handle = fake.ipc_get_event_handle(event)
    assert handle is not None


# ---------------------------------------------------------------------------
# FakeCudaAdapter — device & pointer attributes
# ---------------------------------------------------------------------------


def test_fake_adapter_get_device() -> None:
    fake = FakeCudaAdapter(device=0)
    assert fake.get_device() == 0


def test_fake_adapter_get_device_custom() -> None:
    fake = FakeCudaAdapter(device=1)
    assert fake.get_device() == 1


def test_fake_adapter_pointer_get_attributes_returns_device_memory() -> None:
    fake = FakeCudaAdapter(device=0)
    attrs = fake.pointer_get_attributes(0x1234_5678)
    assert attrs.type == 2  # cudaMemoryTypeDevice
    assert attrs.device == 0


def test_fake_adapter_peek_last_error_clean() -> None:
    fake = FakeCudaAdapter()
    assert fake.peek_last_error() == 0


def test_fake_adapter_sticky_error_raises_on_check() -> None:
    fake = FakeCudaAdapter()
    fake._sticky_error = 1
    with pytest.raises(RuntimeError, match="sticky error"):
        fake.check_sticky_error("test_op")


def test_fake_adapter_no_sticky_error_passes_check() -> None:
    fake = FakeCudaAdapter()
    fake.check_sticky_error("test_op")  # must not raise


# ---------------------------------------------------------------------------
# FakeCudaAdapter — CUDA Graphs
# ---------------------------------------------------------------------------


def test_fake_adapter_graph_roundtrip() -> None:
    fake = FakeCudaAdapter()
    stream = fake.create_stream()
    fake.stream_begin_capture(stream, mode=2)
    graph = fake.stream_end_capture(stream)
    assert graph is not None
    exec_ = fake.graph_instantiate(graph)
    assert exec_ is not None
    fake.graph_launch(exec_, stream)
    fake.graph_exec_destroy(exec_)
    fake.graph_destroy(graph)


def test_fake_adapter_graph_get_nodes_returns_one() -> None:
    fake = FakeCudaAdapter()
    stream = fake.create_stream()
    fake.stream_begin_capture(stream)
    graph = fake.stream_end_capture(stream)
    nodes = fake.graph_get_nodes(graph)
    assert len(nodes) == 1


def test_fake_adapter_graph_exec_memcpy_node_noop() -> None:
    from ctypes import c_void_p

    fake = FakeCudaAdapter()
    stream = fake.create_stream()
    fake.stream_begin_capture(stream)
    graph = fake.stream_end_capture(stream)
    nodes = fake.graph_get_nodes(graph)
    exec_ = fake.graph_instantiate(graph)
    fake.graph_exec_memcpy_node_set_params_1d(exec_, nodes[0], c_void_p(0x1000), c_void_p(0x2000), 256, 3)


def test_fake_adapter_runtime_version() -> None:
    fake = FakeCudaAdapter()
    assert fake.get_runtime_version() >= 11_040


# ---------------------------------------------------------------------------
# CTypesCudaAdapter — Protocol conformance (requires GPU)
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_ctypes_adapter_satisfies_protocol() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    assert isinstance(adapter, CudaPort)


@pytest.mark.requires_cuda
def test_ctypes_adapter_get_device() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    assert adapter.get_device() == 0


@pytest.mark.requires_cuda
def test_ctypes_adapter_malloc_free() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    ptr = adapter.malloc(4096)
    assert ptr is not None and ptr.value != 0
    adapter.free(ptr)


@pytest.mark.requires_cuda
def test_ctypes_adapter_peek_last_error_clean() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    assert adapter.peek_last_error() == 0


@pytest.mark.requires_cuda
def test_ctypes_adapter_stream_lifecycle() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    stream = adapter.create_stream(flags=0x01)
    assert stream is not None
    adapter.stream_synchronize(stream)
    adapter.destroy_stream(stream)


@pytest.mark.requires_cuda
def test_ctypes_adapter_event_lifecycle() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    stream = adapter.create_stream()
    event = adapter.create_sync_event()
    adapter.record_event(event, stream)
    adapter.stream_synchronize(stream)
    adapter.destroy_event(event)
    adapter.destroy_stream(stream)


@pytest.mark.requires_cuda
def test_ctypes_adapter_ipc_event_handle() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    event = adapter.create_ipc_event()
    handle = adapter.ipc_get_event_handle(event)
    assert handle is not None
    adapter.destroy_event(event)


@pytest.mark.requires_cuda
def test_ctypes_adapter_pointer_get_attributes() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    ptr = adapter.malloc(1024)
    attrs = adapter.pointer_get_attributes(ptr.value)
    assert attrs.type in (2, 3)  # device or managed
    assert attrs.device == 0
    adapter.free(ptr)


@pytest.mark.requires_cuda
def test_ctypes_adapter_runtime_version() -> None:
    adapter = CTypesCudaAdapter.for_device(device=0)
    version = adapter.get_runtime_version()
    assert version >= 11_040
