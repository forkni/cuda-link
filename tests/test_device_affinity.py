"""
Tests for C2: source-stream and src-ptr device-affinity validation.

All tests are pure unit tests (no CUDA required) — they mock the CUDA runtime.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exporter(device: int = 0, strict: bool = False) -> object:
    """Build a CUDAIPCExporter with mocked CUDA state (no real CUDA required)."""
    from cuda_link.cuda_ipc_exporter import (
        METADATA_SIZE,
        SHM_HEADER_SIZE,
        SHUTDOWN_FLAG_SIZE,
        SLOT_SIZE,
        TIMESTAMP_SIZE,
        CUDAIPCExporter,
    )

    num_slots = 2
    shm_size = SHM_HEADER_SIZE + num_slots * SLOT_SIZE + SHUTDOWN_FLAG_SIZE + METADATA_SIZE + TIMESTAMP_SIZE

    exp = object.__new__(CUDAIPCExporter)
    exp.shm_name = "test_affinity"
    exp.height = 8
    exp.width = 8
    exp.channels = 4
    exp.dtype = "uint8"
    exp.num_slots = num_slots
    exp.data_size = 8 * 8 * 4
    exp.debug = False
    exp.write_idx = 0
    exp.frame_count = 0
    exp.device = device
    exp._export_sync = False
    exp._initialized = True
    exp._strict_device = strict
    exp._source_sync_device_warned = False
    exp._ptr_device_cache = set()
    exp._shutdown_offset = SHM_HEADER_SIZE + num_slots * SLOT_SIZE
    exp._ts_offset = exp._shutdown_offset + SHUTDOWN_FLAG_SIZE + METADATA_SIZE

    exp.source_sync_event = None
    exp.ipc_stream = MagicMock()
    exp.ipc_events = [None] * num_slots
    exp.dev_ptrs = [MagicMock() for _ in range(num_slots)]

    # Phase 2 CUDA Graphs state (disabled in mock — no real CUDA context)
    exp._use_graphs = False
    exp._graphs_disabled = False
    exp._graph_execs = [None] * num_slots
    exp._graph_memcpy_nodes = [None] * num_slots

    # Phase 3 diagnostic knobs (off in mock — no instrumentation desired)
    exp._export_profile = False
    exp._export_flush_probe = False
    exp.total_sync_us = 0.0
    exp.total_sticky_check_us = 0.0
    exp.total_flush_probe_us = 0.0

    # F9 activation barrier (disabled in mock)
    exp._barrier_enabled = False
    exp._barrier_stale_ns = 5_000_000_000
    exp._barrier_shm = None
    exp._barrier_skip_log_last_ns = 0
    exp._barrier_stale_log_last_ns = 0

    mock_attrs = MagicMock()
    mock_attrs.type = 2  # cudaMemoryTypeDevice
    mock_attrs.device = device
    exp.cuda = MagicMock()
    exp.cuda.get_device.return_value = device
    exp.cuda.pointer_get_attributes.return_value = mock_attrs

    exp.shm_handle = MagicMock()
    exp.shm_handle.buf = bytearray(shm_size)

    return exp


# ---------------------------------------------------------------------------
# C2a: record_source_sync device check
# ---------------------------------------------------------------------------


def test_record_source_sync_no_warning_on_matching_device() -> None:
    """No error logged when current device matches exporter device."""
    exp = _make_exporter(device=0)
    exp.source_sync_event = MagicMock()
    exp.cuda.get_device.return_value = 0  # matches exp.device=0

    with patch("cuda_link.cuda_ipc_exporter.logger") as mock_logger:
        exp.record_source_sync(12345)

    mock_logger.error.assert_not_called()
    assert exp._source_sync_device_warned is False


def test_record_source_sync_wrong_device_logs_error_by_default() -> None:
    """Wrong device logs error but does NOT raise when CUDALINK_STRICT_DEVICE is off."""
    exp = _make_exporter(device=0)
    exp.source_sync_event = MagicMock()
    exp.cuda.get_device.return_value = 1  # wrong device

    with patch("cuda_link.cuda_ipc_exporter.logger") as mock_logger:
        exp.record_source_sync(12345)  # should not raise

    mock_logger.error.assert_called_once()
    assert "CUDALINK_STRICT_DEVICE=1" in mock_logger.error.call_args[0][0]
    assert exp._source_sync_device_warned is True


def test_record_source_sync_wrong_device_raises_when_strict() -> None:
    """CUDALINK_STRICT_DEVICE=1: wrong device raises ValueError."""
    exp = _make_exporter(device=0, strict=True)
    exp.source_sync_event = MagicMock()
    exp.cuda.get_device.return_value = 1  # wrong device

    with pytest.raises(ValueError, match="does not match exporter device"):
        exp.record_source_sync(12345)


def test_record_source_sync_wrong_device_warns_once() -> None:
    """Device mismatch is logged at most once per exporter instance."""
    exp = _make_exporter(device=0)
    exp.source_sync_event = MagicMock()
    exp.cuda.get_device.return_value = 1  # wrong device

    with patch("cuda_link.cuda_ipc_exporter.logger") as mock_logger:
        exp.record_source_sync(12345)
        exp.record_source_sync(12345)
        exp.record_source_sync(12345)

    assert mock_logger.error.call_count == 1  # warn-once


# ---------------------------------------------------------------------------
# C2b: export_frame pointer device/type check
# ---------------------------------------------------------------------------


def test_export_frame_valid_device_ptr_no_warning() -> None:
    """No error when pointer is device memory on the correct device."""
    exp = _make_exporter(device=0)

    with patch("cuda_link.cuda_ipc_exporter.logger") as mock_logger:
        result = exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)

    assert result is True
    mock_logger.error.assert_not_called()


def test_export_frame_managed_ptr_no_warning() -> None:
    """Managed memory (type=3) is treated as valid for D2D."""
    exp = _make_exporter(device=0)
    exp.cuda.pointer_get_attributes.return_value.type = 3  # cudaMemoryTypeManaged

    with patch("cuda_link.cuda_ipc_exporter.logger") as mock_logger:
        result = exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)

    assert result is True
    mock_logger.error.assert_not_called()


def test_export_frame_host_ptr_logs_error() -> None:
    """Host pointer (type=1) logs an error when passed to export_frame."""
    exp = _make_exporter(device=0)
    exp.cuda.pointer_get_attributes.return_value.type = 1  # cudaMemoryTypeHost

    with patch("cuda_link.cuda_ipc_exporter.logger") as mock_logger:
        result = exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)

    assert result is True  # continues in non-strict mode
    mock_logger.error.assert_called_once()


def test_export_frame_host_ptr_raises_when_strict() -> None:
    """Strict mode raises ValueError for host pointer."""
    exp = _make_exporter(device=0, strict=True)
    exp.cuda.pointer_get_attributes.return_value.type = 1  # cudaMemoryTypeHost

    with pytest.raises(ValueError, match="not device/managed memory"):
        exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)


def test_export_frame_wrong_device_ptr_logs_error() -> None:
    """Pointer on wrong device logs error in non-strict mode."""
    exp = _make_exporter(device=0)
    exp.cuda.pointer_get_attributes.return_value.device = 1  # wrong device

    with patch("cuda_link.cuda_ipc_exporter.logger") as mock_logger:
        result = exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)

    assert result is True
    mock_logger.error.assert_called_once()
    assert "belongs to device" in mock_logger.error.call_args[0][0]


def test_export_frame_wrong_device_ptr_raises_when_strict() -> None:
    """Strict mode raises ValueError when pointer is on wrong device."""
    exp = _make_exporter(device=0, strict=True)
    exp.cuda.pointer_get_attributes.return_value.device = 1  # wrong device

    with pytest.raises(ValueError, match="belongs to device"):
        exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)


def test_export_frame_ptr_check_cached_after_first_call() -> None:
    """pointer_get_attributes is called once per unique pointer, not every frame."""
    exp = _make_exporter(device=0)
    ptr = 0xABCD0000

    exp.export_frame(gpu_ptr=ptr, size=exp.data_size)
    exp.export_frame(gpu_ptr=ptr, size=exp.data_size)
    exp.export_frame(gpu_ptr=ptr, size=exp.data_size)

    # Should be called exactly once — subsequent calls use the cache
    assert exp.cuda.pointer_get_attributes.call_count == 1


def test_export_frame_ptr_cache_bounded_at_eight() -> None:
    """Pointer cache never exceeds 8 entries."""
    exp = _make_exporter(device=0)

    for i in range(12):
        exp._ptr_device_cache.discard(i)  # simulate 12 distinct pointers
        new_ptr = 0x1000 + i
        exp.cuda.pointer_get_attributes.reset_mock()
        exp.export_frame(gpu_ptr=new_ptr, size=exp.data_size)

    assert len(exp._ptr_device_cache) <= 8


def test_export_frame_env_gate_strict_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_STRICT_DEVICE env var is read at construction time."""
    monkeypatch.setenv("CUDALINK_STRICT_DEVICE", "1")

    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="test", height=8, width=8)
    assert exp._strict_device is True


def test_export_frame_env_gate_strict_device_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_STRICT_DEVICE defaults to 0 (warn, not raise)."""
    monkeypatch.delenv("CUDALINK_STRICT_DEVICE", raising=False)

    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="test", height=8, width=8)
    assert exp._strict_device is False


# ---------------------------------------------------------------------------
# F10: heartbeat-preserving F9 early-return
# ---------------------------------------------------------------------------


def test_f10_f9_skip_clears_shutdown_flag() -> None:
    """F9 skip path must zero shutdown_flag even when bypassing L803 heartbeat.

    Regression test for the false 'Sender shutdown detected' caused by F9's
    early-return bypassing the per-frame shutdown_flag=0 reassertion.
    """
    exp = _make_exporter(device=0)
    exp._barrier_enabled = True
    # Simulate a stale shutdown_flag=1 (e.g. from a prior producer cleanup).
    exp.shm_handle.buf[exp._shutdown_offset] = 1

    with patch.object(exp, "_check_activation_barrier", return_value=True):
        result = exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)

    assert result is False  # F9 skipped publish
    assert exp.shm_handle.buf[exp._shutdown_offset] == 0  # F10 cleared the stale byte


def test_f10_f9_skip_does_not_advance_write_idx() -> None:
    """F9 skip must not increment write_idx (no phantom frame published)."""
    exp = _make_exporter(device=0)
    exp._barrier_enabled = True
    initial_write_idx = exp.write_idx

    with patch.object(exp, "_check_activation_barrier", return_value=True):
        exp.export_frame(gpu_ptr=0xDEAD0000, size=exp.data_size)

    assert exp.write_idx == initial_write_idx
