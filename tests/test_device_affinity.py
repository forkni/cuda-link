"""
Tests for C2: source-stream and src-ptr device-affinity validation.

All tests are pure unit tests (no CUDA required) — they use FakeCudaAdapter.
Rewritten in v1.6 to replace object.__new__(CUDAIPCExporter) with
Exporter.open(spec, policy=ExportPolicy(...), cuda=FakeCudaAdapter(device=device)).
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from conftest import FakeShmAdapter

from cuda_link import ExportPolicy, FrameSpec, GpuFrame
from cuda_link._cuda_adapters import FakeCudaAdapter
from cuda_link.activation_barrier import CheckerBarrier
from cuda_link.exporter import Exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_SIZE = 8 * 8 * 4  # H=8, W=8, C=4, uint8


def _make_exporter(
    device: int = 0, strict: bool = False, barrier: CheckerBarrier | None = None
) -> tuple[Exporter, FakeCudaAdapter]:
    """Open a real Exporter backed by FakeCudaAdapter (no GPU required).

    Returns (exporter, fake_cuda) so tests can configure the fake directly.
    Always call exporter.close() in a finally block.
    """
    shm_name = f"test_affinity_{uuid.uuid4().hex[:8]}"
    fake = FakeCudaAdapter(device=device)
    policy = ExportPolicy(
        strict_device=strict,
        export_sync=False,
        use_graphs=False,
        flush_probe=False,
        barrier_enabled=False,
        high_priority_stream=False,
        export_profile=False,
    )
    exp = Exporter.open(
        FrameSpec(shm_name=shm_name, height=8, width=8, channels=4, dtype="uint8", num_slots=2, device=device),
        policy=policy,
        cuda=fake,
        barrier=barrier,
    )
    return exp, fake


# ---------------------------------------------------------------------------
# C2a: record_source_sync device check
# ---------------------------------------------------------------------------


def _attrs(type: int = 2, device: int = 0):
    """Return a MagicMock that looks like cudaPointerAttributes."""
    from unittest.mock import MagicMock

    a = MagicMock()
    a.type = type
    a.device = device
    return a


# ---------------------------------------------------------------------------
# C2a: record_source_sync device check
# ---------------------------------------------------------------------------


def test_record_source_sync_no_warning_on_matching_device() -> None:
    """No error logged when current device matches exporter device."""
    exp, fake = _make_exporter(device=0)
    try:
        with patch("cuda_link.exporter.logger") as mock_logger:
            exp.record_source_sync(12345)
        mock_logger.error.assert_not_called()
        assert exp._source_sync_device_warned is False
    finally:
        exp.close()


def test_record_source_sync_wrong_device_logs_error_by_default() -> None:
    """Wrong device logs error but does NOT raise when strict_device is off."""
    exp, fake = _make_exporter(device=0)
    try:
        with (
            patch.object(fake, "get_device", return_value=1),
            patch("cuda_link.exporter.logger") as mock_logger,
        ):
            exp.record_source_sync(12345)
        mock_logger.error.assert_called_once()
        assert "ExportPolicy(strict_device=True)" in mock_logger.error.call_args[0][0]
        assert exp._source_sync_device_warned is True
    finally:
        exp.close()


def test_record_source_sync_wrong_device_raises_when_strict() -> None:
    """strict_device=True: wrong device raises ValueError."""
    exp, fake = _make_exporter(device=0, strict=True)
    try:
        with (
            patch.object(fake, "get_device", return_value=1),
            pytest.raises(ValueError, match="does not match exporter device"),
        ):
            exp.record_source_sync(12345)
    finally:
        exp.close()


def test_record_source_sync_wrong_device_warns_once() -> None:
    """Device mismatch is logged at most once per exporter instance."""
    exp, fake = _make_exporter(device=0)
    try:
        with (
            patch.object(fake, "get_device", return_value=1),
            patch("cuda_link.exporter.logger") as mock_logger,
        ):
            exp.record_source_sync(12345)
            exp.record_source_sync(12345)
            exp.record_source_sync(12345)
        assert mock_logger.error.call_count == 1
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# C2b: export pointer device/type check
# ---------------------------------------------------------------------------


def test_export_valid_device_ptr_no_warning() -> None:
    """No error when pointer is device memory on the correct device."""
    exp, fake = _make_exporter(device=0)
    try:
        with patch("cuda_link.exporter.logger") as mock_logger:
            outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        from cuda_link import FrameOutcome

        assert outcome != FrameOutcome.FAILED
        mock_logger.error.assert_not_called()
    finally:
        exp.close()


def test_export_managed_ptr_no_warning() -> None:
    """Managed memory (type=3) is treated as valid for D2D."""
    exp, fake = _make_exporter(device=0)
    try:
        with (
            patch.object(fake, "pointer_get_attributes", return_value=_attrs(type=3, device=0)),
            patch("cuda_link.exporter.logger") as mock_logger,
        ):
            outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        from cuda_link import FrameOutcome

        assert outcome != FrameOutcome.FAILED
        mock_logger.error.assert_not_called()
    finally:
        exp.close()


def test_export_host_ptr_logs_error() -> None:
    """Host pointer (type=1) logs an error in non-strict mode."""
    exp, fake = _make_exporter(device=0)
    try:
        with (
            patch.object(fake, "pointer_get_attributes", return_value=_attrs(type=1, device=0)),
            patch("cuda_link.exporter.logger") as mock_logger,
        ):
            outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        from cuda_link import FrameOutcome

        assert outcome != FrameOutcome.FAILED  # continues in non-strict mode
        mock_logger.error.assert_called_once()
    finally:
        exp.close()


def test_export_host_ptr_raises_when_strict() -> None:
    """Strict mode raises ValueError for host pointer."""
    exp, fake = _make_exporter(device=0, strict=True)
    try:
        with (
            patch.object(fake, "pointer_get_attributes", return_value=_attrs(type=1, device=0)),
            pytest.raises(ValueError, match="not device/managed memory"),
        ):
            exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
    finally:
        exp.close()


def test_export_wrong_device_ptr_logs_error() -> None:
    """Pointer on wrong device logs error in non-strict mode."""
    exp, fake = _make_exporter(device=0)
    try:
        with (
            patch.object(fake, "pointer_get_attributes", return_value=_attrs(type=2, device=1)),
            patch("cuda_link.exporter.logger") as mock_logger,
        ):
            outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        from cuda_link import FrameOutcome

        assert outcome != FrameOutcome.FAILED
        mock_logger.error.assert_called_once()
        assert "belongs to device" in mock_logger.error.call_args[0][0]
    finally:
        exp.close()


def test_export_wrong_device_ptr_raises_when_strict() -> None:
    """Strict mode raises ValueError when pointer is on wrong device."""
    exp, fake = _make_exporter(device=0, strict=True)
    try:
        with (
            patch.object(fake, "pointer_get_attributes", return_value=_attrs(type=2, device=1)),
            pytest.raises(ValueError, match="belongs to device"),
        ):
            exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
    finally:
        exp.close()


def test_export_ptr_check_cached_after_first_call() -> None:
    """pointer_get_attributes is called once per unique pointer, not every frame."""

    exp, fake = _make_exporter(device=0)
    try:
        ptr = 0xABCD0000
        call_count = 0
        real_method = fake.pointer_get_attributes

        def counting_get_attributes(p):
            nonlocal call_count
            call_count += 1
            return real_method(p)

        fake.pointer_get_attributes = counting_get_attributes
        exp.export(GpuFrame(ptr=ptr, size=_DATA_SIZE))
        exp.export(GpuFrame(ptr=ptr, size=_DATA_SIZE))
        exp.export(GpuFrame(ptr=ptr, size=_DATA_SIZE))
        assert call_count == 1
    finally:
        exp.close()


def test_export_ptr_cache_bounded_at_sixty_four() -> None:
    """Pointer cache never exceeds 64 entries."""
    exp, _ = _make_exporter(device=0)
    try:
        for i in range(70):
            exp.export(GpuFrame(ptr=0x1000 + i, size=_DATA_SIZE))
        assert len(exp._ptr_device_cache) <= 64
    finally:
        exp.close()


def test_export_env_gate_strict_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_STRICT_DEVICE env var is reflected in ExportPolicy.from_env()."""
    monkeypatch.setenv("CUDALINK_STRICT_DEVICE", "1")
    policy = ExportPolicy.from_env()
    assert policy.strict_device is True


def test_export_env_gate_strict_device_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_STRICT_DEVICE defaults to 0 (warn, not raise)."""
    monkeypatch.delenv("CUDALINK_STRICT_DEVICE", raising=False)
    policy = ExportPolicy.from_env()
    assert policy.strict_device is False


# ---------------------------------------------------------------------------
# F10: heartbeat-preserving F9 early-return
# ---------------------------------------------------------------------------


def test_f10_f9_skip_clears_shutdown_flag() -> None:
    """F9 skip path must zero shutdown_flag even when bypassing the heartbeat.

    Regression test for the false 'Sender shutdown detected' caused by F9's
    early-return bypassing the per-frame shutdown_flag=0 reassertion.
    """
    from cuda_link import FrameOutcome

    active_barrier = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=FakeShmAdapter(active_count=1))
    exp, _ = _make_exporter(device=0, barrier=active_barrier)
    try:
        exp.shm_handle.buf[exp._shutdown_offset] = 1  # stale flag

        outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))

        assert outcome == FrameOutcome.SKIPPED_BARRIER
        assert exp.shm_handle.buf[exp._shutdown_offset] == 0
    finally:
        exp.close()


def test_f10_f9_skip_does_not_advance_write_idx() -> None:
    """F9 skip must not increment write_idx (no phantom frame published)."""
    active_barrier = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=FakeShmAdapter(active_count=1))
    exp, _ = _make_exporter(device=0, barrier=active_barrier)
    try:
        initial_write_idx = exp.write_idx

        exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))

        assert exp.write_idx == initial_write_idx
    finally:
        exp.close()
