"""
Coverage tests for Exporter.export()'s general control-flow branches not tied
to CUDA Graphs or the doorbell: not-initialized/size-mismatch early returns,
the barrier-skip-with-no-shm-handle edge case, record_source_sync()'s no-op
guard, the legacy-path profile-timing branches + periodic [PROFILE] summary
log, and the generic (OSError, RuntimeError) failure path.

All tests use FakeCUDAAdapter — no GPU required.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

from fakes import FakeShmAdapter

from cuda_link import ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.activation_barrier import CheckerBarrier
from cuda_link.exporter import Exporter

_H = _W = _C = 4
_DATA_SIZE = _H * _W * _C


def _spec(*, num_slots: int = 1) -> FrameSpec:
    return FrameSpec(
        shm_name=f"test_cov_export_{uuid.uuid4().hex[:8]}",
        height=_H,
        width=_W,
        channels=_C,
        dtype="uint8",
        num_slots=num_slots,
        device=0,
    )


# ---------------------------------------------------------------------------
# export() — not-initialized early return (551-552)
# ---------------------------------------------------------------------------


def test_export_returns_failed_when_not_initialized() -> None:
    exp = Exporter(_spec(), ExportPolicy.for_testing(), FakeCUDAAdapter())
    with patch("cuda_link.exporter.logger") as mock_logger:
        outcome = exp.export(GpuFrame(ptr=0x1000, size=_DATA_SIZE))
    assert outcome == FrameOutcome.FAILED
    mock_logger.warning.assert_any_call("Exporter not initialized")


# ---------------------------------------------------------------------------
# export() — size-mismatch early return (554-555)
# ---------------------------------------------------------------------------


def test_export_size_mismatch_returns_failed() -> None:
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=fake)
    try:
        with patch("cuda_link.exporter.logger") as mock_logger:
            outcome = exp.export(GpuFrame(ptr=0x1000, size=_DATA_SIZE + 1))
        assert outcome == FrameOutcome.FAILED
        mock_logger.error.assert_called_once_with("Size mismatch: expected %d, got %d", _DATA_SIZE, _DATA_SIZE + 1)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# export() — barrier skip with shm_handle=None (559->562)
# ---------------------------------------------------------------------------


def test_export_barrier_skip_with_no_shm_handle_does_not_crash() -> None:
    """A never-_initialize()'d Exporter has shm_handle=None; the barrier-skip
    path must not attempt clear_shutdown() against a missing SHM buffer."""
    exp = Exporter(_spec(), ExportPolicy.for_testing(), FakeCUDAAdapter())
    exp._initialized = True  # bypass the "not initialized" early return
    exp._barrier = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=FakeShmAdapter(active_count=1))

    outcome = exp.export(GpuFrame(ptr=0x1000, size=_DATA_SIZE))

    assert outcome == FrameOutcome.SKIPPED_BARRIER
    assert exp.shm_handle is None


# ---------------------------------------------------------------------------
# record_source_sync() — no-op when source_sync_event was never created (508->exit)
# ---------------------------------------------------------------------------


def test_record_source_sync_no_op_when_event_not_created() -> None:
    exp = Exporter(_spec(), ExportPolicy.for_testing(), FakeCUDAAdapter())
    assert exp.source_sync_event is None
    exp.record_source_sync(999)  # must be a safe no-op
    assert exp._source_sync_recorded is False


# ---------------------------------------------------------------------------
# export() — legacy-path profile timing + periodic [PROFILE] summary (97th frame)
# (572, 663, 667-668, 678-679, 684, 688, 691, 694, 697, 700-705, 708, 715, 720-735)
# ---------------------------------------------------------------------------


def test_export_legacy_path_profiled_covers_timing_and_periodic_summary() -> None:
    policy = ExportPolicy(
        export_sync=False,
        use_graphs=False,
        flush_probe=True,
        strict_device=False,
        require_source_sync=False,
        barrier_enabled=False,
        high_priority_stream=False,
        export_profile=True,
        doorbell=False,
    )
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(), policy=policy, cuda=fake)
    try:
        # Force the "not ipc_events[slot]" True branch so the profiled
        # stream_synchronize block (688, 691) is exercised every frame.
        exp.ipc_events[0] = None
        exp.record_source_sync(0)

        with patch("cuda_link.exporter.logger") as mock_logger:
            for _ in range(97):
                outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
                assert outcome == FrameOutcome.PUBLISHED

        assert exp.frame_count == 97
        messages = [str(c.args[0]) for c in mock_logger.debug.call_args_list if c.args]
        assert any("[PROFILE]" in m for m in messages)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# export() — flush_probe branch without profiling (700->702, 704->707 False direction)
# ---------------------------------------------------------------------------


def test_export_flush_probe_without_profiling_skips_timing_capture() -> None:
    """flush_probe=True + export_sync=False enters the flush-probe block; with
    export_profile=False the two inner `if profile:` guards must take their
    False branch (no timing recorded) while stream_query() still runs."""
    policy = ExportPolicy(
        export_sync=False,
        use_graphs=False,
        flush_probe=True,
        strict_device=False,
        require_source_sync=False,
        barrier_enabled=False,
        high_priority_stream=False,
        export_profile=False,
        doorbell=False,
    )
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(), policy=policy, cuda=fake)
    try:
        outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert outcome == FrameOutcome.PUBLISHED
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# export() — unexpected (OSError, RuntimeError) is caught and returns FAILED (752-753)
# ---------------------------------------------------------------------------


def test_export_unexpected_cuda_error_returns_failed() -> None:
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=fake)
    try:
        with (
            patch.object(fake, "memcpy_async", side_effect=RuntimeError("simulated CUDA failure")),
            patch("cuda_link.exporter.logger") as mock_logger,
        ):
            outcome = exp.export(GpuFrame(ptr=0x1000, size=_DATA_SIZE))
        assert outcome == FrameOutcome.FAILED
        mock_logger.exception.assert_called_once_with("Export failed")
    finally:
        exp.close()
