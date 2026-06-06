"""
Tests for the unordered-export safeguard introduced in the
feature/unordered-export-safeguard branch.

Covers:
  C1. warns once when export() is called without producer-stream ordering.
  C2. no warning when ordering is armed (GpuFrame.producer_stream or record_source_sync).
  C3. strict mode raises ValueError instead of warning.
  C4. _ordering_armed gates stream_wait_event (arch item B): never called when not
      armed; called when armed (covers the legacy copy path gate replacement).

All tests are GPU-free: they use FakeCUDAAdapter + Exporter.open(..., cuda=fake).
Pattern reused from tests/core/test_device_affinity.py:29-54 (_make_exporter) and
the warns-once idiom at :118-131.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from cuda_link import ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.exporter import Exporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_SIZE = 8 * 8 * 4  # H=8, W=8, C=4, uint8


def _make_exporter(
    device: int = 0,
    require_source_sync: bool = False,
) -> tuple[Exporter, FakeCUDAAdapter]:
    """Open a GPU-free Exporter for unordered-export safeguard tests.

    Returns (exporter, fake_cuda). Always call exporter.close() in a finally block.
    """
    shm_name = f"test_unordered_{uuid.uuid4().hex[:8]}"
    fake = FakeCUDAAdapter(device=device)
    policy = ExportPolicy(
        export_sync=False,
        use_graphs=False,
        flush_probe=False,
        strict_device=False,
        require_source_sync=require_source_sync,
        barrier_enabled=False,
        high_priority_stream=False,
        export_profile=False,
    )
    exp = Exporter.open(
        FrameSpec(shm_name=shm_name, height=8, width=8, channels=4, dtype="uint8", num_slots=2, device=device),
        policy=policy,
        cuda=fake,
    )
    return exp, fake


# ---------------------------------------------------------------------------
# C1: warns once when no ordering is armed
# ---------------------------------------------------------------------------


def test_warns_once_on_unordered_export() -> None:
    """Two unordered export() calls emit exactly one logger.warning."""
    exp, _ = _make_exporter()
    try:
        with patch("cuda_link.exporter.logger") as mock_log:
            outcome1 = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
            outcome2 = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert mock_log.warning.call_count == 1
        assert "producer-stream ordering" in mock_log.warning.call_args[0][0]
        assert outcome1 != FrameOutcome.FAILED
        assert outcome2 != FrameOutcome.FAILED
    finally:
        exp.close()


def test_no_error_logged_on_unordered_export() -> None:
    """Unordered export emits a warning — NOT an error."""
    exp, _ = _make_exporter()
    try:
        with patch("cuda_link.exporter.logger") as mock_log:
            exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        mock_log.error.assert_not_called()
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# C2: no warning when ordering is armed
# ---------------------------------------------------------------------------


def test_no_warning_when_producer_stream_in_frame() -> None:
    """GpuFrame(producer_stream=0) arms ordering — no warning emitted."""
    exp, _ = _make_exporter()
    try:
        with patch("cuda_link.exporter.logger") as mock_log:
            exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE, producer_stream=0))
        # Filter to unordered-export warning specifically
        unordered_calls = [
            c for c in mock_log.warning.call_args_list if "producer-stream ordering" in (c[0][0] if c[0] else "")
        ]
        assert len(unordered_calls) == 0
    finally:
        exp.close()


def test_no_warning_when_record_source_sync_called_first() -> None:
    """record_source_sync() before export() arms ordering — no warning emitted."""
    exp, _ = _make_exporter()
    try:
        exp.record_source_sync(0)
        with patch("cuda_link.exporter.logger") as mock_log:
            exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        unordered_calls = [
            c for c in mock_log.warning.call_args_list if "producer-stream ordering" in (c[0][0] if c[0] else "")
        ]
        assert len(unordered_calls) == 0
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# C3: strict mode raises ValueError
# ---------------------------------------------------------------------------


def test_strict_raises_on_unordered_export() -> None:
    """require_source_sync=True raises ValueError on unordered export."""
    exp, _ = _make_exporter(require_source_sync=True)
    try:
        with pytest.raises(ValueError, match="producer-stream ordering"):
            exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
    finally:
        exp.close()


def test_strict_no_raise_when_ordered() -> None:
    """require_source_sync=True does NOT raise when producer_stream is provided."""
    exp, _ = _make_exporter(require_source_sync=True)
    try:
        outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE, producer_stream=0))
        assert outcome != FrameOutcome.FAILED
    finally:
        exp.close()


def test_strict_no_raise_when_record_source_sync_called() -> None:
    """require_source_sync=True does NOT raise after record_source_sync()."""
    exp, _ = _make_exporter(require_source_sync=True)
    try:
        exp.record_source_sync(0)
        outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert outcome != FrameOutcome.FAILED
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# C4: _ordering_armed gates stream_wait_event (arch item B)
# ---------------------------------------------------------------------------


def _count_stream_waits(fake: FakeCUDAAdapter) -> list[int]:
    """Wrap fake.stream_wait_event with a call counter; return a 1-element count list."""
    count = [0]
    real_swe = fake.stream_wait_event

    def counting_swe(stream, event, flags=0):
        count[0] += 1
        return real_swe(stream, event, flags)

    fake.stream_wait_event = counting_swe
    return count


def test_stream_wait_event_not_called_when_ordering_not_armed() -> None:
    """When no ordering is armed, stream_wait_event is never called (legacy path)."""
    exp, fake = _make_exporter()
    count = _count_stream_waits(fake)
    try:
        with patch("cuda_link.exporter.logger"):  # suppress the warning
            exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert count[0] == 0
    finally:
        exp.close()


def test_stream_wait_event_called_when_ordering_armed() -> None:
    """When ordering is armed via producer_stream, stream_wait_event is called."""
    exp, fake = _make_exporter()
    count = _count_stream_waits(fake)
    try:
        exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE, producer_stream=0))
        assert count[0] >= 1
    finally:
        exp.close()
