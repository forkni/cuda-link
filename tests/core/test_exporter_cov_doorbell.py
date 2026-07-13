"""
Coverage tests for the Win32 doorbell integration in Exporter (R2).

_doorbell.create_doorbell / signal / close are patched directly at the
cuda_link.exporter._doorbell module reference so these tests are deterministic
and platform-independent (real Win32 behaviour differs between a Windows dev
box and the ubuntu-latest no-GPU CI runner, where the module's public
functions are documented no-ops).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from cuda_link import ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.exporter import Exporter

_H = _W = _C = 4
_DATA_SIZE = _H * _W * _C


def _spec() -> FrameSpec:
    return FrameSpec(
        shm_name=f"test_cov_doorbell_{uuid.uuid4().hex[:8]}",
        height=_H,
        width=_W,
        channels=_C,
        dtype="uint8",
        num_slots=1,
        device=0,
    )


def _doorbell_policy() -> ExportPolicy:
    return ExportPolicy(
        export_sync=False,
        use_graphs=False,
        flush_probe=False,
        strict_device=False,
        require_source_sync=False,
        barrier_enabled=False,
        high_priority_stream=False,
        export_profile=False,
        doorbell=True,
    )


def test_doorbell_creation_failure_logs_warning() -> None:
    with (
        patch("cuda_link.exporter._doorbell.create_doorbell", return_value=None),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp = Exporter.open(_spec(), policy=_doorbell_policy(), cuda=FakeCUDAAdapter())
    try:
        assert exp._doorbell is None
        messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
        assert any("Doorbell creation failed" in m for m in messages)
    finally:
        exp.close()


def test_doorbell_full_lifecycle_signals_and_closes() -> None:
    """Successful doorbell creation -> export() signals it -> close() signals + closes it."""
    sentinel = MagicMock(name="doorbell_handle")
    with (
        patch("cuda_link.exporter._doorbell.create_doorbell", return_value=sentinel) as mock_create,
        patch("cuda_link.exporter._doorbell.signal") as mock_signal,
        patch("cuda_link.exporter._doorbell.close") as mock_close,
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp = Exporter.open(_spec(), policy=_doorbell_policy(), cuda=FakeCUDAAdapter())
        assert exp._doorbell is sentinel
        messages = [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]
        assert any("Doorbell created" in m for m in messages)

        outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert outcome == FrameOutcome.PUBLISHED
        mock_signal.assert_any_call(sentinel)  # signalled from export()

        exp.close()

    assert mock_create.call_count == 1
    # signal() fires at least twice: once from export(), once from _do_cleanup() STEP 1.
    assert mock_signal.call_count >= 2
    mock_close.assert_called_once_with(sentinel)
    assert exp._doorbell is None
