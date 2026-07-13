"""
Coverage tests for Exporter.__init__ / Exporter.open() / _initialize() branches
not exercised elsewhere: _read_hws_mode() winreg success path, open()'s
policy/cuda defaulting, device-mismatch cleanup, the HWS=0 performance warning,
IPC-stream/source-sync-event reuse, and the early-return guards in
_write_handles_to_shm / _write_metadata_to_shm.

All tests use FakeCUDAAdapter — no GPU required.
"""

from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import patch

import pytest

from cuda_link import ExportPolicy, FrameSpec
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.activation_barrier import CheckerBarrier
from cuda_link.cuda_runtime_types import CudaIpcError
from cuda_link.exporter import CTypesCUDAAdapter, Exporter, _read_hws_mode

_H = _W = _C = 4
_DATA_SIZE = _H * _W * _C  # uint8


def _spec(*, num_slots: int = 2, device: int = 0) -> FrameSpec:
    return FrameSpec(
        shm_name=f"test_cov_init_{uuid.uuid4().hex[:8]}",
        height=_H,
        width=_W,
        channels=_C,
        dtype="uint8",
        num_slots=num_slots,
        device=device,
    )


# ---------------------------------------------------------------------------
# _read_hws_mode() — winreg success path (107-108)
# ---------------------------------------------------------------------------


def test_read_hws_mode_success_via_winreg() -> None:
    """A successful winreg read returns the raw HwSchMode value as a string.

    CI's no-GPU job runs on ubuntu-latest (no real winreg), so a fake module is
    injected into sys.modules — the local `import winreg` inside _read_hws_mode
    picks up sys.modules['winreg'] regardless of the host platform.
    """
    fake_winreg = types.ModuleType("winreg")
    fake_winreg.HKEY_LOCAL_MACHINE = object()
    fake_winreg.OpenKey = lambda *_a, **_k: "fake_key"
    fake_winreg.QueryValueEx = lambda _key, _name: (2, "REG_DWORD")
    fake_winreg.CloseKey = lambda _key: None

    with patch.dict(sys.modules, {"winreg": fake_winreg}):
        result = _read_hws_mode()

    assert result == "2"


# ---------------------------------------------------------------------------
# Exporter.open() — policy/cuda defaulting (225, 227) and device mismatch (235-237, 247)
# ---------------------------------------------------------------------------


def test_open_defaults_policy_and_cuda_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """policy=None -> ExportPolicy.from_env(); cuda=None -> CTypesCUDAAdapter.for_device()."""
    for var in (
        "CUDALINK_EXPORT_SYNC",
        "CUDALINK_USE_GRAPHS",
        "CUDALINK_EXPORT_FLUSH_PROBE",
        "CUDALINK_STRICT_DEVICE",
        "CUDALINK_REQUIRE_SOURCE_SYNC",
        "CUDALINK_ACTIVATION_BARRIER",
        "CUDALINK_BARRIER_STALE_NS",
        "CUDALINK_LIB_STREAM_PRIO",
        "CUDALINK_EXPORT_PROFILE",
        "CUDALINK_DOORBELL",
    ):
        monkeypatch.delenv(var, raising=False)

    fake = FakeCUDAAdapter(device=0)
    with patch.object(CTypesCUDAAdapter, "for_device", return_value=fake) as mock_for_device:
        # barrier passed explicitly so from_env()'s default barrier_enabled=True never
        # touches the real fixed-name activation-barrier SHM segment.
        exp = Exporter.open(_spec(), barrier=CheckerBarrier(enabled=False, stale_ns=0))
    try:
        mock_for_device.assert_called_once_with(0)
        assert exp._cuda is fake
        assert exp._policy.use_graphs is True  # ExportPolicy.from_env() default
        assert exp._initialized is True
    finally:
        exp.close()


def test_open_device_mismatch_raises_cuda_ipc_error_and_cleans_up() -> None:
    """A CUDA context bound to a different device raises; open() cleans up before re-raising."""
    fake = FakeCUDAAdapter(device=1)  # spec asks for device 0
    with pytest.raises(CudaIpcError, match="Device mismatch"):
        Exporter.open(_spec(device=0), policy=ExportPolicy.for_testing(), cuda=fake)
    # Cleanup ran with cuda_valid=False before any allocation happened.
    assert fake.allocations == {}


# ---------------------------------------------------------------------------
# _initialize() — HWS=0 performance warning (257)
# ---------------------------------------------------------------------------


def test_initialize_warns_when_hws_mode_disabled() -> None:
    with (
        patch("cuda_link.exporter._read_hws_mode", return_value="0"),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp = Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=FakeCUDAAdapter())
    try:
        assert exp._hws_mode == "0"
        messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
        assert any("PERFORMANCE" in m for m in messages)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# _initialize() — IPC-stream / source-sync-event reuse (276, 278->282)
# ---------------------------------------------------------------------------


def test_initialize_reuses_preexisting_ipc_stream_and_sync_event() -> None:
    """When ipc_stream / source_sync_event are already set, _initialize() reuses them."""
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter(_spec(), ExportPolicy.for_testing(), fake)
    sentinel_stream = fake.create_stream()
    sentinel_event = fake.create_sync_event()
    exp.ipc_stream = sentinel_stream
    exp.source_sync_event = sentinel_event

    with patch("cuda_link.exporter.logger") as mock_logger:
        exp._initialize()
    try:
        assert exp.ipc_stream is sentinel_stream
        assert exp.source_sync_event is sentinel_event
        messages = [str(c.args[0]) for c in mock_logger.debug.call_args_list if c.args]
        assert any("Reusing IPC stream" in m for m in messages)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# _write_handles_to_shm / _write_metadata_to_shm — early-return guards (363, 381)
# ---------------------------------------------------------------------------


def test_write_helpers_are_no_ops_before_shm_handle_exists() -> None:
    """Calling the write helpers before shm_handle is set must be a safe no-op."""
    exp = Exporter(_spec(), ExportPolicy.for_testing(), FakeCUDAAdapter())
    assert exp.shm_handle is None
    exp._write_handles_to_shm()  # covers line 363 early return
    exp._write_metadata_to_shm()  # covers line 381 early return
    assert exp.shm_handle is None  # no crash, no state mutation


# ---------------------------------------------------------------------------
# _write_handles_to_shm — falsy ipc_event_handles entry (374->368)
# ---------------------------------------------------------------------------


def test_write_handles_skips_falsy_event_handle_slot() -> None:
    """A falsy ipc_event_handles[slot] (e.g. event-handle export failed) is skipped, not written."""
    fake = FakeCUDAAdapter(device=0)
    with patch.object(fake, "ipc_get_event_handle", return_value=None):
        exp = Exporter.open(_spec(num_slots=2), policy=ExportPolicy.for_testing(), cuda=fake)
    try:
        assert exp.ipc_event_handles == [None, None]
        assert exp._initialized is True
    finally:
        exp.close()
