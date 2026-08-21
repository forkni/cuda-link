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
from unittest.mock import ANY, patch

import pytest

from cuda_link import ExportPolicy, FrameSpec
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.activation_barrier import CheckerBarrier
from cuda_link.cuda_runtime_types import CudaIpcError
from cuda_link.exporter import IPC_HANDLE_SIZE, CTypesCUDAAdapter, Exporter, _read_hws_mode

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
    """A CUDA context bound to a different device raises; open()'s except-clause
    calls _do_cleanup(cuda_valid=False) before re-raising (236)."""
    fake = FakeCUDAAdapter(device=1)  # spec asks for device 0
    original_do_cleanup = Exporter._do_cleanup
    with (
        patch.object(Exporter, "_do_cleanup", autospec=True, side_effect=original_do_cleanup) as mock_cleanup,
        pytest.raises(CudaIpcError, match="Device mismatch"),
    ):
        Exporter.open(_spec(device=0), policy=ExportPolicy.for_testing(), cuda=fake)
    mock_cleanup.assert_called_once_with(ANY, cuda_valid=False)


# ---------------------------------------------------------------------------
# _initialize() — check_ipc_capability() wiring
# ---------------------------------------------------------------------------


def test_initialize_logs_ipc_capability_note_when_present() -> None:
    """A non-None check_ipc_capability() return is logged via logger.info before
    any IPC handle is minted (see the ipc_capability_note comment in _initialize())."""
    fake = FakeCUDAAdapter(device=0)
    fake.check_ipc_capability = lambda device=None: "WDDM driver model; legacy CUDA IPC is undocumented here."
    with patch("cuda_link.exporter.logger") as mock_logger:
        exp = Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=fake)
    try:
        messages = [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]
        assert any("WDDM driver model" in m for m in messages)
    finally:
        exp.close()


def test_initialize_propagates_ipc_capability_failure_and_cleans_up() -> None:
    """cudaDevAttrIpcEventSupport == 0 -> check_ipc_capability() raises -> open()
    propagates it and still runs cleanup (mirrors the device-mismatch test above)."""
    fake = FakeCUDAAdapter(device=0)
    fake.fail_ipc_capability = True
    original_do_cleanup = Exporter._do_cleanup
    with (
        patch.object(Exporter, "_do_cleanup", autospec=True, side_effect=original_do_cleanup) as mock_cleanup,
        pytest.raises(RuntimeError, match="simulated cudaDevAttrIpcEventSupport=0"),
    ):
        Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=fake)
    mock_cleanup.assert_called_once_with(ANY, cuda_valid=False)


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


class _DistinctiveEventHandle:
    """A fake IPC event handle whose bytes are NOT all-zero.

    FakeCUDAAdapter's own _FakeIpcHandle.reserved is always bytes(64) (all-zero),
    which is indistinguishable from an untouched (never-written) SHM region. This
    handle uses a non-zero fill so the write-vs-skip branches are actually
    distinguishable by inspecting the SHM buffer contents.
    """

    reserved = b"\xab" * IPC_HANDLE_SIZE


def test_write_handles_skips_falsy_event_handle_slot() -> None:
    """A falsy ipc_event_handles[slot] (e.g. event-handle export failed) is skipped:
    its SHM region is left untouched (374), while a truthy sibling slot's region IS
    written with that slot's handle bytes."""
    fake = FakeCUDAAdapter(device=0)
    truthy_handle = _DistinctiveEventHandle()
    with patch.object(fake, "ipc_get_event_handle", side_effect=[None, truthy_handle]):
        exp = Exporter.open(_spec(num_slots=2), policy=ExportPolicy.for_testing(), cuda=fake)
    try:
        assert exp.ipc_event_handles == [None, truthy_handle]
        assert exp._initialized is True

        evt_off_0 = exp._layout.event_handle_offset(0)
        evt_off_1 = exp._layout.event_handle_offset(1)
        # Slot 0 was skipped: its region stays at the SHM segment's zero-initialized default.
        assert bytes(exp.shm_handle.buf[evt_off_0 : evt_off_0 + IPC_HANDLE_SIZE]) == bytes(IPC_HANDLE_SIZE)
        # Slot 1 was written: its region matches the distinctive (non-zero) handle bytes.
        assert bytes(exp.shm_handle.buf[evt_off_1 : evt_off_1 + IPC_HANDLE_SIZE]) == bytes(truthy_handle.reserved)
    finally:
        exp.close()
