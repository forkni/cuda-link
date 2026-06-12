"""
Lifecycle integration tests for Importer — no GPU required.

These tests use FakeCUDAAdapter and a synthetic SHM buffer to exercise the
full open→get_frame→close lifecycle without a real CUDA device.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest
from fakes import make_connected_importer, make_fake_ipc_connection

from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
from cuda_link.importer import Format, Importer

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_connected_importer(
    shape: tuple = (8, 8, 4),
    dtype: str = "uint8",
    num_slots: int = 1,
    write_idx: int = 1,
    ipc_version: int = 1,
    timeout_ms: float = 5000.0,
    spin_us: int = 0,
    last_write_idx: int = 0,
    debug: bool = False,
    cupy: object | None = None,
    numpy: object | None = None,
) -> Importer:
    """Build an Importer with a fully-wired IPCConnection (no real SHM / CUDA).

    Delegates to ``fakes.make_connected_importer`` — a thin wrapper kept for
    call-site readability within this file.
    """
    return make_connected_importer(
        shape=shape,
        dtype=dtype,
        num_slots=num_slots,
        write_idx=write_idx,
        ipc_version=ipc_version,
        timeout_ms=timeout_ms,
        spin_us=spin_us,
        last_write_idx=last_write_idx,
        debug=debug,
        cupy=cupy,
        numpy=numpy,
    )


# ---------------------------------------------------------------------------
# ImportOutcome coverage
# ---------------------------------------------------------------------------


def test_outcome_no_frame_when_write_idx_unchanged() -> None:
    """write_idx == last_write_idx → NO_FRAME."""
    imp = _make_connected_importer(write_idx=1, last_write_idx=1)

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.NO_FRAME
    assert result.frame is None


def test_outcome_shutdown_closes_importer() -> None:
    """SHUTDOWN outcome triggers close() and returns SHUTDOWN."""
    imp = _make_connected_importer(write_idx=1)
    # Set shutdown flag in the SHM buffer
    shutdown_offset = imp._conn.layout.shutdown_offset
    imp._conn.shm_handle.buf[shutdown_offset] = 1

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.SHUTDOWN
    assert not imp._initialized


def test_outcome_new_frame_numpy() -> None:
    """NEW_FRAME: get_frame_numpy returns ImportResult with a numpy array."""
    import numpy as np

    shape = (4, 4, 4)
    imp = _make_connected_importer(shape=shape, dtype="uint8", write_idx=1)
    imp._conn.ipc_events[0] = None  # no event → cuda.synchronize() path

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.NEW_FRAME
    assert isinstance(result.frame, np.ndarray)
    assert result.frame.shape == shape


def test_outcome_reconnecting_on_version_bump() -> None:
    """VERSION_CHANGED in SHM → RECONNECTING outcome (single-tick stall)."""
    imp = _make_connected_importer(write_idx=2, ipc_version=1)

    # Bump ipc_version in SHM buffer to simulate producer restart
    struct.pack_into("<Q", imp._conn.shm_handle.buf, 4, 2)

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.RECONNECTING
    assert result.frame is None


def test_outcome_timeout_raises_no_exception_returns_result() -> None:
    """TimeoutError inside _wait_for_slot → TIMEOUT outcome (no exception propagated)."""
    imp = _make_connected_importer(write_idx=1, timeout_ms=1.0)
    # Give slot an event that we can make timeout
    mock_event = MagicMock()
    imp._conn.ipc_events[0] = mock_event
    imp._conn.cuda.query_event.return_value = False  # never ready

    with patch("cuda_link.importer.time") as mock_time:
        mock_time.perf_counter.side_effect = [0.0] + [10.0] * 20
        mock_time.sleep = MagicMock()
        result = imp.get_frame_numpy()

    assert result.outcome is ImportOutcome.TIMEOUT
    assert result.frame is None


# ---------------------------------------------------------------------------
# Idempotent close
# ---------------------------------------------------------------------------


def test_close_is_idempotent() -> None:
    """close() can be called multiple times without error."""
    imp = _make_connected_importer()
    imp.close()
    imp.close()  # second call must not raise
    assert not imp._initialized


def test_close_clears_initialized_flag() -> None:
    imp = _make_connected_importer()
    assert imp._initialized
    imp.close()
    assert not imp._initialized


def test_release_connection_state_shared_teardown_path() -> None:
    """close() and _partial_cleanup_for_reconnect() share _release_connection_state (C2-slice).

    Both callers must null _numpy/_conn and clear _initialized.
    Only the reconnect path calls request_immediate_reconnect() — close() must not.
    """
    from unittest.mock import patch

    # close() path: _conn/_numpy nulled, _initialized cleared; request_immediate_reconnect NOT called.
    imp_close = _make_connected_importer()
    assert imp_close._conn is not None
    assert imp_close._numpy is not None
    assert imp_close._initialized
    with patch.object(imp_close._retry, "request_immediate_reconnect") as mock_reconnect:
        imp_close.close()
        mock_reconnect.assert_not_called()
    assert imp_close._conn is None
    assert imp_close._numpy is None
    assert not imp_close._initialized

    # _partial_cleanup_for_reconnect() path: same nulls + request_immediate_reconnect IS called.
    imp_reconnect = _make_connected_importer()
    with patch.object(imp_reconnect._retry, "request_immediate_reconnect") as mock_reconnect:
        imp_reconnect._partial_cleanup_for_reconnect()
        mock_reconnect.assert_called_once()
    assert imp_reconnect._conn is None
    assert imp_reconnect._numpy is None
    assert not imp_reconnect._initialized


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_calls_close() -> None:
    """with Importer as imp: calls close() on exit."""
    imp = _make_connected_importer()
    with imp:
        assert imp._initialized
    assert not imp._initialized


def test_context_manager_calls_close_on_exception() -> None:
    """close() is called even when the body raises."""
    imp = _make_connected_importer()
    with pytest.raises(RuntimeError, match="test error"), imp:
        raise RuntimeError("test error")
    assert not imp._initialized


# ---------------------------------------------------------------------------
# open() factory — default policy / adapter selection
# ---------------------------------------------------------------------------


def test_open_raises_on_missing_shm() -> None:
    """open() raises FileNotFoundError when SHM does not exist."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_xyzzy_12345")
    policy = ImportPolicy.for_testing()
    fake = FakeCUDAAdapter()
    with pytest.raises(FileNotFoundError):
        Importer.open(spec, policy=policy, cuda=fake)


def test_open_uses_for_testing_policy() -> None:
    """ImportPolicy.for_testing() is safe as a default for unit tests."""
    pol = ImportPolicy.for_testing()
    assert pol.wait_spin_us == 0
    assert pol.allow_pageable_fallback is True


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------


def test_is_ready_when_connected() -> None:
    imp = _make_connected_importer()
    assert imp.is_ready()


def test_is_ready_false_after_close() -> None:
    imp = _make_connected_importer()
    imp.close()
    assert not imp.is_ready()


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


def test_get_stats_structure() -> None:
    imp = _make_connected_importer(shape=(8, 8, 4), dtype="float32")
    stats = imp.get_stats()

    for key in (
        "initialized",
        "shm_name",
        "shape",
        "dtype",
        "device",
        "num_slots",
        "frame_count",
        "torch_available",
        "numpy_available",
        "dev_ptrs",
        "wait_spin_hits",
        "wait_sleep_hits",
        "avg_spin_us",
        "avg_sleep_us",
    ):
        assert key in stats, f"Missing key: {key}"

    assert stats["initialized"] is True
    assert stats["shape"] == (8, 8, 4)


def test_get_stats_zero_counters_no_division_error() -> None:
    imp = _make_connected_importer()
    stats = imp.get_stats()
    assert stats["avg_spin_us"] == 0.0
    assert stats["avg_sleep_us"] == 0.0


# ---------------------------------------------------------------------------
# Reconnect-wait (PR-D)
# ---------------------------------------------------------------------------


def _reconnect_policy(**kwargs) -> ImportPolicy:
    """ImportPolicy with reconnect enabled but minimal waits for tests."""
    return ImportPolicy(
        wait_spin_us=0,
        d2h_num_streams=1,
        d2h_stream_high_priority=False,
        allow_pageable_fallback=True,
        debug=False,
        reconnect_enabled=True,
        reconnect_max_attempts=kwargs.get("reconnect_max_attempts", 20),
        reconnect_backoff_frames=kwargs.get("reconnect_backoff_frames", (1, 2, 4, 8, 16, 32, 64, 120)),
    )


def test_reconnect_open_returns_waiting_importer_on_missing_shm() -> None:
    """With reconnect_enabled=True, open() does not raise — returns Importer in waiting state."""
    from cuda_link.importer import _RetryState

    spec = ImportSpec(shm_name="definitely_does_not_exist_reconnect_test_abc")
    policy = _reconnect_policy()
    fake = FakeCUDAAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)

    assert not imp._initialized
    assert imp._retry is not None
    assert isinstance(imp._retry, _RetryState)


def test_reconnect_disabled_preserves_old_raise_behavior() -> None:
    """With reconnect_enabled=False, open() raises FileNotFoundError on missing SHM."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_reconnect_disabled_test")
    policy = ImportPolicy.for_testing()  # reconnect_enabled=False
    fake = FakeCUDAAdapter()

    with pytest.raises(FileNotFoundError):
        Importer.open(spec, policy=policy, cuda=fake)


def test_reconnect_get_frame_numpy_returns_reconnecting_when_not_initialized() -> None:
    """get_frame_numpy() returns RECONNECTING while waiting for producer."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_reconnect_numpy_test")
    policy = _reconnect_policy()
    fake = FakeCUDAAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)
    assert not imp._initialized

    result = imp.get_frame_numpy()
    assert result.outcome is ImportOutcome.RECONNECTING
    assert result.frame is None


def test_reconnect_backoff_caps_at_max_attempts() -> None:
    """After max_attempts, get_frame_numpy() still returns RECONNECTING (no crash)."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_backoff_test_xyzzy")
    policy = _reconnect_policy(reconnect_max_attempts=3, reconnect_backoff_frames=(1, 1, 1))
    fake = FakeCUDAAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)

    # Drive enough frames to exhaust max_attempts (each attempt needs 1 frame to elapse)
    # Attempt 1 at frame 1, attempt 2 at frame 2, attempt 3 at frame 3 → then silent
    for _ in range(10):
        result = imp.get_frame_numpy()
        assert result.outcome is ImportOutcome.RECONNECTING

    # No crash — silently retrying beyond max_attempts
    assert imp._retry.connect_attempts > 3


def test_reconnect_request_immediate_skips_backoff() -> None:
    """request_immediate_reconnect() causes the next call to attempt connect immediately."""
    spec = ImportSpec(shm_name="definitely_does_not_exist_immediate_test_xyzzy")
    # Use a large backoff so we can confirm immediate override
    policy = _reconnect_policy(reconnect_backoff_frames=(100, 200, 400))
    fake = FakeCUDAAdapter()

    imp = Importer.open(spec, policy=policy, cuda=fake)
    # Drive one frame to get past the first attempt and into backoff
    imp.get_frame_numpy()  # attempt 1, sets retry_interval_frames=100

    frames_before = imp._retry.frames_since_last_retry

    imp.request_immediate_reconnect()

    # After immediate reconnect, frames_since_last_retry == retry_interval_frames
    assert imp._retry.frames_since_last_retry == imp._retry.retry_interval_frames
    # And it changed from the value before
    assert imp._retry.frames_since_last_retry != frames_before or imp._retry.retry_interval_frames == 1


def test_reconnect_for_testing_policy_has_reconnect_disabled() -> None:
    """ImportPolicy.for_testing() has reconnect_enabled=False so unit tests don't wait."""
    policy = ImportPolicy.for_testing()
    assert policy.reconnect_enabled is False


# ---------------------------------------------------------------------------
# Candidate C — new dtype coverage (no GPU required)
# ---------------------------------------------------------------------------


def test_format_from_overrides_new_dtypes() -> None:
    """Format.from_overrides works for all 7 registered dtypes."""
    from cuda_link.shm_protocol import DtypeCodec

    for dtype in DtypeCodec.supported():
        fmt = Format.from_overrides((8, 8, 4), dtype)
        assert fmt.dtype_str == dtype
        assert fmt.frame_nbytes == 8 * 8 * 4 * DtypeCodec.itemsize(dtype)


def test_format_from_shm_all_dtypes() -> None:
    """Format.from_shm decodes all 7 dtypes via Metadata.read_from + DtypeCodec."""
    from cuda_link.shm_protocol import DtypeCodec, Metadata, SHMLayout

    for dtype in DtypeCodec.supported():
        layout = SHMLayout(num_slots=2)
        buf = memoryview(bytearray(layout.total_size))
        kind, bits, flags = DtypeCodec.encode(dtype)
        itemsize = DtypeCodec.itemsize(dtype)
        w, h, c = 16, 16, 4
        Metadata(
            width=w,
            height=h,
            num_comps=c,
            format_kind=kind,
            bits_per_comp=bits,
            flags=flags,
            data_size=w * h * c * itemsize,
        ).pack_into(buf, layout)

        fmt = Format.from_shm(buf, 2)
        assert fmt is not None
        assert fmt.dtype_str == dtype, f"Mismatch for {dtype}: got {fmt.dtype_str}"
        assert fmt.frame_nbytes == w * h * c * itemsize


def test_torch_buffers_int8_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """TorchBuffers.build no longer raises for int8 (previously crashed)."""
    torch = pytest.importorskip("torch")

    from cuda_link.importer import Format, TorchBuffers

    conn, _cuda, _shm = make_fake_ipc_connection(num_slots=1, ipc_version=1, write_idx=0)
    fmt = Format.from_overrides((4, 4, 4), "int8")
    # Patch out as_tensor to avoid actual GPU call
    monkeypatch.setattr("cuda_link.importer.torch.as_tensor", lambda *a, **kw: torch.zeros(4, 4, 4, dtype=torch.int8))
    tb = TorchBuffers.build(conn, fmt)
    assert tb is not None
    assert len(tb.tensors) == 1


def test_torch_buffers_int16_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """TorchBuffers.build works for int16."""
    torch = pytest.importorskip("torch")

    from cuda_link.importer import Format, TorchBuffers

    conn, _cuda, _shm = make_fake_ipc_connection(num_slots=1, ipc_version=1, write_idx=0)
    fmt = Format.from_overrides((4, 4, 4), "int16")
    monkeypatch.setattr("cuda_link.importer.torch.as_tensor", lambda *a, **kw: torch.zeros(4, 4, 4, dtype=torch.int16))
    tb = TorchBuffers.build(conn, fmt)
    assert tb is not None


def test_cupy_buffers_bfloat16_raises_clear_error() -> None:
    """CupyBuffers.build raises a clear ValueError for bfloat16 (CuPy has no bfloat16 dtype)."""

    from cuda_link.importer import CupyBuffers, Format

    conn, _cuda, _shm = make_fake_ipc_connection(num_slots=1, ipc_version=1, write_idx=0)
    fmt = Format.from_overrides((4, 4, 4), "bfloat16")
    with pytest.raises(ValueError, match="bfloat16"):
        CupyBuffers.build(conn, fmt)


def test_reinitialize_uses_format_from_shm(monkeypatch: pytest.MonkeyPatch) -> None:
    """_reinitialize routes through Format.from_shm, not a hand-written decoder."""
    import struct as _struct

    from cuda_link.importer import Format

    original_from_shm = Format.from_shm
    calls: list = []

    def spy_from_shm(shm_buf: object, num_slots: int) -> Format | None:
        calls.append(num_slots)
        return original_from_shm(shm_buf, num_slots)

    monkeypatch.setattr(Format, "from_shm", staticmethod(spy_from_shm))

    imp = _make_connected_importer(shape=(4, 4, 4), dtype="float32", num_slots=1, write_idx=1)
    # Trigger a reinitialize by bumping the SHM version
    shm_buf = imp._conn.shm_handle.buf
    # Bump version to trigger VERSION_CHANGED
    current_ver = _struct.unpack("<Q", bytes(shm_buf[4:12]))[0]
    _struct.pack_into("<Q", shm_buf, 4, current_ver + 1)

    # Call _reinitialize directly
    imp._reinitialize()

    assert len(calls) >= 1, "_reinitialize must call Format.from_shm"


# ---------------------------------------------------------------------------
# Candidate D — _begin_frame dedup + Format.__eq__ sentinel fix
# ---------------------------------------------------------------------------


def test_get_frame_cupy_logs_stats_when_debug_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_frame_cupy calls _log_frame_stats when debug=True (parity with torch/numpy paths)."""
    from unittest.mock import MagicMock, patch

    # Provide a fake cupy arrays structure
    mock_array = MagicMock()
    mock_cupy_buffers = MagicMock()
    mock_cupy_buffers.arrays = [mock_array]

    # Build a connected importer with debug=True and the mock cupy backend injected at factory time.
    imp = _make_connected_importer(
        shape=(4, 4, 4), dtype="float32", num_slots=1, write_idx=1, debug=True, cupy=mock_cupy_buffers
    )
    stats_calls: list = []
    monkeypatch.setattr(imp, "_log_frame_stats", lambda *a, **kw: stats_calls.append(a))

    # Patch cp.cuda.get_current_stream and streamWaitEvent
    with (
        patch("cuda_link.importer.cp") as mock_cp,
        patch("cuda_link.importer.CUPY_AVAILABLE", True),
    ):
        mock_stream = MagicMock()
        mock_stream.ptr = 0
        mock_cp.cuda.get_current_stream.return_value = mock_stream
        mock_cp.cuda.runtime.streamWaitEvent = MagicMock()
        mock_cp.cuda.Stream = type("Stream", (), {})

        result = imp.get_frame_cupy()

    from cuda_link._importer_port import ImportOutcome

    assert result.outcome == ImportOutcome.NEW_FRAME
    assert len(stats_calls) == 1, "get_frame_cupy must call _log_frame_stats when debug=True"
    assert stats_calls[0][0] == "cupy"


def test_reinitialize_no_numpy_teardown_on_sentinel_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """_reinitialize must NOT tear down NumpyBuffers when shape+dtype match but kind/bits/flags differ.

    Scenario: _format is override-derived (kind=bits=flags=0 sentinels) but the SHM-derived
    new_fmt has the same shape+dtype with real non-zero kind/bits/flags.
    The old full-__eq__ comparison would return False (false positive).
    The fixed shape+dtype_str comparison must return True (no change detected).
    """
    import struct as _struct
    from unittest.mock import MagicMock

    from cuda_link.shm_protocol import DtypeCodec, Metadata, SHMLayout

    shape = (4, 4, 4)
    dtype = "float32"

    # Build an importer whose self._format is override-derived (sentinel zeros),
    # injecting a mock NumpyBuffers at factory time to detect if close() is called.
    mock_numpy = MagicMock()
    mock_numpy.needs_rebuild.return_value = False
    imp = _make_connected_importer(shape=shape, dtype=dtype, num_slots=1, write_idx=1, numpy=mock_numpy)
    assert imp._format.kind == 0, "from_overrides must produce kind=0 sentinel"
    assert imp._format.bits == 0, "from_overrides must produce bits=0 sentinel"
    assert imp._format.flags == 0, "from_overrides must produce flags=0 sentinel"

    # Write SHM metadata with the SAME shape+dtype but real non-zero kind/bits/flags
    shm_buf = imp._conn.shm_handle.buf
    layout = SHMLayout(1)
    kind, bits, flags = DtypeCodec.encode(dtype)  # real: (2, 32, 0) for float32
    h, w, c = shape
    Metadata(
        width=w,
        height=h,
        num_comps=c,
        format_kind=kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=w * h * c * DtypeCodec.itemsize(dtype),
    ).pack_into(shm_buf, layout)

    # Bump SHM version so _try_acquire returns VERSION_CHANGED and _reinitialize is needed
    cur_ver = _struct.unpack("<Q", bytes(shm_buf[4:12]))[0]
    _struct.pack_into("<Q", shm_buf, 4, cur_ver + 1)

    # Call _reinitialize directly
    imp._reinitialize()

    mock_numpy.close.assert_not_called(), "NumpyBuffers must NOT be torn down — shape+dtype unchanged"


# ---------------------------------------------------------------------------
# C2-bug — importer NVML seam: attach_nvml_observer + get_stats()["nvml"]
# ---------------------------------------------------------------------------


def _make_fake_nvml_observer(snapshot_value: dict | None = None) -> object:
    """Return a minimal NVMLObserver-like fake whose snapshot() returns a known dict."""
    from unittest.mock import MagicMock

    obs = MagicMock()
    obs.snapshot.return_value = snapshot_value or {"gpu_util": 42, "mem_used": 1024}
    return obs


def test_importer_has_attach_nvml_observer_method() -> None:
    """attach_nvml_observer must be a public method on Importer (symmetric with Exporter)."""
    imp = _make_connected_importer()
    assert hasattr(imp, "attach_nvml_observer")
    assert callable(imp.attach_nvml_observer)


def test_get_stats_no_nvml_key_before_attach() -> None:
    """get_stats() has no 'nvml' key before attach_nvml_observer is called.

    Before the C2-bug fix, getattr(self, '_nvml_observer', None) would silently
    return None and the branch would be dead — this test pins the clean state.
    """
    imp = _make_connected_importer()
    stats = imp.get_stats()
    assert "nvml" not in stats


def test_get_stats_nvml_key_after_attach() -> None:
    """After attach_nvml_observer(obs), get_stats()['nvml'] == obs.snapshot()."""
    imp = _make_connected_importer()
    expected = {"gpu_util": 55, "mem_used": 2048}
    obs = _make_fake_nvml_observer(expected)

    imp.attach_nvml_observer(obs)
    stats = imp.get_stats()

    assert "nvml" in stats, "attach_nvml_observer must activate the stats branch"
    assert stats["nvml"] == expected


def test_attach_nvml_observer_calls_snapshot_each_get_stats() -> None:
    """obs.snapshot() is called on every get_stats() invocation (live telemetry)."""
    imp = _make_connected_importer()
    obs = _make_fake_nvml_observer()

    imp.attach_nvml_observer(obs)
    imp.get_stats()
    imp.get_stats()
    imp.get_stats()

    assert obs.snapshot.call_count == 3


def test_attach_nvml_observer_replaces_previous() -> None:
    """Calling attach_nvml_observer a second time replaces the first observer."""
    imp = _make_connected_importer()
    obs1 = _make_fake_nvml_observer({"source": "first"})
    obs2 = _make_fake_nvml_observer({"source": "second"})

    imp.attach_nvml_observer(obs1)
    imp.attach_nvml_observer(obs2)

    stats = imp.get_stats()
    assert stats["nvml"] == {"source": "second"}


def test_legacy_shim_attach_nvml_observer_populates_stats() -> None:
    """CUDAIPCImporter.attach_nvml_observer() delegates to Importer.attach_nvml_observer()
    via the public API (no longer pokes _nvml_observer directly).

    This tests the C2-bug fix in cuda_ipc_importer.py:
        Before: self._importer._nvml_observer = observer
        After:  self._importer.attach_nvml_observer(observer)
    """
    from fakes import make_connected_importer as _make

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    fake_importer = _make(shape=(4, 4, 4), dtype="uint8")
    expected = {"gpu_util": 77, "mem_used": 999}
    obs = _make_fake_nvml_observer(expected)

    shim = CUDAIPCImporter(shm_name="fake")
    shim._importer = fake_importer

    shim.attach_nvml_observer(obs)

    # Delegate to the inner importer
    stats = fake_importer.get_stats()
    assert "nvml" in stats, "shim.attach_nvml_observer must wire through to Importer"
    assert stats["nvml"] == expected


def test_reinitialize_numpy_teardown_on_genuine_shape_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """_reinitialize tears down NumpyBuffers when the shape genuinely changes."""
    import struct as _struct
    from unittest.mock import MagicMock

    from cuda_link.shm_protocol import DtypeCodec, Metadata, SHMLayout

    old_shape = (4, 4, 4)
    new_shape = (8, 8, 4)
    dtype = "float32"

    mock_numpy = MagicMock()
    mock_numpy.needs_rebuild.return_value = False
    imp = _make_connected_importer(shape=old_shape, dtype=dtype, num_slots=1, write_idx=1, numpy=mock_numpy)

    # Write SHM metadata with a DIFFERENT shape (8×8 instead of 4×4)
    shm_buf = imp._conn.shm_handle.buf
    layout = SHMLayout(1)
    kind, bits, flags = DtypeCodec.encode(dtype)
    h, w, c = new_shape
    Metadata(
        width=w,
        height=h,
        num_comps=c,
        format_kind=kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=w * h * c * DtypeCodec.itemsize(dtype),
    ).pack_into(shm_buf, layout)

    cur_ver = _struct.unpack("<Q", bytes(shm_buf[4:12]))[0]
    _struct.pack_into("<Q", shm_buf, 4, cur_ver + 1)

    imp._reinitialize()

    mock_numpy.close.assert_called_once(), "NumpyBuffers MUST be torn down when shape changes"


# ---------------------------------------------------------------------------
# _event_to_int: IPC event pointer resolution helper
# ---------------------------------------------------------------------------


def test_event_to_int_from_c_uint64() -> None:
    """c_uint64 value (the normal CUDAEvent_t type) resolves via .value."""
    import ctypes

    from cuda_link.importer import _event_to_int

    ptr = 0x02EBB2223B60
    assert _event_to_int(ctypes.c_uint64(ptr)) == ptr


def test_event_to_int_from_bytes() -> None:
    """8-byte little-endian bytes (raw handle) round-trip correctly."""
    from cuda_link.importer import _event_to_int

    ptr = 0x02EBB2223B60
    assert _event_to_int(ptr.to_bytes(8, "little")) == ptr


def test_event_to_int_from_int() -> None:
    """Plain int passes through unchanged."""
    from cuda_link.importer import _event_to_int

    ptr = 0x02EBB2223B60
    assert _event_to_int(ptr) == ptr


def test_event_to_int_all_forms_agree() -> None:
    """All three encodings of the same pointer resolve to the same integer."""
    import ctypes

    from cuda_link.importer import _event_to_int

    ptr = 0x02EBB2223B60
    results = [
        _event_to_int(ctypes.c_uint64(ptr)),
        _event_to_int(ptr.to_bytes(8, "little")),
        _event_to_int(ptr),
    ]
    assert results[0] == results[1] == results[2] == ptr
