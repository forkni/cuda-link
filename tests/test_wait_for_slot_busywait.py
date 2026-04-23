"""
Tests for N1: spin-then-sleep busy-wait in CUDAIPCImporter._wait_for_slot.

All tests are pure unit tests (no CUDA required) — they mock query_event.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_importer(spin_us: int = 200, timeout_ms: float = 5000.0) -> object:
    """Build a CUDAIPCImporter with mocked state (no real CUDA/SHM required)."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    imp = object.__new__(CUDAIPCImporter)
    imp.device = 0
    imp._spin_us = spin_us
    imp.timeout_ms = timeout_ms
    imp._initialized = True
    imp.debug = False
    imp.shm_name = "mock"
    imp.shape = (8, 8, 4)
    imp.dtype = "uint8"
    imp.num_slots = 2
    imp.frame_count = 0
    imp._last_write_idx = 0

    # N1 counters
    imp.total_wait_spin_us = 0.0
    imp.total_wait_sleep_us = 0.0
    imp.wait_spin_hits = 0
    imp.wait_sleep_hits = 0

    # Other perf counters (not N1 but accessed by get_stats / log)
    imp.total_wait_event_time = 0.0
    imp.total_get_frame_time = 0.0
    imp.total_shm_read_us = 0.0
    imp.last_latency = 0.0

    from ctypes import c_void_p

    imp.cuda = MagicMock()
    imp.ipc_events = [MagicMock(), MagicMock()]  # both non-None → GPU path
    imp.dev_ptrs = [c_void_p(0x1000), c_void_p(0x2000)]
    imp.tensors = [None, None]
    imp._wrappers = []
    imp.cupy_arrays = []
    imp._numpy_buffer = None
    imp._pinned_ptr = None
    imp._shutdown_offset = 0
    imp._timestamp_offset = 0
    imp._numpy_stream = MagicMock()
    imp.shm_handle = None

    return imp


# ---------------------------------------------------------------------------
# Phase 1: spin resolves pre-signaled events (no sleep)
# ---------------------------------------------------------------------------


def test_spin_resolves_immediately_no_sleep() -> None:
    """When query_event returns True on first try, no time.sleep call is made."""
    imp = _make_importer(spin_us=200)
    imp.cuda.query_event.return_value = True

    with patch("cuda_link.cuda_ipc_importer.time") as mock_time:
        # Calls: wait_start(1), while-check(2), spin_us-calc(3)
        mock_time.perf_counter.side_effect = [0.0, 0.00001, 0.00001]
        result = imp._wait_for_slot(slot=0)

    mock_time.sleep.assert_not_called()
    assert result >= 0.0
    assert imp.wait_spin_hits == 1
    assert imp.wait_sleep_hits == 0


def test_spin_resolves_on_second_poll_no_sleep() -> None:
    """query_event returns False, True on consecutive calls — still no sleep."""
    imp = _make_importer(spin_us=200)
    imp.cuda.query_event.side_effect = [False, True]

    with patch("cuda_link.cuda_ipc_importer.time") as mock_time:
        # Calls: wait_start(1), while-check(2), timeout-check(3),
        #        while-check(4), spin_us-calc(5)
        mock_time.perf_counter.side_effect = [0.0, 0.00005, 0.00005, 0.00010, 0.00010]
        imp._wait_for_slot(slot=0)

    mock_time.sleep.assert_not_called()
    assert imp.wait_spin_hits == 1
    assert imp.wait_sleep_hits == 0


# ---------------------------------------------------------------------------
# Phase 2: sleep poll used when spin budget expires
# ---------------------------------------------------------------------------


def test_falls_through_to_sleep_after_spin_budget() -> None:
    """Events not ready within spin budget — Phase 2 sleep loop is entered."""
    imp = _make_importer(spin_us=200, timeout_ms=5000.0)
    # query_event always False except at the very end
    call_count = [0]

    def query_side_effect(event):
        call_count[0] += 1
        return call_count[0] > 5  # eventually returns True

    imp.cuda.query_event.side_effect = query_side_effect

    # perf_counter: start=0, then rapidly exceed spin_deadline (0.0002s),
    # then timeout check in phase 2 passes, then True returned
    times = iter(
        [0.0]  # wait_start
        + [0.0003] * 20  # all Phase 1 checks: already past spin_deadline
        + [0.0004] * 20  # Phase 2 checks
    )
    with patch("cuda_link.cuda_ipc_importer.time") as mock_time:
        mock_time.perf_counter.side_effect = lambda: next(times)
        mock_time.sleep = MagicMock()
        imp._wait_for_slot(slot=0)

    assert mock_time.sleep.call_count >= 1  # at least one sleep in Phase 2
    assert imp.wait_sleep_hits == 1
    assert imp.wait_spin_hits == 0


# ---------------------------------------------------------------------------
# Disabled spin (CUDALINK_WAIT_SPIN_US=0)
# ---------------------------------------------------------------------------


def test_spin_us_zero_disables_phase_one() -> None:
    """CUDALINK_WAIT_SPIN_US=0: Phase 1 is skipped entirely, goes straight to sleep."""
    imp = _make_importer(spin_us=0, timeout_ms=5000.0)

    call_count = [0]

    def query_side_effect(event):
        call_count[0] += 1
        return call_count[0] > 2

    imp.cuda.query_event.side_effect = query_side_effect

    times = iter([0.0] + [0.0001] * 20 + [0.0002] * 20)
    with patch("cuda_link.cuda_ipc_importer.time") as mock_time:
        mock_time.perf_counter.side_effect = lambda: next(times)
        mock_time.sleep = MagicMock()
        imp._wait_for_slot(slot=0)

    # With spin_us=0, loop never enters Phase 1, immediately goes to sleep phase
    assert imp.wait_spin_hits == 0
    assert imp.wait_sleep_hits == 1


# ---------------------------------------------------------------------------
# Timeout still raised
# ---------------------------------------------------------------------------


def test_timeout_still_raised() -> None:
    """TimeoutError is raised after timeout_ms regardless of spin configuration."""
    imp = _make_importer(spin_us=200, timeout_ms=1.0)  # 1ms timeout
    imp.cuda.query_event.return_value = False  # never ready

    # wait_start=0.0; all subsequent calls return 2.0, which is:
    # - past spin_deadline (0.0002) → Phase 1 exits immediately
    # - past deadline (0.001)       → Phase 2 raises TimeoutError on first check
    with patch("cuda_link.cuda_ipc_importer.time") as mock_time:
        mock_time.perf_counter.side_effect = [0.0] + [2.0] * 20
        mock_time.sleep = MagicMock()
        with pytest.raises(TimeoutError, match="timed out"):
            imp._wait_for_slot(slot=0)


# ---------------------------------------------------------------------------
# get_stats includes spin counters
# ---------------------------------------------------------------------------


def test_get_stats_includes_spin_counters() -> None:
    """get_stats() returns wait_spin_hits, wait_sleep_hits, avg_spin_us, avg_sleep_us."""
    imp = _make_importer(spin_us=200)
    imp.wait_spin_hits = 50
    imp.wait_sleep_hits = 10
    imp.total_wait_spin_us = 5000.0
    imp.total_wait_sleep_us = 2000.0

    stats = imp.get_stats()

    assert stats["wait_spin_hits"] == 50
    assert stats["wait_sleep_hits"] == 10
    assert stats["avg_spin_us"] == pytest.approx(100.0)
    assert stats["avg_sleep_us"] == pytest.approx(200.0)


def test_get_stats_zero_hits_no_division_error() -> None:
    """get_stats() does not raise when hit counters are 0."""
    imp = _make_importer(spin_us=200)

    stats = imp.get_stats()
    assert stats["avg_spin_us"] == 0.0
    assert stats["avg_sleep_us"] == 0.0


# ---------------------------------------------------------------------------
# Env var
# ---------------------------------------------------------------------------


def test_spin_us_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_WAIT_SPIN_US env var is read at __init__ time."""

    monkeypatch.setenv("CUDALINK_WAIT_SPIN_US", "500")

    # Need a real __init__ call — build a minimal mock that satisfies _initialize()
    # by patching _initialize to be a no-op
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    with patch.object(CUDAIPCImporter, "_initialize", return_value=None):
        imp = CUDAIPCImporter(shm_name="noop")

    assert imp._spin_us == 500


def test_spin_us_env_var_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_WAIT_SPIN_US=0 disables spin."""
    monkeypatch.setenv("CUDALINK_WAIT_SPIN_US", "0")

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    with patch.object(CUDAIPCImporter, "_initialize", return_value=None):
        imp = CUDAIPCImporter(shm_name="noop")

    assert imp._spin_us == 0
