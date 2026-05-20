"""
Tests for N1: spin-then-sleep busy-wait in Importer._wait_for_slot.

All tests are pure unit tests (no CUDA required) — they mock query_event.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_importer(spin_us: int = 200, timeout_ms: float = 5000.0) -> object:
    """Build an Importer with mocked IPCConnection state (no real CUDA/SHM required)."""
    from ctypes import c_void_p

    from cuda_link._cuda_adapters import FakeCudaAdapter
    from cuda_link._importer_port import ImportPolicy, ImportSpec
    from cuda_link.importer import Importer, IPCConnection
    from cuda_link.shm_protocol import SHMLayout

    spec = ImportSpec(shm_name="mock", device=0, timeout_ms=timeout_ms)
    policy = ImportPolicy(wait_spin_us=spin_us)

    mock_cuda = MagicMock()
    layout = SHMLayout(2)
    conn = IPCConnection(
        cuda=mock_cuda,
        shm_handle=None,
        ipc_version=1,
        num_slots=2,
        ipc_handles=[None, None],
        dev_ptrs=[c_void_p(0x1000), c_void_p(0x2000)],
        ipc_events=[MagicMock(), MagicMock()],  # both non-None → GPU event path
        layout=layout,
        shutdown_offset=layout.shutdown_offset,
        timestamp_offset=layout.timestamp_offset,
    )

    imp = Importer(spec, policy, FakeCudaAdapter())
    imp._conn = conn
    imp._initialized = True
    return imp


# ---------------------------------------------------------------------------
# Phase 1: spin resolves pre-signaled events (no sleep)
# ---------------------------------------------------------------------------


def test_spin_resolves_immediately_no_sleep() -> None:
    """When query_event returns True on first try, no time.sleep call is made."""
    imp = _make_importer(spin_us=200)
    imp._conn.cuda.query_event.return_value = True

    with patch("cuda_link.importer.time") as mock_time:
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
    imp._conn.cuda.query_event.side_effect = [False, True]

    with patch("cuda_link.importer.time") as mock_time:
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
    call_count = [0]

    def query_side_effect(event):
        call_count[0] += 1
        return call_count[0] > 5  # eventually returns True

    imp._conn.cuda.query_event.side_effect = query_side_effect

    times = iter(
        [0.0]  # wait_start
        + [0.0003] * 20  # all Phase 1 checks: already past spin_deadline
        + [0.0004] * 20  # Phase 2 checks
    )
    with patch("cuda_link.importer.time") as mock_time:
        mock_time.perf_counter.side_effect = lambda: next(times)
        mock_time.sleep = MagicMock()
        imp._wait_for_slot(slot=0)

    assert mock_time.sleep.call_count >= 1  # at least one sleep in Phase 2
    assert imp.wait_sleep_hits == 1
    assert imp.wait_spin_hits == 0


# ---------------------------------------------------------------------------
# Disabled spin (wait_spin_us=0)
# ---------------------------------------------------------------------------


def test_spin_us_zero_disables_phase_one() -> None:
    """wait_spin_us=0: Phase 1 is skipped entirely, goes straight to sleep."""
    imp = _make_importer(spin_us=0, timeout_ms=5000.0)

    call_count = [0]

    def query_side_effect(event):
        call_count[0] += 1
        return call_count[0] > 2

    imp._conn.cuda.query_event.side_effect = query_side_effect

    times = iter([0.0] + [0.0001] * 20 + [0.0002] * 20)
    with patch("cuda_link.importer.time") as mock_time:
        mock_time.perf_counter.side_effect = lambda: next(times)
        mock_time.sleep = MagicMock()
        imp._wait_for_slot(slot=0)

    assert imp.wait_spin_hits == 0
    assert imp.wait_sleep_hits == 1


# ---------------------------------------------------------------------------
# Timeout still raised
# ---------------------------------------------------------------------------


def test_timeout_still_raised() -> None:
    """TimeoutError is raised after timeout_ms regardless of spin configuration."""
    imp = _make_importer(spin_us=200, timeout_ms=1.0)  # 1ms timeout
    imp._conn.cuda.query_event.return_value = False  # never ready

    # wait_start=0.0; all subsequent calls return 2.0, which is:
    # - past spin_deadline (0.0002) → Phase 1 exits immediately
    # - past deadline (0.001)       → Phase 2 raises TimeoutError on first check
    with patch("cuda_link.importer.time") as mock_time:
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
# Env var (ImportPolicy.from_env)
# ---------------------------------------------------------------------------


def test_spin_us_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_WAIT_SPIN_US env var is read by ImportPolicy.from_env()."""
    monkeypatch.setenv("CUDALINK_WAIT_SPIN_US", "500")

    from cuda_link._importer_port import ImportPolicy

    policy = ImportPolicy.from_env()
    assert policy.wait_spin_us == 500


def test_spin_us_env_var_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_WAIT_SPIN_US=0 disables spin."""
    monkeypatch.setenv("CUDALINK_WAIT_SPIN_US", "0")

    from cuda_link._importer_port import ImportPolicy

    policy = ImportPolicy.from_env()
    assert policy.wait_spin_us == 0
