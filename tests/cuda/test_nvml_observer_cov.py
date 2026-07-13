"""Coverage-focused tests for cuda_link.nvml_observer.

test_nvml_observer.py already covers the pynvml-missing import path,
snapshot()'s populated/partial-failure/throttle-decode cases (for
temperature/power specifically), the env-var gate, ref-count lifecycle
across multiple observers, and Exporter/Importer get_stats() integration.

This file fills the remaining gaps:
  - the ``from Env import env_bool`` td_exporter-flat-namespace fallback
    when ``cuda_link._env`` is unimportable (lines 39-40).
  - _NvmlRefCounter.acquire()/release() early-returns when NVML_AVAILABLE
    is False (lines 71, 79).
  - NVMLObserver.start() short-circuits: disabled (line 147) and
    already-started (line 149).
  - the ``hasattr(pynvml, "NVML_DRIVER_MCDM")`` False branch (162->164).
  - the "acquire() itself raised" branch, where ``acquired`` stays False
    so release() must not be (wrongly) called (171->173).
  - __enter__ / __exit__ context-manager protocol (184-185, 188).
  - snapshot()'s remaining independent try/except blocks: utilization
    rates, memory info, pcie throughput, and throttle reasons each
    failing on their own (207-208, 214-215, 226-227, 241-242).
  - snapshot() including "driver_model" in its output when set (245).
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fakes (duplicated locally per this test suite's established pattern)
# ---------------------------------------------------------------------------


class _NVMLError(Exception):
    """Concrete exception class to stand in for pynvml.NVMLError in tests."""


def _make_mock_pynvml() -> MagicMock:
    """Build a mock pynvml module with realistic return values."""
    mock = MagicMock()
    mock.NVMLError = _NVMLError

    util = MagicMock()
    util.gpu = 42
    util.memory = 18
    mock.nvmlDeviceGetUtilizationRates.return_value = util

    mem = MagicMock()
    mem.used = 512 * 1024 * 1024
    mem.total = 8192 * 1024 * 1024
    mock.nvmlDeviceGetMemoryInfo.return_value = mem

    mock.nvmlDeviceGetClockInfo.side_effect = lambda h, clock_type: 1800 if clock_type == mock.NVML_CLOCK_SM else 7000
    mock.nvmlDeviceGetPcieThroughput.side_effect = lambda h, direction: (
        50000 if direction == mock.NVML_PCIE_UTIL_TX_BYTES else 80000
    )
    mock.nvmlDeviceGetTemperature.return_value = 65
    mock.nvmlDeviceGetPowerUsage.return_value = 150_000
    mock.nvmlDeviceGetEnforcedPowerLimit.return_value = 250_000
    mock.nvmlDeviceGetCurrentClocksThrottleReasons.return_value = 0

    return mock


def _make_started_observer(mock_pynvml: MagicMock):
    """A NVMLObserver already marked started/handled, without going through start()."""
    from cuda_link.nvml_observer import NVMLObserver

    obs = NVMLObserver(device=0, enabled=True)
    obs._started = True
    obs._handle = mock_pynvml.nvmlDeviceGetHandleByIndex(0)
    return obs


# ---------------------------------------------------------------------------
# env_bool import fallback (lines 39-40)
# ---------------------------------------------------------------------------


def test_env_bool_falls_back_to_flat_env_module_when_cuda_link_env_missing():
    """Reload with cuda_link._env forced unimportable -> falls back to `from Env import env_bool`."""
    import cuda_link.nvml_observer as mod

    original = sys.modules.get("cuda_link._env", "SENTINEL")
    sys.modules["cuda_link._env"] = None  # type: ignore[assignment]

    try:
        importlib.reload(mod)
        # The flat td_exporter namespace module must be importable and used.
        import Env

        assert mod.env_bool is Env.env_bool
    finally:
        if original == "SENTINEL":
            sys.modules.pop("cuda_link._env", None)
        else:
            sys.modules["cuda_link._env"] = original
        importlib.reload(mod)


# ---------------------------------------------------------------------------
# _NvmlRefCounter early-returns when NVML_AVAILABLE is False (lines 71, 79)
# ---------------------------------------------------------------------------


def test_ref_counter_acquire_is_noop_when_nvml_unavailable():
    import cuda_link.nvml_observer as mod

    counter = mod._NvmlRefCounter()
    with patch("cuda_link.nvml_observer.NVML_AVAILABLE", False):
        counter.acquire()  # must not touch pynvml at all
    assert counter._count == 0


def test_ref_counter_release_is_noop_when_nvml_unavailable():
    import cuda_link.nvml_observer as mod

    counter = mod._NvmlRefCounter()
    counter._count = 5  # pretend something had been acquired earlier
    with patch("cuda_link.nvml_observer.NVML_AVAILABLE", False):
        counter.release()  # must not touch pynvml, must not touch _count
    assert counter._count == 5


# ---------------------------------------------------------------------------
# start() short-circuits (lines 147, 149)
# ---------------------------------------------------------------------------


def test_start_returns_false_when_disabled():
    from cuda_link.nvml_observer import NVMLObserver

    obs = NVMLObserver(device=0, enabled=False)
    assert obs.start() is False
    assert obs._started is False


def test_start_returns_true_when_already_started():
    from cuda_link.nvml_observer import NVMLObserver

    mock_pynvml = _make_mock_pynvml()
    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml), patch("cuda_link.nvml_observer.NVML_AVAILABLE", True):
        obs = NVMLObserver(device=0, enabled=True)
        obs._started = True
        assert obs.start() is True
        # No pynvml calls should have been made — the early return short-circuits.
        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_not_called()


# ---------------------------------------------------------------------------
# driver-model detection — hasattr(pynvml, "NVML_DRIVER_MCDM") False branch
# (branch 162->164)
# ---------------------------------------------------------------------------


def test_driver_model_detection_without_mcdm_attribute():
    import cuda_link.nvml_observer as nvml_mod
    from cuda_link.nvml_observer import NVMLObserver

    mock_pynvml = _make_mock_pynvml()
    mock_pynvml.NVML_DRIVER_WDDM = 1
    mock_pynvml.NVML_DRIVER_WDM = 0
    mock_pynvml.nvmlDeviceGetCurrentDriverModel.return_value = 1
    # A bare MagicMock auto-vivifies ANY attribute access (so hasattr() would
    # always be True by default) — explicitly delete it to exercise the
    # False direction of `hasattr(pynvml, "NVML_DRIVER_MCDM")`.
    del mock_pynvml.NVML_DRIVER_MCDM
    assert not hasattr(mock_pynvml, "NVML_DRIVER_MCDM")

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml), patch("cuda_link.nvml_observer.NVML_AVAILABLE", True):
        nvml_mod._NVML_REFS._count = 0
        obs = NVMLObserver(device=0, enabled=True)
        try:
            assert obs.start() is True
            assert obs._driver_model == "WDDM"
        finally:
            obs.stop()
            nvml_mod._NVML_REFS._count = 0


# ---------------------------------------------------------------------------
# start() failure before acquire() completes — `acquired` stays False, so
# release() must not be called (branch 171->173).
# ---------------------------------------------------------------------------


def test_start_failure_inside_acquire_does_not_call_release():
    import cuda_link.nvml_observer as nvml_mod
    from cuda_link.nvml_observer import NVMLObserver

    mock_pynvml = _make_mock_pynvml()
    mock_pynvml.nvmlInit.side_effect = _NVMLError("driver not loaded")

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml), patch("cuda_link.nvml_observer.NVML_AVAILABLE", True):
        nvml_mod._NVML_REFS._count = 0
        obs = NVMLObserver(device=0, enabled=True)
        assert obs.start() is False
        assert obs._started is False

        # acquire() raised before incrementing _count, so release() must
        # never have been invoked — nvmlShutdown must not have been called
        # and _count must remain untouched at 0.
        assert nvml_mod._NVML_REFS._count == 0
        mock_pynvml.nvmlShutdown.assert_not_called()


# ---------------------------------------------------------------------------
# __enter__ / __exit__ (lines 184-185, 188)
# ---------------------------------------------------------------------------


def test_context_manager_calls_start_and_stop():
    """__enter__ calls self.start() and __exit__ calls self.stop().

    enabled=False would make both start() and stop() no-ops regardless of whether
    __enter__/__exit__ actually invoke them — asserting only on `obs._started`
    (as a prior version of this test did) can't distinguish "the methods were
    called" from "the methods were never called at all". Patch the instance's
    start/stop with MagicMocks instead, so the call itself is what's pinned.
    """
    from cuda_link.nvml_observer import NVMLObserver

    obs = NVMLObserver(device=0, enabled=False)
    obs.start = MagicMock(return_value=True)
    obs.stop = MagicMock()

    with obs as ctx:
        assert ctx is obs
        obs.start.assert_called_once()
        obs.stop.assert_not_called()

    obs.start.assert_called_once()
    obs.stop.assert_called_once()


# ---------------------------------------------------------------------------
# snapshot() — remaining independent try/except blocks
# (lines 207-208, 214-215, 226-227, 241-242)
# ---------------------------------------------------------------------------


def test_snapshot_missing_utilization_rates_on_error():
    mock_pynvml = _make_mock_pynvml()
    mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = _NVMLError("no rates")
    obs = _make_started_observer(mock_pynvml)

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert "gpu_util_pct" not in result
    assert "mem_bw_util_pct" not in result
    # Other independent metric blocks still populate.
    assert result["mem_used_mb"] == pytest.approx(512.0)


def test_snapshot_missing_memory_info_on_error():
    mock_pynvml = _make_mock_pynvml()
    mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = _NVMLError("no mem info")
    obs = _make_started_observer(mock_pynvml)

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert "mem_used_mb" not in result
    assert "mem_total_mb" not in result
    assert result["gpu_util_pct"] == 42


def test_snapshot_missing_pcie_throughput_on_error():
    mock_pynvml = _make_mock_pynvml()
    mock_pynvml.nvmlDeviceGetPcieThroughput.side_effect = _NVMLError("no pcie")
    obs = _make_started_observer(mock_pynvml)

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert "pcie_tx_kbps" not in result
    assert "pcie_rx_kbps" not in result
    assert result["gpu_util_pct"] == 42


def test_snapshot_missing_throttle_reasons_on_error():
    mock_pynvml = _make_mock_pynvml()
    mock_pynvml.nvmlDeviceGetCurrentClocksThrottleReasons.side_effect = _NVMLError("no throttle info")
    obs = _make_started_observer(mock_pynvml)

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert "throttle_reasons" not in result
    assert result["gpu_util_pct"] == 42


# ---------------------------------------------------------------------------
# snapshot() includes driver_model when set (line 245)
# ---------------------------------------------------------------------------


def test_snapshot_includes_driver_model_when_set():
    mock_pynvml = _make_mock_pynvml()
    obs = _make_started_observer(mock_pynvml)
    obs._driver_model = "WDDM"

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert result["driver_model"] == "WDDM"


def test_snapshot_omits_driver_model_when_unset():
    mock_pynvml = _make_mock_pynvml()
    obs = _make_started_observer(mock_pynvml)
    assert obs._driver_model is None

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert "driver_model" not in result
