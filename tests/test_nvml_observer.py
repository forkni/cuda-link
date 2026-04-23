"""
Tests for NVMLObserver — pure unit tests (no CUDA / pynvml required).

All pynvml calls are mocked so these run on any machine.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# NVML unavailable path
# ---------------------------------------------------------------------------


def test_nvml_available_flag_false_when_pynvml_missing() -> None:
    """NVML_AVAILABLE is False when pynvml is not installed."""
    import importlib
    import sys

    # Simulate pynvml not installed
    original = sys.modules.get("pynvml", "SENTINEL")
    sys.modules["pynvml"] = None  # type: ignore[assignment]

    try:
        # Reload the module to re-evaluate the try/except
        import cuda_link.nvml_observer as mod

        importlib.reload(mod)
        # After reload with pynvml=None, NVML_AVAILABLE should be False
        assert mod.NVML_AVAILABLE is False
    finally:
        if original == "SENTINEL":
            sys.modules.pop("pynvml", None)
        else:
            sys.modules["pynvml"] = original
        import cuda_link.nvml_observer as mod2

        importlib.reload(mod2)


def test_snapshot_returns_unavailable_when_not_started() -> None:
    """snapshot() returns {'nvml_available': False} when observer not started."""
    from cuda_link.nvml_observer import NVMLObserver

    obs = NVMLObserver(device=0, enabled=True)
    # Don't call start()
    result = obs.snapshot()
    assert result == {"nvml_available": False}


def test_snapshot_returns_unavailable_when_disabled() -> None:
    """snapshot() returns {'nvml_available': False} when enabled=False."""
    from cuda_link.nvml_observer import NVMLObserver

    obs = NVMLObserver(device=0, enabled=False)
    result = obs.snapshot()
    assert result == {"nvml_available": False}


# ---------------------------------------------------------------------------
# Mocked pynvml path
# ---------------------------------------------------------------------------


class _NVMLError(Exception):
    """Concrete exception class to stand in for pynvml.NVMLError in tests."""


def _make_mock_pynvml() -> MagicMock:
    """Build a mock pynvml module with realistic return values."""
    mock = MagicMock()
    mock.NVMLError = _NVMLError  # real exception class so except clauses work

    util = MagicMock()
    util.gpu = 42
    util.memory = 18
    mock.nvmlDeviceGetUtilizationRates.return_value = util

    mem = MagicMock()
    mem.used = 512 * 1024 * 1024  # 512 MiB
    mem.total = 8192 * 1024 * 1024  # 8 GiB
    mock.nvmlDeviceGetMemoryInfo.return_value = mem

    mock.nvmlDeviceGetClockInfo.side_effect = lambda h, clock_type: (1800 if clock_type == mock.NVML_CLOCK_SM else 7000)
    mock.nvmlDeviceGetPcieThroughput.side_effect = lambda h, direction: (
        50000 if direction == mock.NVML_PCIE_UTIL_TX_BYTES else 80000
    )
    mock.nvmlDeviceGetTemperature.return_value = 65
    mock.nvmlDeviceGetPowerUsage.return_value = 150_000  # milliwatts
    mock.nvmlDeviceGetEnforcedPowerLimit.return_value = 250_000  # milliwatts
    mock.nvmlDeviceGetCurrentClocksThrottleReasons.return_value = 0  # no throttle

    return mock


def test_snapshot_returns_populated_dict() -> None:
    """snapshot() returns all expected keys when pynvml is available."""
    from cuda_link.nvml_observer import NVMLObserver

    mock_pynvml = _make_mock_pynvml()

    obs = NVMLObserver(device=0, enabled=True)
    obs._handle = MagicMock()
    obs._started = True

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert result["nvml_available"] is True
    assert result["gpu_util_pct"] == 42
    assert result["mem_bw_util_pct"] == 18
    assert result["mem_used_mb"] == pytest.approx(512.0)
    assert result["mem_total_mb"] == pytest.approx(8192.0)
    assert "sm_clock_mhz" in result
    assert "mem_clock_mhz" in result
    assert "pcie_tx_kbps" in result
    assert "pcie_rx_kbps" in result
    assert result["temp_c"] == 65
    assert result["power_w"] == pytest.approx(150.0)
    assert result["power_limit_w"] == pytest.approx(250.0)
    assert result["throttle_reasons"] == []


def test_snapshot_partial_on_nvml_error() -> None:
    """snapshot() skips failing metrics and returns the rest — no exception raised."""
    from cuda_link.nvml_observer import NVMLObserver

    mock_pynvml = _make_mock_pynvml()
    mock_pynvml.nvmlDeviceGetTemperature.side_effect = _NVMLError("permission denied")
    mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = _NVMLError("not supported")

    obs = NVMLObserver(device=0, enabled=True)
    obs._handle = MagicMock()
    obs._started = True

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert result["nvml_available"] is True
    assert result["gpu_util_pct"] == 42  # still returned
    assert "temp_c" not in result  # skipped
    assert "power_w" not in result  # skipped


def test_throttle_reasons_decoded() -> None:
    """throttle_reasons decodes bitmask to list of human-readable strings."""
    from cuda_link.nvml_observer import NVMLObserver

    mock_pynvml = _make_mock_pynvml()
    # hw_slowdown (0x08) | sw_power_cap (0x04)
    mock_pynvml.nvmlDeviceGetCurrentClocksThrottleReasons.return_value = 0x04 | 0x08
    mock_pynvml.NVMLError = _NVMLError

    obs = NVMLObserver(device=0, enabled=True)
    obs._handle = MagicMock()
    obs._started = True

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml):
        result = obs.snapshot()

    assert "sw_power_cap" in result["throttle_reasons"]
    assert "hw_slowdown" in result["throttle_reasons"]
    assert "gpu_idle" not in result["throttle_reasons"]


# ---------------------------------------------------------------------------
# Env var gate
# ---------------------------------------------------------------------------


def test_env_var_enables_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_NVML=1 sets enabled=True when enabled=None (default)."""
    monkeypatch.setenv("CUDALINK_NVML", "1")
    from cuda_link.nvml_observer import NVMLObserver

    obs = NVMLObserver(device=0)  # enabled=None → read env
    assert obs.enabled is True


def test_env_var_default_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_NVML defaults to 0 (disabled)."""
    monkeypatch.delenv("CUDALINK_NVML", raising=False)
    from cuda_link.nvml_observer import NVMLObserver

    obs = NVMLObserver(device=0)
    assert obs.enabled is False


# ---------------------------------------------------------------------------
# get_stats() integration
# ---------------------------------------------------------------------------


def test_exporter_get_stats_no_observer() -> None:
    """get_stats() does not include 'nvml' key when no observer attached."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter

    exp = CUDAIPCExporter(shm_name="test", height=8, width=8)
    stats = exp.get_stats()
    assert "nvml" not in stats


def test_exporter_get_stats_with_observer() -> None:
    """get_stats() includes 'nvml' sub-dict when NVMLObserver attached."""
    from cuda_link.cuda_ipc_exporter import CUDAIPCExporter
    from cuda_link.nvml_observer import NVMLObserver

    obs = MagicMock(spec=NVMLObserver)
    obs.snapshot.return_value = {"nvml_available": True, "gpu_util_pct": 77}

    exp = CUDAIPCExporter(shm_name="test", height=8, width=8)
    exp.attach_nvml_observer(obs)

    stats = exp.get_stats()
    assert "nvml" in stats
    assert stats["nvml"]["gpu_util_pct"] == 77


def test_importer_get_stats_with_observer() -> None:
    """CUDAIPCImporter.get_stats() includes 'nvml' sub-dict when observer attached."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter
    from cuda_link.nvml_observer import NVMLObserver

    obs = MagicMock(spec=NVMLObserver)
    obs.snapshot.return_value = {"nvml_available": True, "temp_c": 72}

    with patch.object(CUDAIPCImporter, "_initialize", return_value=None):
        imp = CUDAIPCImporter(shm_name="test")
        # Manually set required attrs not set by _initialize (which was no-op)
        imp.shape = (8, 8, 4)
        imp.dtype = "uint8"
        imp.num_slots = 0
        imp.frame_count = 0
        imp.dev_ptrs = []
        imp.tensors = []
        imp.wait_spin_hits = 0
        imp.wait_sleep_hits = 0
        imp.total_wait_spin_us = 0.0
        imp.total_wait_sleep_us = 0.0

    imp.attach_nvml_observer(obs)
    stats = imp.get_stats()
    assert "nvml" in stats
    assert stats["nvml"]["temp_c"] == 72


# ---------------------------------------------------------------------------
# Ref-count lifecycle
# ---------------------------------------------------------------------------


def test_ref_count_multiple_observers() -> None:
    """Multiple NVMLObserver.start() calls do not double-init."""
    mock_pynvml = _make_mock_pynvml()

    with patch("cuda_link.nvml_observer.pynvml", mock_pynvml), patch("cuda_link.nvml_observer.NVML_AVAILABLE", True):
        import cuda_link.nvml_observer as nvml_mod
        from cuda_link.nvml_observer import NVMLObserver

        # Reset global ref count
        nvml_mod._nvml_ref_count = 0

        obs1 = NVMLObserver(device=0, enabled=True)
        obs2 = NVMLObserver(device=0, enabled=True)
        obs1.start()
        obs2.start()

        assert mock_pynvml.nvmlInit.call_count == 1  # only once

        obs1.stop()
        assert mock_pynvml.nvmlShutdown.call_count == 0  # still one ref

        obs2.stop()
        assert mock_pynvml.nvmlShutdown.call_count == 1  # final ref released
