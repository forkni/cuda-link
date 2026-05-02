"""
NVML Observability Hook for CUDA Link.

Optional GPU telemetry via NVIDIA Management Library (pynvml).
Follows the existing optional-dep pattern: import fails gracefully,
NVML_AVAILABLE flag gates all usage.

Usage:
    from cuda_link import NVMLObserver, NVML_AVAILABLE

    if NVML_AVAILABLE:
        obs = NVMLObserver(device=0)
        obs.start()
        exporter.attach_nvml_observer(obs)
        stats = exporter.get_stats()  # includes stats["nvml"]
        obs.stop()

Install optional dep:
    pip install "cuda-link[nvml]"    # adds nvidia-ml-py>=12.535
"""

from __future__ import annotations

import contextlib
import os
import threading

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    NVML_AVAILABLE = False

# Process-global ref-count for nvmlInit / nvmlShutdown — tolerates multiple
# NVMLObserver instances in the same process without double-init/shutdown errors.
_nvml_ref_count = 0
_nvml_lock = threading.Lock()


def _nvml_init() -> None:
    global _nvml_ref_count
    with _nvml_lock:
        if _nvml_ref_count == 0:
            pynvml.nvmlInit()
        _nvml_ref_count += 1


def _nvml_shutdown() -> None:
    global _nvml_ref_count
    with _nvml_lock:
        _nvml_ref_count = max(0, _nvml_ref_count - 1)
        if _nvml_ref_count == 0:
            pynvml.nvmlShutdown()


_THROTTLE_NAMES: dict[int, str] = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "sw_thermal_slowdown",
    0x0000000000000040: "hw_thermal_slowdown",
    0x0000000000000080: "hw_power_brake_slowdown",
    0x0000000000000100: "display_clocks_setting",
}


def _decode_throttle(bitmask: int) -> list[str]:
    return [name for bit, name in _THROTTLE_NAMES.items() if bitmask & bit]


class NVMLObserver:
    """Pull-based GPU telemetry via pynvml.

    Call snapshot() (or let get_stats() call it) to sample once.
    No background thread — caller controls cadence.

    Metrics returned by snapshot():
        gpu_util_pct, mem_bw_util_pct  (from nvmlDeviceGetUtilizationRates)
        mem_used_mb, mem_total_mb       (from nvmlDeviceGetMemoryInfo)
        sm_clock_mhz, mem_clock_mhz    (from nvmlDeviceGetClockInfo)
        pcie_tx_kbps, pcie_rx_kbps     (from nvmlDeviceGetPcieThroughput)
        temp_c                          (from nvmlDeviceGetTemperature)
        power_w, power_limit_w          (from nvmlDeviceGetPowerUsage)
        throttle_reasons                (decoded bitmask list)
        driver_model                    "WDDM" / "TCC" / "MCDM" (Windows only; absent on Linux)
    """

    def __init__(self, device: int = 0, enabled: bool | None = None) -> None:
        """Initialize NVML observer.

        Args:
            device: CUDA device index (default 0).
            enabled: If None, reads CUDALINK_NVML env var ("1" = enabled).
                     If False, snapshot() returns {"nvml_available": False} immediately.
        """
        self.device = device
        if enabled is None:
            self.enabled = os.getenv("CUDALINK_NVML", "0") == "1"
        else:
            self.enabled = enabled
        self._handle = None
        self._started = False
        self._driver_model: str | None = None

    def start(self) -> bool:
        """Initialize NVML and open device handle.

        Returns:
            True if NVML is available and handle opened, False otherwise.
        """
        if not NVML_AVAILABLE or not self.enabled:
            return False
        if self._started:
            return True
        try:
            _nvml_init()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            with contextlib.suppress(pynvml.NVMLError):
                # Raises NVMLError_NotSupported on Linux (driver-model is Windows-only).
                _model = pynvml.nvmlDeviceGetCurrentDriverModel(self._handle)
                _names = {
                    pynvml.NVML_DRIVER_WDDM: "WDDM",
                    pynvml.NVML_DRIVER_WDM: "TCC",
                }
                if hasattr(pynvml, "NVML_DRIVER_MCDM"):
                    _names[pynvml.NVML_DRIVER_MCDM] = "MCDM"
                self._driver_model = _names.get(_model, f"unknown({_model})")
            self._started = True
            return True
        except Exception:  # noqa: BLE001
            return False

    def stop(self) -> None:
        """Release NVML handle and decrement global ref-count."""
        if self._started:
            _nvml_shutdown()
            self._handle = None
            self._started = False

    def __enter__(self) -> NVMLObserver:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def snapshot(self) -> dict:
        """Sample all GPU metrics once (non-blocking, ~50-200µs total).

        Returns:
            Dict of metric name → value. If NVML is unavailable or not started,
            returns {"nvml_available": False}.
        """
        if not self._started or self._handle is None:
            return {"nvml_available": False}

        out: dict = {"nvml_available": True}
        h = self._handle

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            out["gpu_util_pct"] = util.gpu
            out["mem_bw_util_pct"] = util.memory
        except pynvml.NVMLError:
            pass

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out["mem_used_mb"] = mem.used / (1024 * 1024)
            out["mem_total_mb"] = mem.total / (1024 * 1024)
        except pynvml.NVMLError:
            pass

        with contextlib.suppress(pynvml.NVMLError):
            out["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)

        with contextlib.suppress(pynvml.NVMLError):
            out["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)

        try:
            out["pcie_tx_kbps"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            out["pcie_rx_kbps"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        except pynvml.NVMLError:
            pass

        with contextlib.suppress(pynvml.NVMLError):
            out["temp_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)

        with contextlib.suppress(pynvml.NVMLError):
            out["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0

        with contextlib.suppress(pynvml.NVMLError):
            out["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0

        try:
            bitmask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            out["throttle_reasons"] = _decode_throttle(bitmask)
        except pynvml.NVMLError:
            pass

        if self._driver_model is not None:
            out["driver_model"] = self._driver_model

        return out
