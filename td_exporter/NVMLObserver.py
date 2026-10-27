# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
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

Note: this module imports ``pynvml``, a module name shared by both the deprecated
``pynvml`` PyPI package and the maintained ``nvidia-ml-py`` package. cuda-link's
``[nvml]`` extra declares ``nvidia-ml-py>=12.535``. If the deprecated ``pynvml``
package is also installed (often as a transitive dep of torch), it may win the
import race and emit a ``FutureWarning``. That specific warning is suppressed at
the import site below so it doesn't pollute logs; users who prefer to remove the
duplicate install can run ``pip uninstall pynvml && pip install nvidia-ml-py``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import warnings

from Env import env_bool

logger = logging.getLogger(__name__)

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The pynvml package is deprecated.*",
            category=FutureWarning,
        )
        import pynvml

    NVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    NVML_AVAILABLE = False


class _NvmlRefCounter:
    """Process-global ref-count for nvmlInit/nvmlShutdown.

    Tolerates multiple NVMLObserver instances without double-init/shutdown errors.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._lock: threading.Lock = threading.Lock()

    def acquire(self) -> None:
        if not NVML_AVAILABLE:
            return
        with self._lock:
            if self._count == 0:
                pynvml.nvmlInit()
            self._count += 1

    def release(self) -> None:
        if not NVML_AVAILABLE:
            return
        with self._lock:
            self._count = max(0, self._count - 1)
            if self._count == 0:
                pynvml.nvmlShutdown()


_NVML_REFS = _NvmlRefCounter()


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
        gpu_util_pct, mem_bw_util_pct  (from nvmlDeviceGetUtilizationRates). gpu_util_pct is
                                        device activity: the % of the sample period in which at
                                        least one kernel was running, not SM occupancy or
                                        throughput. A GPU running one tiny kernel back to back
                                        reads 100%.
        proc_sm_util_pct, proc_mem_util_pct,
        other_sm_util_pct, other_mem_util_pct
                                        (from nvmlDeviceGetProcessesUtilizationInfo, or the
                                        legacy nvmlDeviceGetProcessUtilization). proc_* is this
                                        process; other_* is the sum over every other process.
                                        Absent when there is no usable sample (unsupported
                                        driver, nothing sampled yet, or an NVML error).
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
            self.enabled = env_bool("CUDALINK_NVML", default=False)
        else:
            self.enabled = enabled
        self._handle = None
        self._started = False
        self._driver_model: str | None = None
        self._last_proc_ts: int = 0
        self._last_proc_util: dict | None = None

    def start(self) -> bool:
        """Initialize NVML and open device handle.

        Returns:
            True if NVML is available and handle opened, False otherwise.
        """
        if not NVML_AVAILABLE or not self.enabled:
            return False
        if self._started:
            return True
        acquired = False
        try:
            _NVML_REFS.acquire()
            acquired = True
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
        except (pynvml.NVMLError, RuntimeError, OSError) as e:
            # Only release a ref we actually acquired — if nvmlInit() itself (inside
            # acquire()) was what failed, release() would wrongly nvmlShutdown() an
            # NVML that was never successfully initialized by this call.
            if acquired:
                _NVML_REFS.release()
            logger.warning("NVML start failed for device %d: %s", self.device, e)
            return False

    def stop(self) -> None:
        """Release NVML handle and decrement global ref-count."""
        if self._started:
            _NVML_REFS.release()
            self._handle = None
            self._started = False
            self._last_proc_ts = 0
            self._last_proc_util = None

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

        out.update(self._process_utilization(h))

        if self._driver_model is not None:
            out["driver_model"] = self._driver_model

        return out

    def _process_utilization(self, h: object) -> dict:
        """Per-process SM/memory utilization: this process vs the sum of all others.

        NVML returns only samples newer than the last timestamp seen, so a poll faster than
        the driver's sample period raises NotFound; the last good values are re-emitted then
        (they keep the keys from flickering). Any other NVML error yields no keys.
        """
        fn = getattr(pynvml, "nvmlDeviceGetProcessesUtilizationInfo", None) or getattr(
            pynvml, "nvmlDeviceGetProcessUtilization", None
        )
        if fn is None:
            return {}
        try:
            samples = fn(h, self._last_proc_ts)
        except pynvml.NVMLError_NotFound:
            return dict(self._last_proc_util) if self._last_proc_util else {}
        except pynvml.NVMLError:
            return {}
        if not isinstance(samples, (list, tuple)):
            return {}

        newest: dict[int, object] = {}
        for s in samples:
            prev = newest.get(s.pid)
            if prev is None or s.timeStamp > prev.timeStamp:
                newest[s.pid] = s
            self._last_proc_ts = max(self._last_proc_ts, s.timeStamp)

        mine = newest.pop(os.getpid(), None)
        util = {
            "proc_sm_util_pct": mine.smUtil if mine is not None else 0,
            "proc_mem_util_pct": mine.memUtil if mine is not None else 0,
            "other_sm_util_pct": sum(s.smUtil for s in newest.values()),
            "other_mem_util_pct": sum(s.memUtil for s in newest.values()),
        }
        self._last_proc_util = util
        return dict(util)
