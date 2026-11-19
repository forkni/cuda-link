"""Sample GPU load with NVML while a benchmark cell runs.

Device: GPU % and memory-controller % (share of time busy), SM clock and power, every
100 ms. Process: SM % per PID from nvmlDeviceGetProcessUtilization, every 500 ms.
Prints one summary line (p50/p95/max) and, with --out, appends it to a Markdown file.

NVML "GPU %" is the share of time a kernel ran, and SM % is per process. Task Manager's
"3D" value is a different measure (the 3D queue's busy time, which includes UI drawing and
present waits): use gpu_engine_probe.ps1 for that one. See docs/BENCHMARKS.md "GPU load".

Needs the `nvml` extra (nvidia-ml-py).

Usage:
    python scripts/profiling/gpu_load_probe.py 30 G1_tox --label 1234=TD_A --label 5678=TD_B
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import pynvml as nv
except ImportError as e:  # pragma: no cover - optional dependency
    raise SystemExit("gpu_load_probe needs nvidia-ml-py: uv sync --extra nvml") from e


def _q(values: list[float], frac: float) -> float | None:
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * frac))] if v else None


def _fmt(values: list[float]) -> str:
    return f"{_q(values, 0.5)}/{_q(values, 0.95)}/{max(values) if values else None}"


def sample(seconds: float, device: int = 0) -> tuple[dict[str, list[float]], dict[int, list[float]]]:
    """Return device samples and per-PID SM % samples over `seconds`."""
    nv.nvmlInit()
    try:
        h = nv.nvmlDeviceGetHandleByIndex(device)
        dev: dict[str, list[float]] = {"gpu": [], "mem": [], "sm_mhz": [], "w": []}
        proc: dict[int, list[float]] = {}
        last_ts = 0
        t0 = time.perf_counter()
        next_proc = t0
        while time.perf_counter() - t0 < seconds:
            u = nv.nvmlDeviceGetUtilizationRates(h)
            dev["gpu"].append(u.gpu)
            dev["mem"].append(u.memory)
            dev["sm_mhz"].append(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
            dev["w"].append(nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
            now = time.perf_counter()
            if now >= next_proc:
                next_proc = now + 0.5
                try:
                    for s in nv.nvmlDeviceGetProcessUtilization(h, last_ts):
                        last_ts = max(last_ts, s.timeStamp)
                        proc.setdefault(s.pid, []).append(s.smUtil)
                except nv.NVMLError:
                    pass  # NVML_ERROR_NOT_FOUND: no new process samples in this window
            time.sleep(0.1)
        return dev, proc
    finally:
        nv.nvmlShutdown()


def summarize(
    tag: str, seconds: float, dev: dict[str, list[float]], proc: dict[int, list[float]], names: dict[int, str]
) -> str:
    parts = [
        f"gpu% {_fmt(dev['gpu'])}",
        f"mem% {_fmt(dev['mem'])}",
        f"SM MHz p50 {_q(dev['sm_mhz'], 0.5)}",
        f"W p50 {_q(dev['w'], 0.5):.0f}",
    ]
    for pid, v in sorted(proc.items(), key=lambda kv: -sum(kv[1])):
        if pid in names or max(v) > 0:
            parts.append(f"{names.get(pid, pid)} sm% {_fmt(v)} (n={len(v)})")
    return f"- {tag} ({seconds:.0f} s, p50/p95/max): " + " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seconds", type=float)
    ap.add_argument("tag")
    ap.add_argument(
        "--label", action="append", default=[], metavar="PID=NAME", help="name a process in the output (repeatable)"
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, help="append the summary line to this file")
    a = ap.parse_args()
    names = {int(k): v for k, v in (lbl.split("=", 1) for lbl in a.label)}
    dev, proc = sample(a.seconds, a.device)
    line = summarize(a.tag, a.seconds, dev, proc, names)
    print(line)
    if a.out:
        with a.out.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
