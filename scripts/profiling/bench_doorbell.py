"""
bench_doorbell.py — R2 Win32 doorbell poll-vs-doorbell benchmark.

Measures consumer idle CPU % and frame notify latency for two strategies
against a fixed-rate GPU producer:

  poll      Baseline: on NO_FRAME, sleep 1 ms then retry.
            (Existing behaviour when CUDALINK_DOORBELL is unset or 0.)

  doorbell  R2 opt-in: on NO_FRAME, block on the Win32 named auto-reset
            event that the producer signals after each publish_frame().
            Falls back to 1 ms sleep only on the 2 s safety timeout.

  native    R5 opt-in: native (C++) notification-wait backend, engaged via
            ImportPolicy(wait_backend="native"). Added only when
            cuda-link-native is importable on this host (Windows + sidecar
            installed). This is the authoritative venue for PLAN-002's accept
            gate (p50<10us, p95<50us) because imp.last_latency here is real
            cross-process publish->detect latency -- unlike
            bench_r1_wait.py's get_frame() wall-clock time, which includes
            tensor materialization and can never satisfy a 10us bound.

Both arms run at 30 fps and 60 fps.  At 30 fps the inter-frame gap is
~33 ms — the idle-CPU difference between the two strategies is largest here.

CPU % is measured via Win32 GetProcessTimes (kernel32 ctypes, no psutil).
Notify latency is imp.last_latency: producer publish_frame timestamp to
consumer _begin_frame, in milliseconds.

Every arm's ImportPolicy pins wait_backend explicitly (poll/doorbell -> "python",
native -> "native") when the field exists. ImportPolicy()'s own default is
wait_backend="auto", not "python" -- leaving it unset would let the poll/doorbell
arms silently pick up the native backend on any host with cuda-link-native
installed, contaminating the R2 poll-vs-doorbell comparison (same bug class
fixed for bench_r1_wait.py in commit 0942985).

Usage
-----
Full matrix (2 fps × 2 arms, ~3–5 min wall-clock):
    python scripts/profiling/bench_doorbell.py

Single fps, fewer frames (smoke test, ~30 s):
    python scripts/profiling/bench_doorbell.py --fps 30 --frames 100

Options
-------
    --frames N    Measurement frames per arm after warmup (default 300)
    --fps F       Run only this fps value (default: both 30 and 60)
    --outfile P   JSON output path (default: .profiling/doorbell.json)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import sys
import time
import uuid
from pathlib import Path
from statistics import median, quantiles

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 40
# 512×512 RGBA float32 — 4 MB; same as bench_r1_wait.py for comparability.
HEIGHT, WIDTH, CHANNELS, DTYPE = 512, 512, 4, "float32"

# ---------------------------------------------------------------------------
# Win32 idle-CPU measurement — kernel32 ctypes, no psutil dependency.
# Matches the WinDLL guard pattern used in cuda_ipc_wrapper.py.
# ---------------------------------------------------------------------------

if os.name == "nt":
    import ctypes as _ctypes_cpu
    from ctypes import wintypes as _wt

    class _FILETIME(_ctypes_cpu.Structure):
        _fields_ = [("dwLow", _wt.DWORD), ("dwHigh", _wt.DWORD)]

    _k32_cpu = _ctypes_cpu.WinDLL("kernel32", use_last_error=True)
    _k32_cpu.GetCurrentProcess.restype = _wt.HANDLE
    _k32_cpu.GetCurrentProcess.argtypes = []
    _k32_cpu.GetProcessTimes.restype = _wt.BOOL
    _k32_cpu.GetProcessTimes.argtypes = [
        _wt.HANDLE,
        _ctypes_cpu.POINTER(_FILETIME),
        _ctypes_cpu.POINTER(_FILETIME),
        _ctypes_cpu.POINTER(_FILETIME),
        _ctypes_cpu.POINTER(_FILETIME),
    ]

    def _get_cpu_100ns() -> int:
        """Return kernel+user CPU ticks for the current process (100-ns units)."""
        creation, exit_, kernel, user = (
            _FILETIME(),
            _FILETIME(),
            _FILETIME(),
            _FILETIME(),
        )
        _k32_cpu.GetProcessTimes(
            _k32_cpu.GetCurrentProcess(),
            _ctypes_cpu.byref(creation),
            _ctypes_cpu.byref(exit_),
            _ctypes_cpu.byref(kernel),
            _ctypes_cpu.byref(user),
        )
        k = (kernel.dwHigh << 32) | kernel.dwLow
        u = (user.dwHigh << 32) | user.dwLow
        return k + u

else:
    # Non-Windows: GetProcessTimes is unavailable; cpu_pct will report 0.
    def _get_cpu_100ns() -> int:  # type: ignore[misc]
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: int) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return quantiles(values, n=100)[p - 1]


def _wait_for_shm(shm_name: str, timeout_s: float = 20.0) -> bool:
    """Poll until the named SharedMemory segment appears (producer has created it)."""
    from multiprocessing.shared_memory import SharedMemory

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            return True
        except FileNotFoundError:
            time.sleep(0.05)
    return False


def _policy_has_wait_backend() -> bool:
    """Return True if ImportPolicy has a 'wait_backend' field (R5 implemented)."""
    try:
        from cuda_link._importer_port import ImportPolicy

        return any(f.name == "wait_backend" for f in dataclasses.fields(ImportPolicy))
    except Exception:  # noqa: BLE001
        return False


def _native_backend_available() -> bool:
    """Return True if the R5 native wait backend can actually be engaged here.

    Requires: Windows (the native sidecar is Windows-only), ImportPolicy has the
    'wait_backend' field (R5 implemented), and cuda_link_native is importable
    (the sidecar package is installed on this host).
    """
    if os.name != "nt":
        return False
    if not _policy_has_wait_backend():
        return False
    try:
        import cuda_link_native  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Worker functions — module-level so multiprocessing spawn can pickle them.
# ---------------------------------------------------------------------------


def _worker_producer(
    shm_name: str,
    num_frames: int,
    fps: float,
    doorbell_on: bool,
    result_q: object,
) -> None:
    """Export synthetic GPU frames at a fixed cadence."""
    import ctypes

    try:
        from cuda_link import FrameSpec, GpuFrame
        from cuda_link._exporter_port import ExportPolicy
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
        from cuda_link.exporter import Exporter

        cuda = get_cuda_runtime()
        policy = ExportPolicy(doorbell=doorbell_on)
        exporter = Exporter.open(
            FrameSpec(
                shm_name=shm_name,
                height=HEIGHT,
                width=WIDTH,
                channels=CHANNELS,
                dtype=DTYPE,
                num_slots=2,
            ),
            policy=policy,
        )

        nbytes = exporter.data_size
        src_ptr = cuda.malloc(nbytes)
        if src_ptr is None or src_ptr.value == 0:
            result_q.put(("ERROR", "cudaMalloc failed in producer"))
            return

        host_buf = (ctypes.c_uint8 * nbytes)(*([0] * nbytes))
        cuda.memcpy(
            dst=src_ptr,
            src=ctypes.c_void_p(ctypes.addressof(host_buf)),
            count=nbytes,
            kind=1,
        )
        exporter.record_source_sync(0)

        # Extra headroom: warmup + measurement + safety margin
        total = WARMUP_FRAMES + num_frames + 30
        sleep_s = 1.0 / fps
        for _ in range(total):
            exporter.export(GpuFrame(ptr=int(src_ptr.value), size=nbytes))
            time.sleep(sleep_s)

        time.sleep(2.0)  # give consumer time to drain the last frame
        cuda.free(src_ptr)
        exporter.close()
        result_q.put(("OK", None))
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer(
    shm_name: str,
    num_frames: int,
    doorbell_on: bool,
    wait_backend: str,
    result_q: object,
) -> None:
    """Consume frames; capture idle CPU % and frame notify latency."""
    try:
        from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
        from cuda_link.importer import Importer

        if not _wait_for_shm(shm_name, timeout_s=30.0):
            result_q.put(("ERROR", f"SharedMemory '{shm_name}' never appeared"))
            return

        time.sleep(0.4)  # let producer write IPC handles into the SHM header

        policy_kwargs: dict = {"doorbell": doorbell_on, "debug": True}
        if _policy_has_wait_backend():
            # ImportPolicy()'s own default is wait_backend="auto", not "python" --
            # omitting this would let the poll/doorbell arms silently pick up the
            # native backend via "auto" on any host with cuda-link-native
            # installed, contaminating the R2 poll-vs-doorbell comparison (same
            # bug class fixed for bench_r1_wait.py in 0942985). The native arm
            # passes wait_backend="native" explicitly instead.
            policy_kwargs["wait_backend"] = wait_backend
        policy = ImportPolicy(**policy_kwargs)
        spec = ImportSpec(shm_name=shm_name, shape=(HEIGHT, WIDTH, CHANNELS), dtype=DTYPE)
        imp = Importer.open(spec, policy=policy)

        # ------------------------------------------------------------------
        # Warmup: discard WARMUP_FRAMES without recording metrics.
        # Uses the same NO_FRAME strategy as the measurement phase so the
        # doorbell event handle is exercised during warmup too.
        # ------------------------------------------------------------------
        warmup_done = 0
        warmup_deadline = time.perf_counter() + 60.0
        while warmup_done < WARMUP_FRAMES and time.perf_counter() < warmup_deadline:
            r = imp.get_frame()
            if r.outcome is ImportOutcome.NEW_FRAME:
                warmup_done += 1
            elif r.outcome is ImportOutcome.NO_FRAME:
                # wait_for_doorbell: returns False immediately when handle is None
                # (doorbell_on=False), so poll arm always falls through to sleep.
                if not imp.wait_for_doorbell(2000.0):
                    time.sleep(0.001)
            elif r.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                break

        # R5: verify the native backend actually engaged before measuring. Checked
        # here -- after warmup, not immediately after Importer.open() -- because
        # open() defers _connect() when reconnect_enabled (default True) and the
        # producer races the consumer; by now warmup frames have flowed so the
        # connection (and backend resolution, importer.py:1255) has settled.
        if wait_backend == "native" and getattr(imp, "_wait_backend", None) is None:
            result_q.put(("ERROR", "native wait_backend requested but did not engage (see importer logs)"))
            return

        # ------------------------------------------------------------------
        # Measurement phase: collect get_frame timing, notify latency, CPU%.
        # ------------------------------------------------------------------
        gf_samples: list[float] = []
        latency_samples: list[float] = []
        no_frame_count = 0

        cpu_start = _get_cpu_100ns()
        wall_start = time.perf_counter()

        frames_seen = 0
        meas_deadline = time.perf_counter() + 120.0
        while frames_seen < num_frames and time.perf_counter() < meas_deadline:
            t0 = time.perf_counter()
            r = imp.get_frame()
            gf_us = (time.perf_counter() - t0) * 1e6

            if r.outcome is ImportOutcome.NEW_FRAME:
                gf_samples.append(gf_us)
                # last_latency: producer publish_frame timestamp → consumer _begin_frame (ms).
                latency_samples.append(imp.last_latency)
                frames_seen += 1
            elif r.outcome is ImportOutcome.NO_FRAME:
                no_frame_count += 1
                if not imp.wait_for_doorbell(2000.0):
                    time.sleep(0.001)
            elif r.outcome in (ImportOutcome.SHUTDOWN, ImportOutcome.TIMEOUT):
                break

        wall_s = time.perf_counter() - wall_start
        cpu_100ns = _get_cpu_100ns() - cpu_start
        # cpu_pct: fraction of wall-clock time consumed as CPU (kernel+user).
        # 1 tick = 100 ns, so wall_s * 1e7 converts wall seconds to 100-ns units.
        cpu_pct = (cpu_100ns / (wall_s * 1e7) * 100.0) if wall_s > 0 else 0.0

        stats = imp.get_stats()
        imp.close()

        if len(gf_samples) < 10:
            result_q.put(("ERROR", f"Too few measurement samples: {len(gf_samples)}"))
            return

        def _lat(lst: list[float], p: int) -> float:
            return _percentile(lst, p) if len(lst) >= 2 else (lst[0] if lst else 0.0)

        result_q.put(
            (
                "OK",
                {
                    "n": len(gf_samples),
                    "no_frame_count": no_frame_count,
                    "cpu_pct": round(cpu_pct, 3),
                    "wall_s": round(wall_s, 2),
                    # get_frame() call duration (inner GPU-wait path + D2H, µs)
                    "gf_p50_us": median(gf_samples),
                    "gf_p95_us": _percentile(gf_samples, 95),
                    "gf_p99_us": _percentile(gf_samples, 99),
                    # notify latency: producer publish → consumer pickup (ms)
                    "latency_p50_ms": _lat(latency_samples, 50),
                    "latency_p95_ms": _lat(latency_samples, 95),
                    "latency_p99_ms": _lat(latency_samples, 99),
                    # importer spin/sleep telemetry (debug=True)
                    "wait_spin_hits": stats["wait_spin_hits"],
                    "wait_sleep_hits": stats["wait_sleep_hits"],
                    "avg_spin_us": stats["avg_spin_us"],
                    "avg_sleep_us": stats["avg_sleep_us"],
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        result_q.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# Cell runner — one (producer, consumer) pair per (fps, arm) cell
# ---------------------------------------------------------------------------


def _run_cell(
    label: str,
    fps: float,
    num_frames: int,
    doorbell_on: bool,
    wait_backend: str,
    ctx: object,
) -> dict | None:
    """Spawn one producer + one consumer; return the consumer result dict or None on failure."""
    shm_name = f"__bench_db_{uuid.uuid4().hex[:8]}"

    prod_q = ctx.Queue()
    cons_q = ctx.Queue()

    prod = ctx.Process(
        target=_worker_producer,
        args=(shm_name, num_frames, fps, doorbell_on, prod_q),
        daemon=True,
    )
    cons = ctx.Process(
        target=_worker_consumer,
        args=(shm_name, num_frames, doorbell_on, wait_backend, cons_q),
        daemon=True,
    )

    prod.start()
    cons.start()

    try:
        status, payload = cons_q.get(timeout=120)
    except Exception as e:  # noqa: BLE001
        prod.terminate()
        cons.terminate()
        print(f"  [{label}] TIMEOUT waiting for consumer: {e}")
        return None

    cons.join(timeout=5)
    prod.join(timeout=15)

    if status == "ERROR":
        print(f"  [{label}] ERROR: {payload}")
        return None

    r = payload
    print(
        f"  [{label:<36}]"
        f"  CPU={r['cpu_pct']:5.1f}%"
        f"  lat p50/p95/p99={r['latency_p50_ms']:.2f}/{r['latency_p95_ms']:.2f}/{r['latency_p99_ms']:.2f} ms"
        f"  gf_p50={r['gf_p50_us']:6.0f} µs"
        f"  NO_FRAME={r['no_frame_count']:6d}"
        f"  n={r['n']}"
    )
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="R2 doorbell benchmark — poll vs. Win32 named-event doorbell")
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Measurement frames per arm after warmup (default 300)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="Producer FPS (default: run both 30 and 60 fps scenarios)",
    )
    parser.add_argument(
        "--outfile",
        default=".profiling/doorbell.json",
        help="JSON output path (default: .profiling/doorbell.json)",
    )
    args = parser.parse_args()

    fps_list: list[float] = [args.fps] if args.fps > 0 else [30.0, 60.0]

    print("=" * 72)
    print("R2 doorbell benchmark — poll (baseline) vs. Win32 named-event doorbell")
    print(f"  Resolution  : {HEIGHT}x{WIDTH}x{CHANNELS} {DTYPE}")
    print(f"  Frames/arm  : {args.frames} measurement + {WARMUP_FRAMES} warmup")
    print(f"  FPS         : {fps_list}")
    print(f"  CPU method  : {'Win32 GetProcessTimes (kernel32)' if os.name == 'nt' else 'N/A (non-Windows)'}")
    print("=" * 72)

    ctx = multiprocessing.get_context("spawn")
    all_results: dict[str, dict] = {}

    for fps in fps_list:
        fps_label = f"{fps:.0f}fps"
        print(f"\n--- Scenario: {fps_label} ---\n")
        scenario: dict[str, dict] = {}

        arms: list[tuple[bool, str, str]] = [(False, "poll", "python"), (True, "doorbell", "python")]
        if _native_backend_available():
            arms.append((True, "native", "native"))
        else:
            print("  [native arm skipped -- cuda-link-native not importable on this host]")
        for doorbell_on, arm_key, wait_backend in arms:
            label = f"{fps_label}/{arm_key}"
            r = _run_cell(label, fps, args.frames, doorbell_on, wait_backend, ctx)
            if r is not None:
                scenario[arm_key] = r

        all_results[fps_label] = scenario

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------
    for fps_label, scenario in all_results.items():
        print()
        print("=" * 72)
        print(f"SUMMARY: {fps_label}")
        print("=" * 72)
        hdr = f"  {'Arm':<12} {'CPU%':>6} {'lat-p50':>9} {'lat-p95':>9} {'lat-p99':>9} {'NO_FRAME':>9} {'n':>5}"
        print(hdr)
        print("  " + "-" * 63)
        for arm_key, arm_label in [("poll", "poll"), ("doorbell", "doorbell"), ("native", "native (R5)")]:
            if arm_key not in scenario:
                continue
            r = scenario[arm_key]
            print(
                f"  {arm_label:<12}"
                f" {r['cpu_pct']:>5.1f}%"
                f" {r['latency_p50_ms']:>8.2f}ms"
                f" {r['latency_p95_ms']:>8.2f}ms"
                f" {r['latency_p99_ms']:>8.2f}ms"
                f" {r['no_frame_count']:>9}"
                f" {r['n']:>5}"
            )

        if "poll" in scenario and "doorbell" in scenario:
            p = scenario["poll"]
            d = scenario["doorbell"]
            cpu_delta = p["cpu_pct"] - d["cpu_pct"]
            lat95_delta = p["latency_p95_ms"] - d["latency_p95_ms"]
            noframe_ratio = p["no_frame_count"] / max(d["no_frame_count"], 1)
            print()
            print(f"  CPU reduction   : {p['cpu_pct']:.1f}% -> {d['cpu_pct']:.1f}%  ({cpu_delta:+.1f} pp)")
            print(
                f"  Latency p95     : {p['latency_p95_ms']:.2f} ms -> {d['latency_p95_ms']:.2f} ms"
                f"  ({lat95_delta:+.2f} ms)"
            )
            print(
                f"  NO_FRAME count  : {p['no_frame_count']} -> {d['no_frame_count']}  ({noframe_ratio:.0f}x reduction)"
            )

        if "doorbell" in scenario and "native" in scenario:
            d = scenario["doorbell"]
            n = scenario["native"]
            native_p50_us = n["latency_p50_ms"] * 1000.0
            native_p95_us = n["latency_p95_ms"] * 1000.0
            doorbell_p50_us = d["latency_p50_ms"] * 1000.0
            doorbell_p95_us = d["latency_p95_ms"] * 1000.0
            cpu_delta_r5 = d["cpu_pct"] - n["cpu_pct"]
            gate = "PASS" if native_p50_us < 10.0 and native_p95_us < 50.0 else "MISS"
            print()
            print(
                f"  R5 CPU (doorbell->native)     : {d['cpu_pct']:.1f}% -> {n['cpu_pct']:.1f}%  ({cpu_delta_r5:+.1f} pp)"
            )
            print(
                f"  R5 latency p50 (doorbell->native): {doorbell_p50_us:.1f}us -> {native_p50_us:.1f}us"
                f"  (delta {doorbell_p50_us - native_p50_us:+.1f}us)"
            )
            print(
                f"  R5 latency p95 (doorbell->native): {doorbell_p95_us:.1f}us -> {native_p95_us:.1f}us"
                f"  (delta {doorbell_p95_us - native_p95_us:+.1f}us)"
            )
            print(
                f"  R5 accept gate (publish->detect) : p50<10us p95<50us -- "
                f"p50={native_p50_us:.1f}us p95={native_p95_us:.1f}us [{gate}]"
            )

    print()
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "2",
        "config": {
            "height": HEIGHT,
            "width": WIDTH,
            "channels": CHANNELS,
            "dtype": DTYPE,
            "frames": args.frames,
            "warmup_frames": WARMUP_FRAMES,
            "fps_list": fps_list,
            "r5_present": _native_backend_available(),
        },
        "results": all_results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
