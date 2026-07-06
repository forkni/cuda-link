# cuda-link Benchmarks

All results measured on **RTX 4090 / PCIe 4.0 x16 / Windows 11 / NVIDIA driver 596.36**.
RGBA (4-channel) frames unless noted. Numbers produced by scripts in the local-only
`benchmarks/` folder (gitignored). See [README](../README.md) for a summary view and
[ARCHITECTURE.md](ARCHITECTURE.md#comparison-cuda-ipc-vs-cpu-sharedmemory) for
methodology and hardware caveats.

> **Reproduction**: The benchmark scripts (`bench_graphs.py`, `bench_d2h_streams.py`,
> `bench_sweep.py`) are not included in the repository. Contributors with a local clone
> from a version predating `v1.4.1` will find them in their on-disk `benchmarks/` folder.
> See the [v1.1.0 CHANGELOG entry](../CHANGELOG.md) for the bench_sweep design and the
> [v1.2.1 CHANGELOG entry](../CHANGELOG.md) for the benchmark refresh methodology.

---

## Summary

| Operation | p50 | Notes |
|-----------|-----|-------|
| `export_frame()` — 512×512 RGBA float32 | 24 µs | Standalone, EXPORT_SYNC=1; GPU D2D + stream_synchronize |
| `export_frame()` — 1080p RGBA float32 | 106 µs | Standalone, EXPORT_SYNC=1 |
| `export_frame()` — 4K RGBA float32 | 357 µs | Standalone, EXPORT_SYNC=1 |
| `get_frame_numpy()` D2H — 512×512 float32 | 0.18 ms | Standalone, ~22 GB/s |
| `get_frame_numpy()` D2H — 1080p float32 | 1.32 ms | Standalone, ~24 GB/s PCIe 4.0 |
| `get_frame_numpy()` D2H — 4K float32 | 5.7 ms | Standalone, ~21 GB/s PCIe 4.0 |
| `get_frame()` / `get_frame_cupy()` GPU | <5 µs | Zero-copy tensor/array view, no D2H |
| IPC notification latency | ~136–286 µs | Producer publish → consumer detect (cross-process) |
| Initialization | ~50–100 µs | One-time IPC handle opening |

> **Telemetry note (v1.10.3):** The `export=` (Sender) and `copy=` (Receiver) averages shown
> in the Debug summary line are **windowed (~150-frame) averages**, not lifetime cumulative
> means. They reset with each report window (period controlled by
> `CUDALINK_SENDER_REPORT_EVERY` / `CUDALINK_RECEIVER_REPORT_EVERY`). Cross-version timing
> comparisons that use these reported averages must account for this change.

---

## `export_frame()` — P1 Async Gain & P3 CUDA Graph Submission Collapse

**2×2 matrix** at 1080p uint8 RGBA, 2000 frames, single-process standalone.
Measured 2026-06-10 via `scripts/profiling/profile_export.py --frames 2000 --export-profile`.

```
Cell   EXPORT_SYNC   USE_GRAPHS   median µs   p95 µs   p99 µs   sync region µs   WDDM subs/frame
----   -----------   ----------   ---------   ------   ------   ---------------   ---------------
A      1 (blocking)  0 (off)         45.2      51.2    106.3         17.91               3
B      0 (async)     0 (off)         17.9      29.3     52.1          0.00               3
C      1 (blocking)  1 (on)          28.8      43.1     52.1         12.88               1
D      0 (async)     1 (on)          13.9      22.7     25.1          0.00               1
```

**P1 gain (A→B): −27.3 µs p50 (60%)** — eliminates the 17.91 µs `cudaStreamSynchronize`
on the producer side. `CUDALINK_EXPORT_SYNC=0` enables async mode; `CUDALINK_EXPORT_SYNC=1`
forces blocking. **As of v1.10.1 the TD Sender defaults to blocking** (Cell A / Cell C) to
prevent a source-buffer lifetime race with TD's transient cook-scoped TOP texture that causes
CUDA 719 under a loaded consumer (see CHANGELOG 1.10.1). The Python library `Exporter` keeps
async as its default (Cell D / Cell B). Coexistence safety comes from explicit per-engine
streams and producer-stream ordering (`record_source_sync` / `require_source_sync`), not from
blocking export.

**P3 graph gain (B→D): −4.0 µs p50 (22%)** — `cudaStreamWaitEvent` + `cudaEventRecord`
are folded into the CUDA graph; only `cudaGraphLaunch` fires per frame. nsys confirms:
graphs OFF → 3 submissions/frame (waitEvent + memcpyAsync + eventRecord = 550 calls each);
graphs ON → 1 submission/frame (`cudaGraphLaunch`, 550 calls; waitEvent/eventRecord absent).

**Full P1+P3 (A→D): −31.3 µs p50 (69%)**

`per_region_avg_us` breakdown for reference:

```
Region            Cell A (µs)   Cell D (µs)   Notes
--------------    -----------   -----------   ----------------------------------------
stream_wait            1.26          0.00     folded into CUDA graph (P3)
memcpy                 7.76          6.77     PCIe-bound, ~22 GB/s
record_event           4.50          0.00     folded into CUDA graph (P3)
shm_write              1.92          1.21     SHM slot write (unaffected)
sync                  17.91          0.00     cudaStreamSynchronize eliminated (P1)
flush_probe            0.00          1.13     stream query kick (async path only)
```

Note: WDDM submission count/frame is inferred from nsys call counts ÷ frame count.

Reproduce (all four cells; on Windows use `SET VAR=value &` prefix):
```bash
CUDALINK_EXPORT_SYNC=1 CUDALINK_USE_GRAPHS=0 python scripts/profiling/profile_export.py --frames 2000 --export-profile --outfile .profiling/exp_A.json
CUDALINK_EXPORT_SYNC=0 CUDALINK_USE_GRAPHS=0 python scripts/profiling/profile_export.py --frames 2000 --export-profile --outfile .profiling/exp_B.json
CUDALINK_EXPORT_SYNC=1 CUDALINK_USE_GRAPHS=1 python scripts/profiling/profile_export.py --frames 2000 --export-profile --outfile .profiling/exp_C.json
CUDALINK_EXPORT_SYNC=0 CUDALINK_USE_GRAPHS=1 python scripts/profiling/profile_export.py --frames 2000 --export-profile --outfile .profiling/exp_D.json
```

---

## `export_frame()` — CUDA Graphs A/B (historical, EXPORT_SYNC=1)

Single-process, EXPORT_SYNC=1 (CPU waits for GPU D2D completion), 2000 frames.
Historical baseline pre-P1/P3 (v1.4.1 era). Updated blocking-arm numbers via
`scripts/profiling/profile_export.py` (v1.10.1, 2026-06-10): 512×512 f32 → **24 µs**,
1080p f32 → **106 µs**, 4K f32 → **357 µs** (see Summary table above; graphs ON,
driver 596.36). Historical table retained for the graphs ON vs OFF comparison.

```
Resolution    Graphs off (p50 µs)   Graphs on (p50 µs)
----------    -------------------   ------------------
512x512                      22.4                 19.4
1280x720                     42.7                 41.7
1920x1080                   117.1                115.7
3840x2160                   367.4                366.9
```

With EXPORT_SYNC=1 the GPU D2D copy dominates; CUDA Graphs saves WDDM submission
transitions but the net wall-clock difference is small (<5%). See the P1/P3 section above
for the full breakdown including async mode (the large win).

Reproduce with:
```bash
python benchmarks/bench_graphs.py --frames 2000 --sizes 512 1280 1920 3840
```

---

## `get_frame_numpy()` — P5 Pipelined D2H Double-Buffer

**Opt-in** via `CUDALINK_D2H_PIPELINED=1`. Overlaps the D2H copy with the consumer's CPU
work by enqueuing the next copy asynchronously while returning the previous frame. First
call returns `NO_FRAME` (priming); steady-state adds +1 frame latency. **On reconnect
(v1.10.3)**, the pipeline drains and re-primes — one additional `NO_FRAME` per reconnect
event (same priming contract as initial open).

**When to enable:** only when consumer CPU work time > D2H copy time. Break-even at 4K ≈
1.3 ms workload; at 1080p ≈ 0.38 ms. Disabled by default pending broader validation.

Measured 2026-06-10 via `scripts/profiling/bench_d2h_pipelined.py`, 5 ms synthetic CPU
workload, 150 measurement frames (30 warmup), spawn-process IPC pair.

```
Resolution   non-pipe d2h µs   non-pipe cycle µs   pipe d2h µs   pipe cycle µs   gain µs   gain %   priming NO_FRAME
----------   ---------------   -----------------   -----------   -------------   -------   ------   ----------------
512x512                  97              5099              89            5091         8       0%      YES
1920x1080               383              5384              74            5075       309       6%      YES
3840x2160              1354              6355              78            5079      1276      20%      YES
```

D2H copy rates derived from non-pipelined d2h time: 512² ≈ 10 GB/s (1 MB frame, latency
dominated), 1080p ≈ 22 GB/s, 4K ≈ 25 GB/s. Pipelined d2h ≈ 85 µs is residual
`stream_synchronize` noise (copy completes during 5 ms workload).

Gain formula: `cycle_gain ≈ d2h_copy_time` when `workload > copy`. At 4K with 5 ms work:
predicted gain = 1320 µs; measured = 1235 µs (within 7%).

P5 contract verified on real GPU: first `get_frame_numpy()` returns `NO_FRAME` (priming)
on all three resolutions.

Reproduce:
```bash
python scripts/profiling/bench_d2h_pipelined.py --resolution all --work-ms 5 --frames 150
```

---

## `get_frame_numpy()` D2H — stream count

Standalone D2H copy, no IPC overhead, 2000 frames.

```
Resolution    1 stream p50 (ms)   2 streams p50 (ms)   1 stream GB/s
----------    -----------------   ------------------   -------------
512x512                    0.18                 0.19            22.2
1280x720                   0.61                 0.61            23.1
1920x1080                  1.32                 1.34            23.7
3840x2160                  5.69                 6.82            21.4
```

PCIe 4.0 saturates at ~23–24 GB/s. Single stream is sufficient; `CUDALINK_D2H_STREAMS=1`
(default) is optimal for this platform.

Reproduce with:
```bash
python benchmarks/bench_d2h_streams.py --frames 2000 --streams 1 2 --sizes 512 1280 1920 3840
```

---

## Full IPC Roundtrip Sweep

Two separate Python processes (producer + consumer), 500 warmup + 2000 measurement frames
at 60 FPS. `export p50` and `get_numpy p50` are inflated vs standalone because both
processes share PCIe bandwidth concurrently. `IPC notify p50` measures
producer-publish → consumer-detects-write_idx (signaling latency, resolution-independent).

```
Resolution    dtype     Graphs   export p50 (µs)   get_numpy p50 (ms)   IPC notify p50 (µs)
----------    -------   ------   ---------------   ------------------   -------------------
512x512       float32   off                898                 1.33                     172
512x512       float32   on                 885                 1.33                     200
512x512       uint8     off                871                 0.38                     203
1280x720      float32   off                907                 4.48                     160
1920x1080     float32   off               1483                 5.02                     136
1920x1080     uint8     off                873                 2.54                     179
3840x2160     float32   off                662                 5.01                     286
3840x2160     uint8     off               1471                 5.03                     196
```

Full 16-cell results (CSV + JSON) live in the local `benchmarks/results/` folder.

> **v1.10.2 P11 note:** When the producer is idle (no new frame), the receiver's Script-TOP
> cook is now skipped entirely, so observable cook counts in slow-producer scenarios will be
> lower than in pre-v1.10.2 measurements. Sweep figures above use a 60 FPS active producer
> and are unaffected.

Reproduce with:
```bash
python benchmarks/bench_sweep.py          # full 16-cell sweep (~12 min)
python benchmarks/bench_sweep.py --quick  # smoke test, 1 cell (~1 min)
```

---

## TD-in-the-loop: Original (TOX) vs C++ Custom TOP

First head-to-head with TouchDesigner actually in the loop, comparing the pre-existing
pure-Python **Original** path (`CUDAIPCExtension` Sender/Receiver, `.tox`) against the native
**C++ Custom TOP** operators (`CudaLinkOutTOP` / `CudaLinkInTOP`). Measured 2026-07-05,
RTX 4090 / driver 596.36, 1920×1080 RGBA uint8, 60 FPS target, windowed (~97–150-frame)
averages. **TD→TD uses two separate TD instances** (same topology as the C++ TOPs' own
cross-process design), not an in-process loopback.

```
Direction      Metric                    Original (TOX)              C++ Custom TOP
-----------    ----------------------    -------------------------   ------------------------
TD -> Python   FPS                       59.7                        60.0
               consumer read             get_frame 39.9 us avg       get_frame 72.2 us avg
               latency                   0.01-1.50 ms                0.03-1.53 ms

Python -> TD   FPS                       ~58.7                       58.6-58.8
               transfer                  TD copy 77-100 us           In TOP copy_us ~21 us [1]
               sender export             not captured this run       4693-6213 us [2]
               latency (TD side)         1.33 -> 11.8 ms (drifts)    not captured

TD -> TD       FPS                       57-60                       ~57-60
               transfer copy             83-167 us                   ~21 us (In) / ~28 us (Out)
               latency                   13.5-15.8 ms                not logged by C++ path [3]

Python<->Python  export / get_numpy /    873 us / 2.54 ms / 179 us   N/A -- C++ TOPs are TD
                 IPC-notify p50          (see Full IPC Roundtrip      operators only, no
                                          Sweep above)                Python<->Python mode
```

[1] Different measurement domain than the Original's Python-side `copy=` — the In TOP's
`copy_us` is the op's own D2D-copy timing, not a Python-process read.

[2] Per `td_exporter/example_sender_python.py:274`, this window (`export_us = (now - t0) * 1e6`)
bundles the synchronous H2D staging fill (`_fill_ctypes`) *and* `exporter.export()`, excluding
the frame-pacing sleep. It is **not** comparable to the Original's TD-side `export=` line, which
only times the extension's own export path.

[3] The C++ TOPs' `bench:` telemetry (`avg_cook_us` / `avg_copy_us` / `avg_begin_us` /
`avg_end_us`, from `%TEMP%\cudalink_{in,out}_top_debug.log` when Debug is enabled) has no
end-to-end wire-timestamp latency field, unlike the Original's `latency=` (producer-publish →
consumer-detect delta). This is a known instrumentation gap, not a zero.

**Headline 1 — raw GPU transfer is ~4-8x faster on the C++ TOPs.** The one clean
apples-to-apples number in this table: TD→TD `copy` drops from 83–167 µs (Original) to ~21 µs
(C++ In TOP). Both `cook_us`/`copy_us` values sit well under one frame period even at 60 FPS.

**Headline 2 — the ~14 ms TD→TD latency is a cook-cadence artifact, not a transport cost.**
The Original's TD receiver is frame-loop-polled (checks for a new frame once per TD cook), so
its latency floors near one 16.7 ms frame period regardless of how fast the underlying copy is
— the `copy` itself is still only 83–167 µs. Python receivers instead event-wait
(`WaitForSingleObject`) and see 0–1.5 ms latency for the same producer cadence. The C++ TOPs
almost certainly share this same TD-side cook-cadence floor for TD→TD (both are driven by TD's
cook loop), but it isn't measured above since the C++ `bench:` line doesn't carry an end-to-end
latency field (see [3]).

**Known gaps** (not filled with invented numbers): C++ TD→TD end-to-end latency; a single run
that captures both the Python-sender and TD-receiver halves of Python→TD for the same
implementation; Original Python-sender `export=` for its own Python→TD run.

### 2026-07-06 refresh — doorbell wait (`Framewaitms`) + GPU-side timing

Follow-up session to the table above: the In TOP gained an opt-in bounded doorbell wait
(`Framewaitms`, commit `63d85e6`) fixing a continuous 16F/32F stutter, plus Debug-gated
GPU-side timing (`gpu_ipc_us`/`gpu_pass_us` on the Out TOP, `event_wait_us`/`gpu_copy_us`
on the In TOP) and cadence counters (`noframe_ratio`, `rescued`). Numbers below are
windowed (~97-frame) averages from an interactive TD→TD session (format switches mid-run),
**not** controlled p50 runs — directionally solid, tighter methodology pending.

```
Format (TD->TD, 60 FPS)   Out cook/gpu_ipc/gpu_pass (µs)   In cook/event_wait/gpu_copy (µs)   noframe_ratio
-----------------------   ------------------------------   --------------------------------   -------------
1080p uint8   (8 MB)          320 /  60 / 109                  282 /  30 / 146                 0.007
1080p float32 (33 MB)         233 / 215 / 564                  490 / 509 / 210                 0.001
4K uint8      (33 MB)         262 / 308 / 662                  366 / 586 / 337                 0.012–0.021
```

**Stutter root cause: publish-vs-poll phase aliasing between two vsync-locked TD processes.**
Pre-fix, the In TOP's `noframe_ratio` sat at **9–31%** on 1080p 32F (silent repeat frames =
continuous judder). With `Framewaitms=4`, worst case anywhere is **0.026**; typical ≤0.7%.
Sweep of the wait budget: 2 ms → up to 2.6% repeats in bursts (phase drift outruns the
budget); 3 ms → 0.2% (phase-lucky run, not proof); 4 ms → reliable across all formats.
Cook cost including the wait stays 280–490 µs — the budget is a cap, not a per-frame cost
(the doorbell wakes the cook the moment the producer publishes; ~73% of frames arrived via
the rescue path in the long wait=4 run).

**The Original is not immune — same pathology, wire-timestamped.** A fresh Original (TOX)
TD→TD 1080p float32 run the same day shows its `latency=` (publish→detect) drifting across
the whole frame period: 0.5 → 16.7 ms, including ~450 consecutive frames parked at
15.4–16.7 ms with effective fresh-frame rate sagging to 52–56 FPS (5–8 repeats/s — the same
magnitude as the C++ pre-fix `noframe_ratio`). Its Script-TOP receiver polls once per cook
and never calls the library's `wait_for_doorbell()`, i.e. it has the C++ In TOP's *pre-fix*
behavior. The Original's sender-side publish jitter (blocking `EXPORT_SYNC=1` export:
uint8 ≈ 570 µs avg; float32 206–3,112 µs across 150-frame windows) partially dithers the
phase, which is why the aliasing rarely parks as long as the C++ pair's metronome-steady
publishes did pre-fix.

**Instrumentation footnote:** the In TOP `copy_us` CPU bracket now reads ~107 µs vs ~21 µs
in the 2026-07-05 table. That delta is the Debug-gated `GpuTimerRing` `cudaEventRecord`
enqueues added inside the bracket (~20 µs each under WDDM), not a transport regression —
the Debug-off hot path is unchanged.

**Gap status vs the list above:** Original TD→TD latency at float32 now captured (drifting,
see above). C++ TD→TD end-to-end latency remains open — the planned `frame_age_ms` field
(producer QPC publish stamp already in the slot metadata) would close it.

---

## vs CPU SharedMemory

End-to-end at typical resolutions (float32 RGBA), CUDA-Link vs UT_SharedMem-class CPU
SharedMemory baseline (PCIe 4.0):

```
Resolution    Method              Producer write   Consumer read   E2E
----------    ----------------    --------------   -------------   ---------
1920x1080     CPU SharedMemory          2.60 ms         2.48 ms     5.37 ms
1920x1080     CUDA-Link                  138 µs         1.35 ms     ~1.6 ms      (~3.4x faster E2E)
512x512       CPU SharedMemory           361 µs          350 µs     1.02 ms
512x512       CUDA-Link                   42 µs         0.23 ms    ~0.49 ms      (~2.1x faster E2E)
```

Producer write is 4–19× faster (no CPU transit). With zero-copy GPU consumers
(`get_frame()` / `get_frame_cupy()`), the read path collapses to <5 µs and the
end-to-end gap widens further.

**TouchOUT and Spout** baselines were never measured — see methodology notes in
[ARCHITECTURE.md](ARCHITECTURE.md#comparison-cuda-ipc-vs-cpu-sharedmemory) for the
full hardware caveats and source data.

---

## R5: Native notification-wait accelerator (`Importer._wait_for_slot`)

**Opt-in** via the `cuda-link-native` sidecar (installer default-on on Windows via
`--native`; `--no-native` reverts to pure-Python). `ImportPolicy.wait_backend` selects
`"auto"` (default) | `"python"` | `"native"`. See
[PLAN-002](plans/PLAN-002-native-waiter.md) for the full design and
[ADR-0006](adr/0006-stay-pure-python-no-rust.md) for why this narrow native escape hatch
was taken instead of a broader Rust/C++ rewrite.

Measured 2026-07-04 via `scripts/profiling/bench_doorbell.py` (this script, not
`bench_r1_wait.py`, hosts the accept-gate measurement — see note below), 512×512 RGBA
float32, 300 measurement frames + 40 warmup, single producer/consumer process pair,
30 and 60 fps:

```
fps   Arm         CPU%   latency p50   latency p95
---   --------    ----   -----------   -----------
30    doorbell    1.1%      69.3 us       141.4 us
30    native      0.8%      66.4 us       138.7 us
60    doorbell    1.8%      64.6 us       113.4 us
60    native      0.3%      67.4 us       140.2 us
```

**Accept gate (PLAN-002: p50 < 10 µs, p95 < 50 µs, measured as producer-publish →
consumer-detect latency): MISS at both fps.** native and doorbell land within a few µs
of each other at both fps — well inside run-to-run noise, i.e. **native is not
measurably faster than plain doorbell on this hardware** for this metric.

This is not evidence that R5's own wait logic is slow: the native backend's internal
re-check-after-wake latency (`avg_spin_us`, measured separately via
`bench_r1_wait.py`) is ~0.02–6.5 µs — genuinely fast. The likely explanation is that
`imp.last_latency` (the metric above) captures the *full* cross-process round trip
including the Windows kernel's `WaitForSingleObject` wake/scheduling latency, which is
the same floor regardless of whether the post-wake re-check happens in Python or C++.
R5 removes the Python polling-loop cost; it does not — and was never positioned to —
make the underlying OS wake faster. Recorded honestly: the 10 µs/50 µs target does not
pass on this hardware, but R5 ships anyway because it never regresses (native ≈
doorbell, never worse) and the seam is clean, tested, and fully reversible
(`CUDALINK_WAIT_BACKEND=python` reproduces prior behavior exactly).

> **Why `bench_doorbell.py`, not `bench_r1_wait.py`, hosts this gate**: `bench_r1_wait.py`
> measures `get_frame()` **wall-clock time** (spin + tensor materialization + Python
> overhead) — a different, larger quantity than the publish→detect notification latency
> PLAN-002's gate is defined on. That script's own printout for the native arm now
> explicitly points here instead of asserting a PASS/MISS against the wrong metric.

Reproduce:
```bash
python scripts/profiling/bench_doorbell.py --outfile .profiling/r5_doorbell.json
python scripts/profiling/bench_r1_wait.py --outfile .profiling/r5_wait.json  # informational only, see note above
```

---

## Performance Tuning

See [README.md §Performance Tuning](../README.md#performance-tuning-env-vars) for the
full table of `CUDALINK_*` environment variables and their effect on throughput.

For GPU-timeline profiling (Nsight Systems / Nsight Compute) see [PROFILING.md](PROFILING.md).
