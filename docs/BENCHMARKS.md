# cuda-link Benchmarks

All results measured on **RTX 4090 / PCIe 4.0 x16 / Windows 11 / NVIDIA driver 596.36**
unless a section names another GPU.
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

```text
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
prevent a source-buffer lifetime race that caused CUDA 719 under a loaded consumer. From
v2.2.0 the Sender also holds TD's `CUDAMemory` object until the copy has read the block (see
CHANGELOG 2.2.0). The Python library `Exporter` keeps
async as its default (Cell D / Cell B). Coexistence safety comes from explicit per-engine
streams and producer-stream ordering (`record_source_sync` / `require_source_sync`), not from
blocking export.

> **Caveat:** these export timings re-export the same source buffer in a tight loop, so the
> frame stays in L2 (the 4090's 72 MB holds a 1080p float32 frame). See
> [Hot vs cold L2](#hot-vs-cold-l2-rtx-4060-laptop) for how L2 temperature and GPU idle time move
> an export timing on another GPU.

**P3 graph gain (B→D): −4.0 µs p50 (22%)** — `cudaStreamWaitEvent` + `cudaEventRecord`
are folded into the CUDA graph; only `cudaGraphLaunch` fires per frame. nsys confirms:
graphs OFF → 3 submissions/frame (waitEvent + memcpyAsync + eventRecord = 550 calls each);
graphs ON → 1 submission/frame (`cudaGraphLaunch`, 550 calls; waitEvent/eventRecord absent).

### Full P1+P3 (A→D): −31.3 µs p50 (69%)

`per_region_avg_us` breakdown for reference:

```text
Region            Cell A (µs)   Cell D (µs)   Notes
--------------    -----------   -----------   ----------------------------------------
stream_wait            1.26          0.00     folded into CUDA graph (P3)
memcpy                 7.76          6.77     device-to-device (same GPU), HBM-bound, ~1 TB/s class
record_event           4.50          0.00     folded into CUDA graph (P3)
shm_write              1.92          1.21     SHM slot write (unaffected)
sync                  17.91          0.00     cudaStreamSynchronize eliminated (P1)
flush_probe            0.00          1.13     stream query kick (async path only)
```

Note: WDDM submission count/frame is inferred from nsys call counts ÷ frame count.

Reproduce (all four cells; on Windows use `SET VAR=value &` prefix):

```bash
CUDALINK_EXPORT_SYNC=1 python scripts/profiling/profile_export.py --no-graphs --frames 2000 --export-profile --outfile .profiling/exp_A.json
CUDALINK_EXPORT_SYNC=0 python scripts/profiling/profile_export.py --no-graphs --frames 2000 --export-profile --outfile .profiling/exp_B.json
CUDALINK_EXPORT_SYNC=1 python scripts/profiling/profile_export.py --frames 2000 --export-profile --outfile .profiling/exp_C.json
CUDALINK_EXPORT_SYNC=0 python scripts/profiling/profile_export.py --frames 2000 --export-profile --outfile .profiling/exp_D.json
```

---

## `export_frame()` — CUDA Graphs A/B (historical, EXPORT_SYNC=1)

Single-process, EXPORT_SYNC=1 (CPU waits for GPU D2D completion), 2000 frames.
Historical baseline pre-P1/P3 (v1.4.1 era). Updated blocking-arm numbers via
`scripts/profiling/profile_export.py` (v1.10.1, 2026-06-10): 512×512 f32 → **24 µs**,
1080p f32 → **106 µs**, 4K f32 → **357 µs** (see Summary table above; graphs ON,
driver 596.36). Historical table retained for the graphs ON vs OFF comparison.

```text
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

The historical `bench_graphs.py` script used for this table is not included in the repository,
so this comparison is reference data rather than a command that can be run from a fresh checkout.

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

```text
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

```text
Resolution    1 stream p50 (ms)   2 streams p50 (ms)   1 stream GB/s
----------    -----------------   ------------------   -------------
512x512                    0.18                 0.19            22.2
1280x720                   0.61                 0.61            23.1
1920x1080                  1.32                 1.34            23.7
3840x2160                  5.69                 6.82            21.4
```

PCIe 4.0 saturates at ~23–24 GB/s. Single stream is sufficient; `CUDALINK_D2H_STREAMS=1`
(default) is optimal for this platform.

The historical `bench_d2h_streams.py` script used for this table is not included in the repository,
so this comparison is reference data rather than a command that can be run from a fresh checkout.

---

## Full IPC Roundtrip Sweep

Two separate Python processes (producer + consumer), 500 warmup + 2000 measurement frames
at 60 FPS. `export p50` and `get_numpy p50` are inflated vs standalone because both
processes share PCIe bandwidth concurrently. `IPC notify p50` measures
producer-publish → consumer-detects-write_idx (signaling latency, resolution-independent).

```text
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

The historical `bench_sweep.py` script used for this table is not included in the repository, so
these results are reference data rather than commands that can be run from a fresh checkout. For
a tracked runnable benchmark, see [examples/07_python_to_td_exporter_and_benchmark.py](../examples/07_python_to_td_exporter_and_benchmark.py).

### Hot vs cold L2 (RTX 4060 Laptop)

The sweep above exports the same source buffer on every frame and lets the GPU idle between
frames at 60 FPS, so its `export p50` mixes two conditions a real pipeline may not share: a hot
L2 and an idle (possibly downclocked) GPU. A local run of the sweep script on an **RTX 4060 Laptop
(32 MB L2, PCIe 4.0 x8, driver 617.42)** separated them with two options that run a synchronous
device-to-device copy before each timed export (not timed): `--wake` copies 4 KB (GPU awake, L2
untouched) and `--cold` copies a buffer the size of L2 (GPU awake, L2 flushed). Float32 RGBA,
graphs on, 200 warmup + 1000 frames at 60 FPS, one run per cell.

```text
Resolution   Frame     default p50 (µs)   --wake p50 (µs)   --cold p50 (µs)   cold − wake
----------   -------   ----------------   ---------------   ---------------   -----------
512x512        4 MB               234.2             115.5              97.8         −17.7
1920x1080     33 MB               425.6             336.0             349.4         +13.4
3840x2160    133 MB              1298.1            1209.3            1216.0          +6.7
```

- **Cache temperature is not the lever.** `cold − wake` stays within ±5% at every size. The
  512x512 difference is run-to-run noise: repeat runs gave 107–119 µs cold and 108–112 µs with the
  4 KB copy. Only frames that fit in L2 could show a gap, and 1080p float32 (33 MB) is the
  borderline case here; the 133 MB frame is cold either way.
- **Idle time is.** Letting the GPU idle between frames adds about 90–120 µs (`default − wake`).
  This is consistent with the GPU waking from a low-power state; the clocks were not sampled, so
  the mechanism is not confirmed.
- **Reading the tables.** The default numbers include that wake cost, so they are an upper bound
  for a producer that has just run GPU work (StreamDiffusionTD exports right after its diffusion
  step), where `--wake` is the closer reference. The 4090 numbers above were produced in default
  mode and were not re-measured.

---

## TD-in-the-loop: Original (TOX) vs C++ Custom TOP

First head-to-head with TouchDesigner actually in the loop, comparing the pre-existing
pure-Python **Original** path (`CUDAIPCExtension` Sender/Receiver, `.tox`) against the native
**C++ Custom TOP** operators (`CudaLinkOutTOP` / `CudaLinkInTOP`). Measured 2026-07-05,
RTX 4090 / driver 596.36, 1920×1080 RGBA uint8, 60 FPS target, windowed (~97–150-frame)
averages. **TD→TD uses two separate TD instances** (same topology as the C++ TOPs' own
cross-process design), not an in-process loopback.

```text
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

[2] Per the `export_us` line of `td_exporter/example_sender_python.py`, this window
(`export_us = (now - t0) * 1e6`) bundles the synchronous H2D staging fill (`_fill_ctypes`)
*and* `exporter.export()`, excluding the frame-pacing sleep. It is **not** comparable to the
Original's TD-side `export=` line, which only times the extension's own export path.

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

```text
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

**Cross-language legs re-measured the same day (Original implementation):**

- **TD→Python** (1080p uint8, torch mode, doorbell on): `latency=` 0.02–1.33 ms
  publish→detect across the whole run, `get_frame` 81.4 µs avg (min 25.1 / max 359.6,
  n=3697), 59.1 FPS avg, `doorbell_wakes` ≈ 1 per frame. The event-waiting consumer shows
  **no phase aliasing at all** — confirming the pathology is specific to poll-once-per-cook
  receivers, not to the transport.
- **Python→TD** (512×512 uint8, CUDA Graphs ON): `export=` 622–1,830 µs windowed avg,
  58.7 FPS over 3,942 frames. Consistent with the Full IPC Roundtrip Sweep's two-process
  512² numbers; per footnote [2] this window bundles the synchronous ctypes staging fill,
  so it is an example-script figure, not a library `export_frame()` cost.

**Instrumentation footnote:** the In TOP `copy_us` CPU bracket now reads ~107 µs vs ~21 µs
in the 2026-07-05 table. That delta is the Debug-gated `GpuTimerRing` `cudaEventRecord`
enqueues added inside the bracket (~20 µs each under WDDM), not a transport regression —
the Debug-off hot path is unchanged.

**Gap status vs the list above:** Original TD→TD latency at float32 now captured (drifting,
see above). C++ TD→TD end-to-end latency remains open — the planned `frame_age_ms` field
(producer QPC publish stamp already in the slot metadata) would close it.

---

## `.tox` 2.0.0.dev0 (protocol v0.6) vs 1.14.1 (v0.5), live

Date 2026-11-17. Dev box: RTX 4090, driver 621.24, TD 2025.33230 Non-Commercial, two TD processes
(A sends, B receives), 1280x720 rgba8 at 60 fps, Debug off, 60 s per cell. Each version talks
only to itself (v0.6 cannot talk to v0.5). The 2.0 COMPs ran classic mode (mirror DATs). The
1.14.1 COMPs ran library mode with `cuda_link` 1.14.1 from the main checkout
(`CUDALINK_LIB_PATH`). Each cell ran with `CUDALINK_DOORBELL` unset and with `=1`.

Method: A stamps R = `absFrame % 256` into a 4x4 patch; B reads it back per frame (TOP to CHOP)
and counts new, duplicate, gap (skipped stamps) and backward steps. A Python sender stamps byte 0,
which the `.tox` Receiver shows as B (BGRA). "Send" is the CPU time of the Sender COMP's
`execute1.onFrameEnd` (the export). "Import" is the `ImportBuffer` Script TOP CPU cook time.

| Cell | Version | Env | Send p50 / p95 µs | Receiver dup % | gaps | Other |
|---|---|---|---|---|---|---|
| T1 `.tox` → `.tox` | 2.0 | unset | 378 / 620 | 1.03 | 33 | import p50 0.23 ms |
| | 2.0 | unset (run 2) | 427 / 676 | 4.95 | 185 | import p50 0.15 ms |
| | 2.0 | =1 | 426 / 677 | 10.41 | 380 | import p50 0.20 ms |
| | 2.0 | =1 (run 2) | 413 / 477 | 0.08 | 3 | import p50 0.14 ms |
| | 1.14.1 | unset | 524 / 600 | 0.53 | 7 | import p50 0.14 ms |
| | 1.14.1 | =1 | 534 / 662 | 0.06 | 2 | import p50 0.15 ms |
| T2 `.tox` → Python | 2.0 | unset | 451 / 488 | 0 same | 0 | latency p50/p99 0.71 / 1.67 ms, backend native |
| | 2.0 | =1 | 372 / 530 | 0 same | 1 | latency p50/p99 0.048 / 1.64 ms, 3486 wakes |
| | 1.14.1 | unset | 580 / 670 | 0 same | 0 | latency p50/p99 0.78 / 1.71 ms, backend python |
| | 1.14.1 | =1 | 671 / 1057 | 0 same | 0 | latency p50/p99 0.036 / 1.65 ms, 3487 wakes |
| T3 Python → `.tox` | 2.0 | unset (2 runs) | - | 3.00, 3.14 | 102, 110 | sender ≈ 59.8 fps |
| | 2.0 | =1 | - | 3.39 | 80 | sender ≈ 59.3 fps |
| | 1.14.1 | unset (2 runs) | - | 2.83, 2.50 | 18, 8 | sender ≈ 58.6 fps |
| | 1.14.1 | =1 | - | 2.81 | 16 | sender ≈ 58.6 fps |

T3 sender fps = (new + gaps) / 60 s from the stamps. No cell had a backward step or a wrong
stamp. Project errors stayed at 0.

Findings:

- Correctness: `.tox` Sender → `.tox` Receiver, `.tox` → Python and Python → `.tox` work on
  v0.6. Gaps and duplicates come from the two frame clocks, not from lost or reordered frames.
- Sender export cost: not attributed. The 2.0 cells had a lower p50 (372-451 µs against
  524-671 µs), but the send path since `v1.14.1` only adds work (slot meta, `Streamid`, the
  doorbell signal, the `slot_meta` CHOP cook). Both versions show two cost levels (about
  400 µs and about 550-650 µs), and most of the time is `top.cudaMemory()`, not `export()`
  (14-45 µs in the library benchmarks). The versions also ran in different modes and in
  sequence (all 2.0 cells first). Open: a same-mode, interleaved run that times
  `cudaMemory()` and `export()` apart. Closed 2026-11-21: the two versions cost the same; see
  [`.tox` Sender on TD 2025.32820 at 1080p](#tox-sender-on-td-202532820-at-1080p-2026-11-21).
- Import cost: the same in both versions (`ImportBuffer` p50 0.14-0.23 ms).
- TD → TD duplicate %: one 60 s run does not rank versions. 2.0 gave 0.08-10.4 % over four runs,
  with no relation to the env. The `.tox` Receiver polls `write_idx` once per frame start and
  never waits on the doorbell, so `CUDALINK_DOORBELL` does not change it.
- TD → Python latency: a Python consumer with `CUDALINK_DOORBELL=1` gets about 0.04 ms p50
  against about 0.7-0.8 ms with poll-sleep, in both versions. In 2.0 the producer makes the
  event by default, so a consumer with the variable unset gets the native wait backend (1.14.1:
  python). In this run the consumer still had to set `=1` to wait on the event; since the
  later 2.0 change the consumer waits by default.
- Python → TD: the sender rate is a property of the test, not of the library. The example
  sender paces with a relative `time.sleep`, so its rate is 1/60 s plus the Windows sleep
  overshoot; export cost is absorbed. The stamp-derived rate was 59.3-59.9 fps in the 2.0
  runs and 58.6 fps in the 1.14.1 runs (the sender's own last status line read 58.5-58.8 in
  all runs). A 58.6 fps sender gives a duplicate floor of (60 - 58.6) / 60 = 2.3 %, so most
  T3 duplicates come from the pacing. A rate near 60 fps gives more phase gaps (80-110
  against 8-18).
- Install: the 1.14.1 release `.tox` has no mirror DATs. With only `cuda_link` 1.13.0 on TD's
  path it stays inert (Status warning "found but not version 1.14.1"); it needs a matching
  `cuda_link` next to the `.toe` or on `CUDALINK_LIB_PATH`.

### `.tox` Receiver `Framewaitms` (prototype, 2026-11-17)

The `.tox` Receiver got the In TOP's doorbell wait (`TDReceiver.wait_for_new_frame()`): when
`onFrameStart` finds no new frame, it waits on the doorbell in 1 ms slices up to `Framewaitms`
(default 3 ms, 0 = off) and polls `write_idx` after each slice. Measured with the change
loaded into the dev scene, not yet committed. Same setup as above, 2.0 only, doorbell `=1`,
fw0 and fw3 runs alternated. "Sweep": TD B at 59.9 fps, so the phase crosses a full frame
every 10 s (6 cycles per 60 s run); this gives a repeatable TD → TD duplicate number.
"`onFrameStart`" is B's Execute DAT `onFrameStart` time, which holds the wait.

| Case | Framewaitms | Dup % | Latency p50 / p95 ms | `onFrameStart` p50 / p95 ms | Waits that got a frame |
|---|---|---|---|---|---|
| T1 `.tox` → `.tox`, sweep | 3 | 0.17 | 7.2 / 15.2 | 0.22 / 0.47 | 334 |
| | 0 | 1.06 | 8.2 / 15.6 | 0.21 / 0.32 | 0 |
| T1 `.tox` → `.tox`, both 60 fps | 3 | 0.50 | 14.1 / 14.6 | 0.24 / 2.65 | 904 |
| | 0 | 0.81 | 14.2 / 14.9 | 0.22 / 0.42 | 0 |
| T3 Python → `.tox` | 3 | 2.53 | 5.0 / 12.6 | 0.26 / 3.43 | 772 |
| | 0 | 2.23 | 8.6 / 15.9 | 0.22 / 0.35 | 0 |

- Gain: with drifting clocks the wait cuts duplicates about 6x (1.06 % to 0.17 %); at equal
  rates 0.81 % to 0.50 %. T3 latency p50 goes from 8.6 to 5.0 ms. T3 duplicates stay at the
  sender's pacing floor.
- Cost: TD B's main thread waits up to the budget on a frame with no new frame:
  `onFrameStart` p95 2.65-3.43 ms at fw3 against 0.32-0.42 ms at fw0. At equal rates nearly
  every wait got its frame (904 waits, 18 duplicates left). The 10-17 ms maxima also occur at
  fw0 (TD hitches), so they are not from the wait.
- A sender slower than the receiver (a 30 fps camera into 60 fps TD) or a paused sender would
  wait the full budget for nothing. Decided 2026-11-17: a predictive guard in both receivers
  (wait only when the next expected publish is within the budget). Measured with a 30 fps
  Python sender into 60 fps TD, fw3, same frames with and without the guard: `.tox` Receiver
  `onFrameStart` p50 3.12 -> 0.22 ms, In TOP cook p50 3.13 -> 0.12 ms (`docs/CPP_TOP.md`).

### GPU load (2026-11-17)

Two probes in `scripts/profiling/`, run beside a cell:

- `gpu_load_probe.py` (NVML, `nvml` extra): GPU % (share of time a kernel ran), SM clock,
  power, and SM % per process.
- `gpu_engine_probe.ps1`: the Windows "GPU Engine" counters per process. These are the numbers
  Task Manager shows. Its "3D" value is the share of time a process has work on the 3D queue,
  which includes UI drawing and present waits.

Dev box (RTX 4090), both TD instances at 60 fps, 1280×720 rgba8, 30-36 s per cell, NVML:

| Cell | GPU % p50 / p95 / max | TD A SM % p50 | TD B SM % p50 |
|---|---|---|---|
| No link active | 8 / 9 / 9 | 5 | 8 |
| `.tox` Sender → `.tox` Receiver, fw3 | 10 / 12 / 12 | 6 | 10 |
| Out TOP → In TOP (guard), fw3 | 11 / 11 / 12 | 7 | 10 |
| Python 30 fps → In TOP, guard | 9 / 11 / 11 | 5 | 9 |
| Python 30 fps → In TOP, no guard | 9 / 11 / 12 | 5 | 9 |

SM clock 2520 MHz and about 77 W in all cells. The Python sender used 3-5 % SM.

- The link adds at most about 3 points over the idle scene. The guard does not change GPU load:
  it saves TD main-thread time, not GPU time.
- Task Manager showed 40-50 % "GPU" in bursts. The engine counters put this on the TD
  processes' 3D queue, with no link active, while NVML stayed near 10 %. The bursts followed
  use of the TD windows (network editor, viewers) and hit both instances together; turning
  off cooking of an unrelated receiver COMP did not remove them. This is TD UI drawing, not
  cuda-link.
- A sender TD whose Perform window is covered by another window reads 86-97 % "3D" in
  Task Manager. On the dev box both TDs open their Perform window at the same place on the one
  monitor, so TD B's covers TD A's. All three are needed: TD A in Perform mode, TD A sending
  (any sender: `.tox` Sender, a bare `cudaMemory()` call, the Out TOP in either install form),
  and TD A's window covered. Moving TD A's Perform window clear of TD B's: 97 % -> 0.9 %, with
  everything else the same; moving it back: 97 %. NVML stays near 2 %, and TD A's frame rate
  (59.8 fps) and publish rate (about 60/s) do not change. This is queue wait from the covered
  window, not cuda-link work.
- For a benchmark: record each TD's Perform mode, keep the Perform windows from overlapping,
  keep the TD windows unfocused (or the node viewers off) while a cell runs, and read GPU load
  from NVML, not from Task Manager.

## vs CPU SharedMemory

End-to-end at typical resolutions (float32 RGBA), CUDA-Link vs UT_SharedMem-class CPU
SharedMemory baseline (PCIe 4.0):

```text
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

Compiled into the core wheel on Windows (see
[ADR-0012](adr/0012-native-extension-in-core-wheel.md) for the fold from a separate
`cuda-link-native` sidecar into the single wheel). `ImportPolicy.wait_backend` selects
`"auto"` (default) | `"python"` | `"native"`. See
PLAN-002 for the full design and
[ADR-0006](adr/0006-stay-pure-python-no-rust.md) for why this narrow native escape hatch
was taken instead of a broader Rust/C++ rewrite.

Measured 2026-07-04 via `scripts/profiling/bench_doorbell.py` (this script, not
`bench_r1_wait.py`, hosts the accept-gate measurement — see note below), 512×512 RGBA
float32, 300 measurement frames + 40 warmup, single producer/consumer process pair,
30 and 60 fps:

```text
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

## Flag winners (2026-11-20)

Two variables were measured on against off, to keep the winner and remove the variable.
Dev box: RTX 4090, three idle TouchDesigner projects open (GPU load 4-5 %). Python `Exporter` and
`Importer` from `development` (bd1d7e1), `torch` 2.11.0+cu128.

### Exporter cost

`scripts/profiling/profile_export.py --frames 2000 --width 1280 --height 720 --dtype uint8`,
`CUDALINK_EXPORT_SYNC=0`, 3 runs per state, states interleaved. `export_frame` median / p95 in µs.

| Doorbell | Graphs | Run 1 | Run 2 | Run 3 |
|---|---|---|---|---|
| on | on | 16.5 / 23.7 | 13.6 / 22.0 | 13.4 / 20.4 |
| off | on | 13.3 / 23.0 | 13.4 / 23.3 | 12.9 / 23.5 |
| on | off | 23.1 / 28.9 | 23.9 / 33.4 | 22.9 / 26.8 |
| off | off | 22.1 / 43.4 | 23.2 / 28.2 | 22.6 / 25.5 |

- Graphs: on is 9 to 10 µs faster per frame in each run. Graphs on is the winner.
- Doorbell: off is 0.2 to 0.7 µs faster in the median (run 1 of the first row is an outlier),
  which is smaller than the spread between runs of one state. No benefit from off.

### Consumer notify latency

`scripts/profiling/bench_doorbell.py --fps 60 --frames 600`, one run.

| Arm | CPU % | Latency p50 | p95 | p99 | NO_FRAME polls |
|---|---|---|---|---|---|
| poll (doorbell off) | 2.3 | 0.93 ms | 1.60 ms | 1.71 ms | 6448 |
| doorbell | 2.1 | 0.03 ms | 0.04 ms | 0.06 ms | 601 |
| native wait backend | 1.8 | 0.03 ms | 0.04 ms | 0.05 ms | 600 |

- Doorbell on is the winner by a factor of about 30 in notify latency.

### Result

`CUDALINK_DOORBELL` and `CUDALINK_USE_GRAPHS` are removed in 2.0.0; both paths are always on.
For an A/B run, `profile_export.py --no-graphs` selects the legacy export path, and
`bench_doorbell.py` selects polling through `ImportPolicy(doorbell=False)`.

Not measured here: `CUDALINK_TD_USE_GRAPHS` (the `.tox` Sender). It needs two TouchDesigner
processes on a quiet machine. `CUDALINK_CPP_USE_GRAPHS` was measured on 2026-11-17 (no gain, see
ADR-0010 and ADR-0011). `CUDALINK_TD_USE_GRAPHS` was measured on 2026-11-21 (the next section)
and removed in 2.0.0.

## `.tox` Sender on TD 2025.32820 at 1080p (2026-11-21)

Dev box: RTX 4090, driver 621.24, TD 2025.32820 with a Commercial key (no 1280x1280 cap), three
TD processes with empty projects. A: 2.0 `.tox` Sender (`TOXES/CUDAIPCLink_v2.0.0.tox`,
`cuda_link` 2.0.0.dev0 in library mode). B: 2.0 `.tox` Receiver. C: 1.14.1 `.tox` Sender
(`cuda_link` 1.14.1 from tag `v1.14.1` on `CUDALINK_LIB_PATH`). Source: animated Noise TOP,
1920x1080 `rgba8fixed`, 60 fps. Both Senders: blocking export (`CUDALINK_EXPORT_SYNC` unset).

Method: Debug on with `CUDALINK_SENDER_REPORT_EVERY=30`, so each `export=` value is the mean
over 30 frames (0.5 s). p50 and p95 are over those windows, about 120 per run. `export=` times
only `Exporter.export()` in both versions. Each run: extension restarted, Active on, 10 s
warm-up, 60 s measured, Active off. A script in each TD ran the runs from one shared timed plan,
so the conditions alternated without a person between runs.

Two 32820 instances run 1920x1080 at the same time: Sender 60.0 fps, Receiver 58-60 fps,
latency 0.19-0.21 ms, `copy` 93-115 µs.

### `CUDALINK_TD_USE_GRAPHS` on against off

A sends to B; the variable alternated, 3 runs per state, in one pass.

| Graphs | export p50 | export p95 | Run p50s | `cuda_memory` p50 |
|---|---|---|---|---|
| off | 540.4 µs | 725.3 µs | 527.1, 541.7, 640.8 | 111.8 µs |
| on | 541.0 µs | 604.4 µs | 528.8, 540.0, 549.8 | 114.0 µs |

On against off: p50 +0.1 %, inside the 3 % noise band. No gain: the variable was removed in 2.0.0,
and the `.tox` Sender keeps the plain export path. Not shown by the run: the Sender prints no line that says
graphs are in use, so the run shows only that the variable was set before each restart.

### 2.0 against 1.14.1 (Sender only)

Receiver B off; A (2.0) and C (1.14.1) alternated, 3 runs each, in one pass. The pass ran
twice, once with A's TD window in front and once with C's (set by process ID, checked every 5 s).

| Window in front | 2.0 export p50 | 1.14.1 export p50 | 2.0 / 1.14.1 |
|---|---|---|---|
| A (2.0) | 466.9 µs (531.4, 451.6, 446.6) | 460.2 µs (374.6, 488.0, 494.1) | 1.015 |
| C (1.14.1) | 297.4 µs (301.1, 295.7, 295.7) | 299.8 µs (305.9, 282.5, 321.2) | 0.992 |

- 2.0 is within 1.5 % of 1.14.1 in both passes (pass rule: 2.0 p50 <= 1.10 x 1.14.1). In the
  C pass the 2.0 p95 is lower (353.8 against 488.8 µs).
- The window in front sets the level for both versions: about 460 µs with A's window in front,
  about 297 µs with C's. Why is not known; both TDs show the same network. A first pass with
  the window in front not controlled gave a ratio of 1.60 (464 against 290 µs), which is this
  effect, not a version difference. Rule for TD A/B runs: record the window in front and
  compare only inside one pass.
- With `CUDALINK_EXPORT_PROFILE=1`, the exporter's per-region times (µs, mean of 3 runs; the
  window in front was not recorded for this pass):

  | Version | memcpy | record | sync | sticky | shm | other | total |
  |---|---|---|---|---|---|---|---|
  | 1.14.1 | 20.6 | 8.3 | 203.6 | 2.4 | 11.0 | 4.6 | 264.2 |
  | 2.0 | 24.7 | 9.5 | 252.6 | 2.6 | 16.0 | 4.0 | 323.2 |

  `sync` (the host wait for the copy on the IPC stream) is most of the export in both
  versions. `shm` +5 µs is the 2.0 slot meta write. The profiled runs varied from run to run
  (2.0 totals 361, 269 and 340 µs), so the `sync` difference is inside the spread.
- `top.cudaMemory()` is called once per frame by both versions with the same arguments; 2.0
  only times it (`cuda_memory=` p50 106-130 µs). It is not part of `export=`.

This closes the open item of the 2026-11-17 section: in a same-mode, interleaved run that times
`cudaMemory()` and `export()` apart, the 2.0 send path costs the same as 1.14.1.

### Python `Exporter` graphs on against off at 1920x1080

The 2026-11-20 run of the Python `Exporter` was at 1280x720. This run repeats it at 1920x1080:
`scripts/profiling/profile_export.py --frames 2000 --width 1920 --height 1080 --dtype uint8`,
`CUDALINK_EXPORT_SYNC=0`, `--no-graphs` for off, 3 runs per state, states interleaved, with the
three 32820 TD processes above open. `cuda_link` from `development` (d8ff756), `torch`
2.11.0+cu128. `export_frame` median / p95 in µs.

| Graphs | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| on | 13.6 / 36.1 | 13.6 / 36.8 | 13.5 / 37.9 |
| off | 20.1 / 63.2 | 19.2 / 42.7 | 20.0 / 38.3 |

On is about 6 µs (30 %) faster per frame, as at 1280x720. The Python `Exporter` keeps graphs on.
This is a different path from the `.tox` Sender, where `CUDALINK_TD_USE_GRAPHS` (removed in 2.0.0) gave no gain.

### C++ Out TOP against the 1.14.1 `.tox` Sender

One TD process (C, the 1.14.1 TD), C's window in front (109 of 110 checks). A `CudalinkoutTOP`
fed by the same 1920x1080 `rgba8fixed` Noise TOP, on its own stream, and the 1.14.1 `.tox`
Sender alternated, 3 runs each, Active on for one at a time, no Receiver. Each run: 10 s warm-up,
60 s measured, 3600 frames. `.tox`: a `perf_counter` timer around the extension's
`export_frame()`, which is the whole send (format checks, `cudaMemory()`, `export()`), with
Debug on and `CUDALINK_SENDER_REPORT_EVERY=30` as above. Out TOP (installed DLLs, build `852b350-dirty`):
the Info CHOP `cook_us`, `begin_us`, `copy_us` and `end_us` channels, read once per cook, Debug
off. Median / p95 in µs.

| Run | `.tox` `export_frame()` | Out TOP `cook_us` | Out TOP `begin_us` / `copy_us` / `end_us` (median) |
|---|---|---|---|
| 1 | 768.0 / 1078.5 | 126.3 / 166.3 | 75.8 / 30.4 / 7.9 |
| 2 | 649.7 / 1068.0 | 136.1 / 177.7 | 81.9 / 32.6 / 8.4 |
| 3 | 693.6 / 1046.2 | 125.5 / 169.9 | 74.7 / 30.6 / 8.0 |

- The Out TOP uses about one fifth of the `.tox` Sender's time on TD's main thread. Part of the
  difference is the design: the `.tox` waits on the host for the copy (`sync` above), and the
  Out TOP does not wait. `begin_us` (`beginCUDAOperations()`, the interop map) is the Out TOP's
  counterpart of `cudaMemory()`.
- One more Out TOP run with Debug on gave the GPU copy time: `gpu_ipc_us` median 22.5 µs. Debug
  raises `cook_us` to 294 µs median, so the Debug-on cook times are not comparable.
- The 1.14.1 `export=` level in this pass (median 558.6 µs in run 1) is higher than in the
  C-in-front pass above (1.14.1 p50 299.8 µs). With Active off the Out TOP returns before any GPU work
  (`CudaLinkOutTOP.cpp`, the `evalActive()` check in `execute()`), so it did not cause this.
  As above: compare only inside one pass.

### C++ Out/In TOP pair against TD's own sharing ops

Sender in TD C, receiver in TD A (the same TD processes as above; the runs did not record
whether the `.tox` Senders in A and C were off), C's window in front (250 of 250 checks).
Source as above (1920x1080 `rgba8fixed` Noise TOP, 60 fps). One pair ran at a time and the
other pairs were bypassed: 3 runs of 60 s per pair, pairs interleaved, 10 s warm-up. A receiver with no viewer
does not cook by itself, so each receiver got one forced cook at each frame end. Times are TD's
own `cpuCookTime` and `gpuCookTime` per frame (µs) and the Perform CHOP `cpumsec` (ms, whole TD,
median). All pairs ran at 60 fps, with no errors or warnings in the 15 runs (sender and
receiver).

| Pair | Sender CPU cook median (runs) | Sender GPU cook | Sender `cpumsec` | Receiver CPU cook median (runs) | Receiver GPU cook | Receiver `cpumsec` |
|---|---|---|---|---|---|---|
| C++ Out TOP / In TOP | 138.8 (149.1, 135.0, 130.9) | 2195 | 2.77 | 249.1 (291.1, 140.1, 256.8) | 2209 | 2.86 |
| Touch Out/In, HAP Q (default) | 129.8 (115.8, 134.6, 157.5) | 4663 | 2.84 | 113.4 | 11 | 4.53 |
| Touch Out/In, uncompressed | 135.2 (119.1, 154.1, 147.9) | 6519 | 3.00 | 68.8 | 0 | 3.01 |
| Shared Mem Out/In (next frame) | 898.6 | 3015 | 3.63 | 714.2 | 0 | 3.60 |
| Spout Out/In | 78.9 | 10.5 | 2.70 | 58.7 | 11.8 | 2.59 |

- The C++ In TOP waits up to `Framewaitms` (3 ms default) when it cooks and no new frame is
  there (`CudaLinkInTOP.cpp`, `waitForFreshFrame()` called from `execute()`), so its receiver
  cook includes that wait. The built-in receivers show the frame that is there. The next table
  removes the wait.
- TD's GPU cook time for the CUDA ops (about 2.2 ms) is much more than the real GPU copy
  (`gpu_ipc_us` median 22.5 µs, above). It probably includes the interop waits. Do not compare
  GPU cook times across op kinds.
- Perform `msec` (wall time per frame) on the TD in front was 9-16 ms with vsync. It is not a
  cost metric.
- Not measured: latency and a fresh-frame check for the built-in pairs (they carry no frame
  counter).

### C++ In TOP `Framewaitms` 0 against 3

Same setup as the table above, C++ pair only, 3 runs of 30 s per value, values interleaved.
CPU cook in µs.

| `Framewaitms` | Receiver CPU cook median (runs) | Receiver p95 / p99 | Sender CPU cook median |
|---|---|---|---|
| 0 | 159.6 (155.6, 156.6, 163.8) | 253.5 / 1078.2 | 146.3 |
| 3 (default) | 299.1 (270.1, 314.1, 321.6) | 565.1 / 1875.5 | 155.3 |

- The wait adds about 140 µs to each receiver cook at the median.
- With `Framewaitms` 0 the C++ In TOP still costs about 100 µs more than Spout In (58.7 µs) and
  uncompressed Touch In (68.8 µs). These two values come from the first table, a different
  pass, so the 100 µs is a rough guide only. Its cook includes the CUDA copy into its own
  output texture.
- The C++ In TOP cooked once per frame (cooked = 1.000) and also cooked without the forced cook.
  The built-in receivers cooked only on the forced cook.

## `.tox` Sender: process priority class and window in front (2026-11-23)

Question: does the `ABOVE_NORMAL` process priority class remove the window-in-front effect of the
2026-11-21 section?

Dev box: RTX 4090, two TD 2025.32820 processes (A and B) with the same project, each with one
`.tox` Sender on a Noise TOP, 1920x1080 `rgba8fixed`, 60 fps, blocking export, no Receiver. Not
recorded: the `.tox` version and the driver version.

Method: a script in each TD timed the full `export_frame()` call of the Sender with
`perf_counter`, one sample per frame. Thus the values include `top.cudaMemory()` and are not
comparable with the `export=` values of the 2026-11-21 section. Each cell: 5 s warm-up, 25 s
measured (about 1500 frames), both Senders measured in the same cell. A host script set the
priority class of both processes (`SetPriorityClass`), moved one window to the front and read the
window in front 4 times per second. Cell order: normal (A, B, neutral), above normal (A, B,
neutral), and then the same again. "Neutral" is a third application's window.

| Priority class | Window in front | A p50 | A p95 | B p50 | B p95 |
|---|---|---|---|---|---|
| normal | A | 496, 540 | 802, 855 | 475, 670 | 953, 976 |
| normal | B | 489, 629 | 1002, 956 | 709, 532 | 961, 819 |
| normal | neutral | 468 [1], 569 | 987 [1], 1035 | 480 [1], 569 | 847 [1], 952 |
| above normal | A | 507, 512 | 812, 829 | 613, 444 | 955, 956 |
| above normal | B | 471, 505 | 939, 969 | 504, 493 | 825, 829 |
| above normal | neutral | 612, 571 | 1052, 1023 | 581, 552 | 999, 940 |

Values in µs, round 1 and round 2. [1] B's window was in front for about 69 % of this cell, so it
is not a clean neutral cell. Both TDs held 60.0 fps in each cell.

- Priority class: no gain. Mean p50 of the six cells, normal against above normal: A 532 against
  530 µs, B 573 against 531 µs. The same cell repeated gives p50 values up to 195 µs apart
  (B: 475 and 670 µs), so the B difference is inside the spread. There is no operator setting to
  recommend.
- p50 and the window in front: no relation. The 2026-11-21 pattern (the two Senders slow together
  or fast together, by the window in front) did not occur again. The p50 moved between about
  445 and 710 µs with no relation to the window in front or to the priority class.
- p95 and the window in front: the TD with its window in front has a lower p95, about 800-855 µs
  against 940-1050 µs for the same TD when its window is not in front. This is true for 7 of the
  8 non-neutral cells (the exception: B, 961 µs) and for the two priority classes.
- Limit: 25 s cells cannot show a p50 difference smaller than about 150 µs.
