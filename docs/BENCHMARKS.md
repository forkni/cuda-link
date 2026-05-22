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
| `export_frame()` — 512×512 RGBA float32 | 22 µs | Standalone, EXPORT_SYNC=1; GPU D2D + stream_synchronize |
| `export_frame()` — 1080p RGBA float32 | 117 µs | Standalone, EXPORT_SYNC=1 |
| `export_frame()` — 4K RGBA float32 | 367 µs | Standalone, EXPORT_SYNC=1 |
| `get_frame_numpy()` D2H — 512×512 float32 | 0.18 ms | Standalone, ~22 GB/s |
| `get_frame_numpy()` D2H — 1080p float32 | 1.32 ms | Standalone, ~24 GB/s PCIe 4.0 |
| `get_frame_numpy()` D2H — 4K float32 | 5.7 ms | Standalone, ~21 GB/s PCIe 4.0 |
| `get_frame()` / `get_frame_cupy()` GPU | <5 µs | Zero-copy tensor/array view, no D2H |
| IPC notification latency | ~136–286 µs | Producer publish → consumer detect (cross-process) |
| Initialization | ~50–100 µs | One-time IPC handle opening |

---

## `export_frame()` — CUDA Graphs A/B

Single-process, EXPORT_SYNC=1 (CPU waits for GPU D2D completion), 2000 frames.

```
Resolution    Graphs off (p50 µs)   Graphs on (p50 µs)
----------    -------------------   ------------------
512x512                      22.4                 19.4
1280x720                     42.7                 41.7
1920x1080                   117.1                115.7
3840x2160                   367.4                366.9
```

With EXPORT_SYNC=1 the GPU D2D copy dominates; CUDA Graphs saves WDDM submission
transitions but the net wall-clock difference is small (<5%). The Graphs path stays on
by default for consistency with async workflows.

Reproduce with:
```bash
python benchmarks/bench_graphs.py --frames 2000 --sizes 512 1280 1920 3840
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

Reproduce with:
```bash
python benchmarks/bench_sweep.py          # full 16-cell sweep (~12 min)
python benchmarks/bench_sweep.py --quick  # smoke test, 1 cell (~1 min)
```

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

## Performance Tuning

See [README.md §Performance Tuning](../README.md#performance-tuning-env-vars) for the
full table of `CUDALINK_*` environment variables and their effect on throughput.

For GPU-timeline profiling (Nsight Systems / Nsight Compute) see [PROFILING.md](PROFILING.md).
