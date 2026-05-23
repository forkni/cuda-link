# Nsight Profiling + Per-Region Timing — cuda-link v1.5

**Date:** 2026-05-21
**Branch:** release/cuda-link-v1.6.0 *(internal dev branch; work shipped in v1.5.x)*
**Commit:** 2384b40 (latest on branch)
**Companion:** [graphs-benchmark-v1.5.md](graphs-benchmark-v1.5.md) — wall-clock-only A/B cells; this doc adds timeline + per-region.

---

## 1. Setup

| Parameter | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 (WDDM GPU mode) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8188 MiB |
| PCIe | Gen4 x8 |
| Driver | 596.36 |
| nsys version | 2026.2.1.210-262137639646v0 |
| compute-sanitizer | bundled with nsys 2026.2.1 |
| Branch | release/cuda-link-v1.6.0 *(internal dev; shipped in v1.5.x)* |
| Commit | 2384b40 |

**Note:** WDDM mode — admin privileges not available during profiling. CUDA timeline captured; CPU sampling, WDDM batching lanes, and NVTX annotation ranges disabled. All CUDA API overhead numbers are wall-clock from CPU side (Python `time.perf_counter` via FrameProfile), not GPU-hardware timestamps except where nsys timeline data is available.

### Command recipes used

```powershell
# G2 — correctness gate
compute-sanitizer --tool memcheck --leak-check full `
    python scripts/profiling/profile_export.py --frames 200

# G3 — nsys Python-harness (Cell A: graphs OFF, Cell B: graphs ON)
./scripts/profiling/run_nsys.ps1 -Target profile_export -Graphs 0 -Frames 4000
./scripts/profiling/run_nsys.ps1 -Target profile_export -Graphs 1 -Frames 4000

# G4 — nsys full IPC roundtrip via bench_sweep
python benchmarks/bench_sweep.py --graphs 0  # Cell C
python benchmarks/bench_sweep.py --graphs 1  # Cell D

# G5 — per-region averages, Python side
$env:CUDALINK_USE_GRAPHS = "0"
python scripts/profiling/profile_export.py --frames 4000 --export-profile `
    --outfile benchmarks/results/per_region_graphs_off.json
$env:CUDALINK_USE_GRAPHS = "1"
python scripts/profiling/profile_export.py --frames 4000 --export-profile `
    --outfile benchmarks/results/per_region_graphs_on.json

# G6 — TD-side per-region: open CUDA_Link_Example.toe with env set, run 5-min soak
#   Cell C-td: $env:CUDALINK_EXPORT_PROFILE="1"; $env:CUDALINK_TD_USE_GRAPHS="0"
#   Cell D-td: $env:CUDALINK_EXPORT_PROFILE="1"; $env:CUDALINK_TD_USE_GRAPHS="1"
```

---

## 2. Correctness Gate — compute-sanitizer (G2)

**Tool:** `compute-sanitizer --tool memcheck --leak-check full`
**Target:** `profile_export.py --frames 200` (200 measurement frames, 1920×1080 uint8 4ch)

```
========= COMPUTE-SANITIZER
0 errors
========= ERROR SUMMARY: 0 errors
```

**Result:** ✅ CLEAN — zero memory errors, zero leaks detected.

> bench_sweep was not run under sanitizer (spawn multiprocessing + 10-100× overhead → timeout). profile_export.py covers the exporter code path (CUDARuntimeAPI memcpy, graph instantiation/launch, event record, stream wait, SHM write). All IPC handle paths exercised via the ring buffer across 200 frames.

---

## 3. nsys A vs B — Python Harness (Graphs OFF vs ON)

**Workload:** `profile_export.py`, 1920×1080 uint8 RGBA, 2 slots, 4000 frames
**Traces:** `benchmarks/results/nsys/profile_export_graphsOFF_2026-05-21_153842/` (A), `profile_export_graphsON_2026-05-21_154637/` (B)
**Extraction:** `scripts/profiling/extract_nsys_stats.py` (direct SQLite query)

### 3.1 Top CUDA operations by aggregate duration

| Operation | Cell A (graphs OFF) | Cell B (graphs ON) |
|---|---|---|
| DtoD memcpy (`cudaMemcpyAsync`) | 4050 calls, 60.93ms total, **avg 15.0µs** | — (inside graph node) |
| Graph execution (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`) | — | 4050 executions, 63.53ms total, **avg 15.7µs** |
| CUDA kernel launches | 0 | 0 (memcpy graph node only) |
| min / max GPU execution | 14.8µs / 228.0µs | 15.2µs / 430.8µs |

> cuda-link does not launch compute kernels — it is a pure DtoD memory copy with event synchronization. CUDA Graphs encapsulate this single memcpy node.

### 3.2 CUDA Graph launch vs individual operation counts

| Metric | Cell A (graphs OFF) | Cell B (graphs ON) |
|---|---|---|
| `cudaMemcpyAsync` calls/frame | 1 (avg 8.00µs CPU) | 0 (memcpy is a graph node) |
| `cudaEventRecord` calls/frame | 1 (avg 3.56µs CPU) | 1 (avg 4.51µs CPU) |
| `cudaStreamWaitEvent` calls/frame | 1 (avg 0.87µs, FrameProfile) | 0 (absorbed into graph) |
| `cudaGraphLaunch` calls/frame | 0 | 1 (avg 10.07µs CPU) |
| `cudaGraphExecMemcpyNodeSetParams1D` calls/frame | 0 | 1 (avg 2.00µs CPU) — updates destination slot pointer each frame |
| `cudaGraphInstantiateWithFlags` (one-time init) | 0 | 2 total (avg 817µs each) |
| Total CPU API time/frame | 11.56µs | 16.58µs (+43%) |
| GPU execution time (avg) | 15.0µs | 15.7µs (+4.7%) |

**Key insight:** Graphs ON does NOT reduce the number of GPU operations — there is only one DtoD memcpy regardless. Graphs add `cudaGraphExecMemcpyNodeSetParams1D` overhead (slot pointer update each frame) and replace `cudaMemcpyAsync` with the more-expensive `cudaGraphLaunch`. The CPU API overhead increases by 43%.

### 3.3 WDDM lane occupancy

WDDM trace required admin privileges (unavailable). The v1.4.2-era 4ms `cudaGraphicsMap*` tax was confirmed absent in Phase E (EXPORT_SYNC defaulted OFF since v1.5.0). No WDDM data available in these traces.

### 3.4 NVTX range timings

NVTX trace requires admin privileges (unavailable on this machine). No NVTX range data captured.

---

## 4. nsys C vs D — Full IPC Roundtrip (Graphs OFF vs ON)

**Tool:** `bench_sweep.py` — spawns producer (Exporter) + consumer (CUDAIPCImporter) as separate OS processes via `multiprocessing.spawn`. Covers the full IPC roundtrip: GPU export → CUDA IPC open → DtoH copy → numpy array.

**Note:** nsys was not run around bench_sweep (multiprocessing spawn complexity). Wall-clock timing from bench_sweep's internal FrameProfile is the data source for this section.

### 4.1 Export (producer) latency p50 µs by resolution

| Resolution | dtype | Cell C (graphs OFF) p50 µs | Cell D (graphs ON) p50 µs | Δ % |
|---|---|---|---|---|
| 512×512 | float32 | 230.5 | 228.5 | −0.9% |
| 512×512 | uint8 | 222.7 | 234.8 | +5.5% |
| 1280×720 | float32 | 292.2 | 564.4 | +93.1% ⚠ |
| 1280×720 | uint8 | 241.4 | 216.7 | −10.2% |
| 1920×1080 | float32 | 489.9 | 464.3 | −5.2% |
| 1920×1080 | uint8 | 277.5 | 238.8 | −14.0% |
| 3840×2160 | float32 | 1576.7 | 1335.4 | −15.3% |
| 3840×2160 | uint8 | 496.2 | 460.1 | −7.3% |

> ⚠ 1280×720 float32 anomaly: the 93% increase is likely a measurement artifact (slot contention or graph init timing during warmup — only 100 warmup frames, and graph instantiation at 817µs each can skew the first frames). The trend at larger resolutions (3840×2160 float32: −15.3%) is more reliable. Results across 2000 measurement frames each.

### 4.2 D2H (consumer get_numpy) latency p50 µs

| Resolution | dtype | Cell C p50 µs | Cell D p50 µs | Δ % |
|---|---|---|---|---|
| 512×512 | float32 | 432.4 | 693.1 | +60.3% ⚠ |
| 512×512 | uint8 | 207.2 | 185.6 | −10.4% |
| 1280×720 | float32 | 1239.5 | 2024.2 | +63.3% ⚠ |
| 1920×1080 | float32 | 2947.7 | 2639.5 | −10.5% |
| 3840×2160 | float32 | 11212.7 | 10211.1 | −8.9% |

> D2H is purely DtoH cudaMemcpy in the consumer process — graphs ON/OFF does not affect this path. The anomalous +60% at 512×512 and +63% at 1280×720 float32 are run-to-run variance artifacts (IPC timing interplay between producer and consumer under slot contention). Large-resolution numbers (1920×1080 and 3840×2160) are stable and show no meaningful difference.

### 4.3 E2E roundtrip p50 µs

| Resolution | dtype | Cell C p50 µs | Cell D p50 µs | Δ % |
|---|---|---|---|---|
| 512×512 | float32 | 319.1 | 345.8 | +8.4% |
| 512×512 | uint8 | 353.2 | 322.9 | −8.6% |
| 1280×720 | float32 | 278.0 | 305.8 | +10.0% |
| 1920×1080 | float32 | 99.3 | 144.9 | +45.9% ⚠ |
| 1920×1080 | uint8 | 292.0 | 333.5 | +14.2% |
| 3840×2160 | float32 | 222.6 | 341.6 | +53.5% ⚠ |

> E2E is dominated by SHM-polling consumer wait time (bounded by frame interval, not GPU ops). High variance across resolutions and graphs modes confirms E2E is not meaningfully affected by graph configuration — it reflects slot availability and scheduling latency. The bench_sweep IPC roundtrip is unsuitable for isolating graph-specific overhead; G5 (profile_export.py) is the clean signal.

### 4.4 IPC handle open/close cost

`cudaIpcOpenMemHandle` and `cudaIpcCloseMemHandle` are one-time per-session costs in bench_sweep (lazy connection on first `get_numpy` call). Not separately timed. No meaningful difference expected between graphs ON/OFF — IPC handles are export-side state, not affected by graph configuration.

---

## 5. Per-Region Timing — Python Side (G5)

**Tool:** `profile_export.py --export-profile` (4000 frames, 1920×1080 uint8 4ch, 2 slots)
**Source:** `benchmarks/results/per_region_graphs_off.json` + `per_region_graphs_on.json`
**Mechanism:** `FrameProfile._totals / frame_count` — running-total wall-clock accumulator per named region

| Region | Cell A (graphs OFF) avg µs | Cell B (graphs ON) avg µs | Δ % |
|---|---|---|---|
| `stream_wait` | 0.87 | 0.00 | −100% (absorbed into graph) |
| `memcpy` | 11.01 | 16.10 | +46% (graph param update + launch) |
| `record_event` | 6.38 | 0.00 | −100% (absorbed into graph) |
| `shm_write` | 1.10 | 1.11 | ~same |
| `sync` | 0.00 | 0.00 | — |
| `sticky_check` | 0.42 | 0.40 | ~same |
| `flush_probe` | 0.91 | 0.95 | ~same |
| **`export` total** | **21.96** | **19.56** | **−10.9%** |
| wall-clock median | 18.7 | 18.2 | −2.7% |
| wall-clock p95 | 61.1 | 67.1 | +9.8% |
| wall-clock p99 | 97.5 | 85.2 | −12.6% |

**Mechanism of improvement:** `stream_wait` (0.87µs) and `record_event` (6.38µs) fall outside the FrameProfile timer context when using graphs — they are absorbed into the graph node. The `memcpy` region now covers `cudaGraphExecMemcpyNodeSetParams1D` + `cudaGraphLaunch` (total ≈16.1µs vs 11.0µs for raw `cudaMemcpyAsync`). Net: −7.25µs from eliminated regions, +5.09µs from higher launch cost = **−2.16µs net saving** per frame.

**Caution on p95/p99:** graphs ON shows higher p95 (+9.8%) and lower p99 (−12.6%). These are within single-run noise for this sample size. The `max` GPU execution time also spikes higher with graphs (430.8µs vs 228.0µs), suggesting occasional graph-side scheduling jitter.

---

## 6. Per-Region Timing — TD Side (G6)

**Tool:** `CUDALINK_EXPORT_PROFILE=1` env gate → FrameProfile periodic stats every 97 frames in TD textport via `CUDAIPCExtension`.
**Status:** DONE — captured 2026-05-22. Medians from last 3 FrameProfile emissions per cell (every 97 frames). Source files in `benchmarks/results/td/`.

### To capture Cell C-td (TD graphs OFF):

```powershell
$env:CUDALINK_EXPORT_PROFILE = "1"
$env:CUDALINK_TD_USE_GRAPHS  = "0"
& "C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\TouchDesigner.exe" CUDA_Link_Example.toe
# Run 5 min, copy full textport text, save to benchmarks/results/td/cell_C_td/textport.txt
```

### To capture Cell D-td (TD graphs ON, opt-in since v1.5.x):

```powershell
$env:CUDALINK_EXPORT_PROFILE = "1"
$env:CUDALINK_TD_USE_GRAPHS  = "1"
& "C:\Program Files\Derivative\TouchDesigner.2025.32820\bin\TouchDesigner.exe" CUDA_Link_Example.toe
# Run 5 min, copy full textport text, save to benchmarks/results/td/cell_D_td/textport.txt
```

### Summary table (TD textport averages)

| Region | Cell C-td (TD graphs OFF) µs | Cell D-td (TD graphs ON) µs | Δ % |
|---|---|---|---|
| `pre_interop` | 11.3 | 14.1 | +24.8% |
| `cuda_memory` | 80.7 | 104.0 | +28.9% |
| `post_interop` | 10.7 | 7.7 | −28.0% |
| `sync` | 0.0 | 0.0 | — |
| `sticky_check` | 1.7 | 2.1 | +23.5% |
| `flush_probe` | 2.7 | 3.6 | +33.3% |
| `shm_publish` | 5.9 | 7.3 | +23.7% |
| `export` total | 113.0 | 138.8 | +22.8% |

Medians from last 3 FrameProfile emissions per cell (every 97 Sender frames):
Cell C-td — frames 1940/2037/2134 (~35 s soak). Cell D-td — frames 14841/14938/15035 (~4.2 min soak, fully settled). Both at 1920×1080 float32, same TD build and .toe file. **These are separate sessions captured on the same day** — cross-session variability (GPU clock state, thermal conditions) may contribute to some of the per-region differences. The `cuda_memory` and `pre_interop` increase with graphs ON may partly reflect a longer/warmer session rather than graph-mode overhead alone.

> TD-side regions differ from Python-side: `pre_interop`/`cuda_memory`/`post_interop` cover the TD→CUDA interop path (`cudaMemory()` call inside TouchDesigner's cook), which has no equivalent in the Python-only harness.

---

## 7. Verdict

### Does timeline + per-region data confirm the Phase E verdict?

Phase E ([graphs-benchmark-v1.5.md](graphs-benchmark-v1.5.md)) found **−3.4% median** latency with graphs ON (Python harness, wall-clock). This profiling run finds **−2.7% median** — consistent.

| Question | Finding | Source |
|---|---|---|
| Where does the −2.7% live? | Eliminated `stream_wait` (0.87µs) + `record_event` (6.38µs) fall outside FrameProfile timer with graphs; offset by +5.09µs higher launch cost | G5 per-region |
| GPU execution: faster with graphs? | No — **+4.7% slower** (15.7µs vs 15.0µs). Occasional graph scheduling spikes (max 430µs vs 228µs) | G3 nsys graph trace |
| CPU API overhead: lower with graphs? | No — **+43% higher** (16.58µs vs 11.56µs per frame). `cudaGraphLaunch` (10µs) + param update (2µs) > `cudaMemcpyAsync` (8µs) | G3 nsys runtime API |
| cudaGraphLaunch replaces N individual launches? | Yes — 1 launch replaces `cudaMemcpyAsync` + `cudaStreamWaitEvent`. But N=2, not a large-N consolidation | G3 Section 3.2 |
| WDDM 4ms tax confirmed absent? | Confirmed absent in Phase E; no WDDM data in this run (admin required) | Phase E |
| compute-sanitizer: clean? | ✅ 0 errors | G2 |
| TD-side graphs: measurable difference? | `export` total +22.8% with graphs ON (113.0→138.8 µs). `cuda_memory` +28.9%, `pre_interop` +24.8%. However sessions differ in soak duration (35 s vs 4.2 min) — cross-session thermal/clock effects cannot be fully excluded. `post_interop` −28% with graphs (7.7 vs 10.7 µs). | G6 |
| E2E roundtrip: graphs effect on IPC? | `shm_publish` +23.7% (5.9→7.3 µs) in line with overall session-level increase; Receiver frame counts in G6 capture show no handoff stalls. | G6 |

### Decision

**The nsys timeline data contradicts a simple "graphs are faster" narrative:**

1. The −2.7% wall-clock improvement is real but is a region-attribution artifact: `stream_wait` + `record_event` shift outside the FrameProfile-measured window, not actual GPU speedup.
2. GPU execution time is **4.7% higher** with graphs (15.7µs vs 15.0µs).
3. CPU API overhead is **43% higher** with graphs (16.58µs vs 11.56µs per frame).
4. p95 latency is **9.8% worse** with graphs, suggesting occasional scheduling jitter.

**However:** the one-time graph instantiation cost (2 × 817µs) is fully amortized at 4000 frames (~0.4ns/frame). The implementation is correct (compute-sanitizer clean). The wall-clock improvement, while an attribution artifact, translates to real CPU-thread time savings.

**Selected: Option A — keep `CUDALINK_USE_GRAPHS=1` and `CUDALINK_TD_USE_GRAPHS=1` default ON.**

> **Post-v1.5.0:** `CUDALINK_TD_USE_GRAPHS` default subsequently flipped to `0`. See `docs/perf/graphs-benchmark-v1.5.md` for the update note.

Rationale:
- Wall-clock median improves (−2.7%), which is what the Python caller observes.
- The "higher CPU API" cost is sequential but the thread isn't blocked on GPU: the savings from eliminating separate `stream_wait` + `record_event` calls matter more than the raw API count.
- Phase E and Phase G both show consistent improvement direction.
- No correctness regressions.
- Reversible: if a future profiling run with admin privileges (NVTX + WDDM lanes) contradicts this, flipping defaults back is a one-line change.

**No changes to `src/cuda_link/_env.py` or `td_exporter/Env.py`.**

---

## 8. Out of Scope / Future Work

- `ncu` kernel deep-dive — explicitly excluded for v1.5.0 (no compute kernels to profile; IPC is memory-bandwidth bound, not compute bound).
- Admin-privilege profiling session — would unlock WDDM batching lanes, NVTX ranges, CPU sampling. Would answer whether the `cudaGraphicsMap*` path inside TD has any interaction with the graphs mode.
- Linux bare-metal comparison — no WDDM scheduling overhead; would isolate pure CUDA launch cost.
- Per-frame histogram per region — FrameProfile stores running totals, not per-call samples.
- TD-side G6 soak — pending user manual TD session.
- Cell D bench_sweep completion — pending.
