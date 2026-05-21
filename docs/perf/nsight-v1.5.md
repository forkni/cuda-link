# Nsight Profiling + Per-Region Timing — cuda-link v1.5

**Date:** 2026-05-21 (profiling run TBD)
**Branch:** release/cuda-link-v1.6.0
**Commit:** 66e933e (post-Phase-E; CHANGELOG consolidated)
**Companion:** [graphs-benchmark-v1.5.md](graphs-benchmark-v1.5.md) — wall-clock-only A/B Cells; this doc adds timeline + per-region.

---

## 1. Setup

| Parameter | Value |
|---|---|
| OS | Windows 11 (WDDM GPU mode) |
| GPU | *(fill in from nvidia-smi output)* |
| Driver | *(fill in)* |
| CUDA | *(fill in)* |
| nsys version | *(fill in: `nsys --version`)* |
| compute-sanitizer | *(fill in: `compute-sanitizer --version`)* |
| Branch | release/cuda-link-v1.6.0 |
| Commit | 66e933e |

### Command recipes

```powershell
# G2 — correctness gate (run first, blocks release on any error)
compute-sanitizer --tool memcheck --leak-check full `
    python scripts/profiling/profile_export.py --frames 500

# G3 — nsys Python-harness cells (A-nsys graphs OFF, B-nsys graphs ON)
./scripts/profiling/run_nsys.ps1 -Target profile_export -Graphs 0 -Frames 4000
./scripts/profiling/run_nsys.ps1 -Target profile_export -Graphs 1 -Frames 4000

# G4 — nsys full IPC roundtrip cells (C-nsys graphs OFF, D-nsys graphs ON)
./scripts/profiling/run_nsys.ps1 -Target bench_sweep -Graphs 0
./scripts/profiling/run_nsys.ps1 -Target bench_sweep -Graphs 1

# G5 — per-region averages, Python side (replay cells A/B with FrameProfile)
$env:CUDALINK_USE_GRAPHS = "0"
python scripts/profiling/profile_export.py --frames 4000 --export-profile `
    --outfile benchmarks/results/per_region_graphs_off.json
$env:CUDALINK_USE_GRAPHS = "1"
python scripts/profiling/profile_export.py --frames 4000 --export-profile `
    --outfile benchmarks/results/per_region_graphs_on.json

# G6 — TD-side per-region: open CUDA_Link_Example.toe with env set, run 5-min soak
#       $env:CUDALINK_EXPORT_PROFILE = "1"; $env:CUDALINK_TD_USE_GRAPHS = "0"  → Cell C-td
#       $env:CUDALINK_EXPORT_PROFILE = "1"; $env:CUDALINK_TD_USE_GRAPHS = "1"  → Cell D-td
#       Copy textport scroll to benchmarks/results/td/cell_C/ and cell_D/
```

---

## 2. Correctness Gate — compute-sanitizer

**Toolchain:** `compute-sanitizer --tool memcheck --leak-check full`

**Target 1:** `profile_export.py --frames 500`

```
[PENDING — paste compute-sanitizer output here]
```

**Target 2 (if bench_sweep restored and available):** `bench_sweep.py --quick`

```
[PENDING — paste compute-sanitizer output here]
```

**Result:** ✅ CLEAN / ❌ FINDINGS (list findings here if any)

---

## 3. nsys A vs B — Python Harness (Graphs OFF vs ON)

**Cell A-nsys:** `CUDALINK_USE_GRAPHS=0`, `profile_export.py`, 4000 frames
**Cell B-nsys:** `CUDALINK_USE_GRAPHS=1`, `profile_export.py`, 4000 frames

### 3.1 Top 5 CUDA kernels by aggregate duration

| Kernel name | Cell A (graphs OFF) total ms | Cell B (graphs ON) total ms | Δ % |
|---|---|---|---|
| *(from nsys stats or GUI)* | | | |
| | | | |
| | | | |
| | | | |
| | | | |

> Note: with graphs ON, individual kernels are replaced by `cudaGraphLaunch`. Report the graph launch aggregate vs. the sum of constituent kernels in Cell A.

### 3.2 CUDA Graph launch vs individual kernel-launch counts

| Metric | Cell A (OFF) | Cell B (ON) |
|---|---|---|
| Total kernel launches | | |
| cudaGraphLaunch calls | 0 | |
| cudaMemcpyAsync calls | | |
| WDDM batch submissions | | |

> Cell B should show 1 `cudaGraphLaunch` per frame instead of N individual launches.

### 3.3 WDDM lane occupancy

| Metric | Cell A (OFF) | Cell B (ON) |
|---|---|---|
| WDDM packets / frame (avg) | | |
| Max WDDM gap (µs) | | |
| `cudaGraphicsMap*` 4ms tax seen? | | |

> The v1.4.2-era 4 ms `cudaGraphicsMap*` tax should be absent (EXPORT_SYNC flipped to default-OFF in v1.5.0).

### 3.4 NVTX range timings (from nsys GUI — NVTX track)

| NVTX range | Cell A (OFF) avg µs | Cell B (ON) avg µs | Δ % |
|---|---|---|---|
| `cudalink.exporter.export` | | | |
| `cudalink.exporter.memcpy` | | | |
| `cudalink.exporter.record_event` | | | |
| `cudalink.exporter.shm_write` (est.) | | | |

---

## 4. nsys C vs D — Full IPC Roundtrip (Graphs OFF vs ON)

**Cell C-nsys:** `CUDALINK_USE_GRAPHS=0`, `bench_sweep.py --quick`
**Cell D-nsys:** `CUDALINK_USE_GRAPHS=1`, `bench_sweep.py --quick`

*Note: bench_sweep spawns producer + consumer as multiprocessing.spawn child processes. nsys profiles the orchestrator and its children.*

### 4.1 End-to-end roundtrip latency (µs)

| Metric | Cell C (OFF) | Cell D (ON) | Δ % |
|---|---|---|---|
| Export (producer) p50 µs | | | |
| D2H / get_numpy p50 µs | | | |
| E2E p50 µs | | | |
| E2E p95 µs | | | |

### 4.2 IPC handle open/close cost

| Metric | Cell C (OFF) | Cell D (ON) |
|---|---|---|
| `cudaIpcOpenMemHandle` avg µs | | |
| `cudaIpcCloseMemHandle` avg µs | | |
| One-time init cost vs steady-state visible? | | |

### 4.3 Cross-process sync stalls

| Metric | Cell C (OFF) | Cell D (ON) |
|---|---|---|
| `cudaStreamWaitEvent` avg µs | | |
| SHM polling gaps visible in NVTX? | | |
| Consumer `cudaIpcOpenMemHandle` stall | | |

---

## 5. Per-Region Timing — Python Side (G5)

**Tool:** `profile_export.py --export-profile` (4000 frames, via `FrameProfile._totals / frame_count`)
**Source:** `benchmarks/results/per_region_graphs_off.json` + `per_region_graphs_on.json`

| Region | Cell A (OFF) avg µs | Cell B (ON) avg µs | Δ % |
|---|---|---|---|
| `memcpy` | | | |
| `stream_wait` | | | |
| `record_event` | | | |
| `sync` | | | |
| `sticky_check` | | | |
| `flush_probe` | | | |
| `shm_write` | | | |
| `export` (total) | | | |
| `ptr_cache_miss` (count/frame) | | | n/a |

> Hypothesis from [graphs-benchmark-v1.5.md]: the −3.4 % median saving lives in the CUDA kernel launch path (memcpy/record_event consolidation), not the SHM write.

---

## 6. Per-Region Timing — TD Side (G6)

**Tool:** `CUDALINK_EXPORT_PROFILE=1` env gate in `TDSenderConfig.from_env()` → FrameProfile periodic stats every 97 frames in TD textport.
**Source:** `benchmarks/results/td/cell_C/` and `benchmarks/results/td/cell_D/`

### Cell C-td — `CUDALINK_TD_USE_GRAPHS=0`

```
[PENDING — paste 97-frame stats lines from TD textport here]
```

### Cell D-td — `CUDALINK_TD_USE_GRAPHS=1` (default)

```
[PENDING — paste 97-frame stats lines from TD textport here]
```

### Summary table (extracted from textport averages)

| Region | Cell C-td (TD OFF) avg µs | Cell D-td (TD ON) avg µs | Δ % |
|---|---|---|---|
| `pre_interop` | | | |
| `cuda_memory` | | | |
| `post_interop` | | | |
| `sync` | | | |
| `sticky_check` | | | |
| `flush_probe` | | | |
| `shm_publish` | | | |
| `export` (total) | | | |

> TD-side regions differ slightly from Python-side: `pre_interop`/`cuda_memory`/`post_interop` cover the TD→CUDA interop path (`cudaMemory()` call), which has no equivalent in the Python-only harness.

---

## 7. Verdict

### Does timeline + per-region data confirm the Phase E verdict?

Phase E ([graphs-benchmark-v1.5.md](graphs-benchmark-v1.5.md)) found **−3.4 % median** latency with graphs ON (Python harness, wall-clock), within the ±2 % noise band at tails. Decision: keep defaults ON.

| Question | Finding | Source |
|---|---|---|
| Where does the −3.4 % live? | *(memcpy? launch consolidation? SHM?)* | G5 per-region table |
| WDDM 4ms tax confirmed absent? | | G3 WDDM lane |
| cudaGraphLaunch replaces N individual launches? | | G3 Section 3.2 |
| TD-side graphs: any measurable difference? | | G6 table |
| E2E roundtrip: IPC overhead dominated by open/close or steady-state? | | G4 Section 4.2 |
| compute-sanitizer: clean? | | Section 2 |

### Decision

**Option A (data confirms): keep `CUDALINK_USE_GRAPHS=1` and `CUDALINK_TD_USE_GRAPHS=1` default ON.**
- No changes to `src/cuda_link/_env.py` or `td_exporter/Env.py`.

**Option B (data contradicts): flip defaults OFF.**
- Edit `src/cuda_link/_env.py`: `CUDALINK_USE_GRAPHS` default → `"0"`.
- Run `python scripts/sync_td_wrapper.py` to regenerate `td_exporter/Env.py`.
- Update [graphs-benchmark-v1.5.md](graphs-benchmark-v1.5.md) with a cross-link.
- Commit before Phase F merge.

**Selected:** *(fill in after running profiling)*

---

## 8. Out of Scope / Future Work

- `ncu` kernel deep-dive (explicitly excluded by user for v1.5.0 — not enough signal vs. nsys at WDDM level).
- Linux bare-metal CUDA comparison (no WDDM batching constraint — would isolate pure CUDA launch cost from OS scheduling jitter).
- Per-frame histogram per region (FrameProfile stores running totals, not per-call samples; requires a higher-overhead harness).
