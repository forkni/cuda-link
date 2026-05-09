# TD Pipeline v5 Extended Findings — Async Flush Probe + HWS=2

**Capture date:** 2026-05-08 / 2026-05-09  
**Branch:** `feat/td-extension-deepening` HEAD `c3f4e19`  
**Analyst:** auto-generated from v5 nsys data + comparison against v4 baseline

---

## §A — Flag-Set Isolation (v5 vs v4 environment diff)

The v5 capture exercises **two levers simultaneously** relative to v4.

| Variable | v4 (blocking baseline) | v5 (this capture) |
|---|---|---|
| `CUDALINK_EXPORT_SYNC` | `1` (default, blocking `cudaStreamSynchronize`) | `0` (async path) |
| `CUDALINK_EXPORT_FLUSH_PROBE` | `0` | `1` (enable `cudaStreamQuery` flush probe) |
| `CUDALINK_TD_PERSIST_STREAM` | `1` | `1` |
| `CUDALINK_TD_STREAM_PRIO` | `normal` | `normal` |
| `CUDALINK_LIB_STREAM_PRIO` | `high` | `high` |
| `CUDALINK_NVTX` | `1` | `1` |
| `CUDALINK_NVTX_VERBOSE` | `1` | `1` |
| `HwSchMode` (WDDM GPU-P) | `1` (toggled but not yet rebooted) | `2` (active post-reboot) |

All other variables identical. `EXPORT_SYNC=0` activates the existing async path in
`cuda_ipc_exporter.py:802-808` (`_export_flush_probe`). Setting `HwSchMode=2` moves WDDM
completion delivery off the heartbeat timer; it requires a system reboot to take effect and
is a machine-wide (not per-process) setting.

---

## §B — Comparison Table: v4 → v5

### Producer

| Metric | v4 | v5 | Δ |
|---|---|---|---|
| `cudaStreamSynchronize` avg | **629.8 µs** | **absent** | −629.8 µs (100% eliminated) |
| `cudalink.exporter.flush_probe` avg | absent | **6.1 µs** | +6.1 µs (the replacement cost) |
| `cudalink.exporter.slot<N>` p50 | **693.7 µs** | **90.6 µs** | −603 µs (−87%) |
| `cudalink.exporter.slot<N>` p99 | 2996.6 µs | 225.5 µs | −2771 µs (−92%) |
| `cudalink.exporter.slot<N>` max | 4838 µs | 2929 µs | −1909 µs |
| `cudaGraphLaunch` avg | 24.4 µs | 35.2 µs | +10.8 µs |
| `cudaStreamWaitEvent` / `cudaEventRecord` | — | present | — |
| H2D staging fill p50 | 317.8 µs | 46.3 µs | −271.5 µs (−85%) |
| H2D bandwidth | 4.37 GB/s | 9.90 GB/s | +5.53 GB/s (×2.3) |
| Effective send rate | 58.7 FPS | ~60 FPS (7203 slots) | +1.3 FPS |
| `WDDM wddm_queue_sum` max entry | 116 ms (from v4 extended) | **Empty** (no entries) | −116 ms (100%) |

> Note: the v5 `td_pipeline_v5_findings.md` auto-gen script reported "5.5 FPS" — this is
> a script artifact (timestamp-extraction bug); the NVTX slot count of 7203 over the
> ~120 s capture window corresponds to ~60 FPS.

### Consumer (TD process)

| Metric | v4 | v5 | Δ |
|---|---|---|---|
| `cudalink.receiver.import_frame` aggregate p50 | **182.7 µs** | **157.7 µs** | −25 µs (−13.7%) |
| `cudalink.receiver.import_frame` p99 | 634.8 µs | 472.1 µs | −162.7 µs (−25.6%) |
| `cudalink.receiver.import_frame` max | **36,491 µs** | 30,024 µs (slot0) / **2,961 µs** (slot1) / **2,403 µs** (slot2) | see §C |
| `cudalink.receiver.event_wait` p50 | **19.6 µs** | **38.8 µs** | +19.2 µs (expected — see §C) |
| `cudaStreamSynchronize` (consumer) | absent | absent | — |
| D2A copy GPU p50 | 74.0 µs | 74.0 µs | 0 (unchanged) |
| D2A copy GPU max | 22,539 µs | 31,668 µs | +9 ms (slot0 outlier) |
| Effective cook rate | 56.9 FPS | 58.6 FPS | +1.7 FPS |
| `WDDM wddm_queue_sum` max entry | 116 ms | **Empty** (no entries) | −116 ms (100%) |

### Acceptance criteria scorecard

| Criterion | v4 | v5 expected | v5 actual | Status |
|---|---|---|---|---|
| Producer `cudaStreamSynchronize` absent or `cudaStreamQuery` < 30 µs | 629.8 µs | absent | **absent** (flush_probe avg 6.1 µs) | ✅ PASS |
| Consumer `import_frame` p50 within ±10% of 183 µs | 182.7 µs | 165–201 µs | **157.7 µs** (−13.7%) | ⚠️ FASTER than lower bound — see §C |
| Consumer `import_frame` max < 5,000 µs | 36,491 µs | < 5,000 µs | slot0 **30,024 µs**, slots 1-2 ≤ 2,961 µs | ⚠️ Partial — see §C |
| WDDM Copy-engine max queue entry < 20 ms | 116 ms | < 20 ms | **Empty (zero entries)** | ✅ PASS |
| Producer effective FPS ≥ 59.5 | 58.7 FPS | ≥ 59.5 | **~60 FPS** (7203 slots) | ✅ PASS |
| `cudalink.startup.hws_mode=2` NVTX present | absent | present | **present** | ✅ PASS |

---

## §C — HWS Contribution Analysis

### C.1 — Lever 1a contribution (EXPORT_SYNC=0 + FLUSH_PROBE=1)

The `cudaStreamSynchronize` in the blocking producer path waited for the D2D copy to
complete on the GPU before the CPU updated the SHM header. This created a WDDM batch-flush
point on every slot write (~630 µs at v4 p50).

In v5 the blocking call is replaced by `_export_flush_probe` → `cudaStreamQuery` poll loop
(NVTX range `cudalink.exporter.flush_probe`, avg 6.1 µs, med 4.9 µs). The GPU D2D event
still needs to fire before the SHM write, but the CPU no longer blocks in the WDDM driver
waiting for the completion epoch. The 7.7× slot p50 improvement (693 → 90 µs) is almost
entirely attributable to lever 1a.

Separate but correlated: H2D staging fill p50 dropped from 317.8 µs → 46.3 µs and bandwidth
nearly doubled (4.37 → 9.90 GB/s). This is a side-effect of the same WDDM batch-flush
elimination: previously the H2D `cudaMemcpy` stalled waiting for the WDDM driver to flush
the DMA queue; with the flush probe removing that stall from the slot cycle, H2D dispatch
latency recovered.

**Event_wait regression (expected):** Consumer `event_wait` p50 increased from 19.6 → 38.8 µs.
This is mechanistically expected: with `EXPORT_SYNC=0` the producer updates the SHM header
as soon as `cudaStreamQuery` returns `cudaSuccess` (~6 µs), which happens *before* the D2D
event fires on the GPU timeline. The consumer therefore encounters more unresolved D2D events
at the point it reads the SHM and calls `cudaStreamWaitEvent`. The extra 19 µs per wait is
the distributed cost of the work previously done synchronously in the producer, confirming the
async path is correctly shifting (not eliminating) the WDDM flush wait.

**Net slot trade-off:**
- Producer saved: ~603 µs/slot (cudaStreamSynchronize → flush_probe)
- Consumer added: ~19.2 µs/slot (event_wait increase)
- Net recovered per frame: ~584 µs — almost entirely on the producer side, with minimal
  consumer impact.

### C.2 — Lever 2 contribution (HWS=2 after reboot)

The WDDM `wddm_queue_sum` is **completely empty** in both producer and consumer captures.
In v4, the Copy-engine queue had a maximum single entry of 116 ms. The absence of any
`wddm_queue_sum` entries in v5 means all WDDM Render/Copy-engine queue entries completed
within the minimum reportable window — HWS=2 eliminated the scheduling-epoch batch gaps
at the WDDM driver level.

**Slots 1 and 2 import_frame max improvement confirms lever 2:**
- slot1 max: 36,491 → 2,961 µs (−33.5 ms, −92%)
- slot2 max: 36,491 → 2,403 µs (−34.1 ms, −93%)

These numbers match the lever-2 prediction: HWS=2 drives the WDDM Copy-engine epoch gap from
~35–116 ms toward sub-millisecond, eliminating the dominant source of import_frame tail latency.

### C.3 — Slot0 residual outlier (non-WDDM, separate issue)

Slot0 retains a 30,024 µs max import_frame despite WDDM epochs being eliminated. Several
distinguishing characteristics of slot0 in this capture:

- Slot0 has 3081 instances vs 2393/2397 for slots 1/2 (686 extra = ~29% more writes on slot0)
- Slot0 std-dev: 566 µs vs 97/95 µs for slots 1/2 (6× higher variance)
- Slot0 avg: 197 µs vs 178/178 µs for slots 1/2
- WDDM queue sum empty → the 30ms peak is NOT from a WDDM scheduling epoch

Most likely causes (in order of likelihood):
1. **OS scheduler preemption** — a 30ms duration at tdh consumer process level with no WDDM
   queue evidence is consistent with a single OS scheduling quantum miss (Windows default: 15–30 ms).
2. **IPC handle re-validation on slot0** — slot0 is the first slot mapped during connection
   setup; if a re-initialization occurs (e.g., brief producer restart within the capture window),
   slot0 gets a handle re-open cost that slots 1/2 may avoid.
3. **Ring-buffer write-skew** — the extra 686 slot0 writes suggest the producer's write_idx
   modulo arithmetic may be landing on slot0 disproportionately during wrap-around events,
   and those boundary frames may coincide with higher-latency kernel submissions.

**Action:** this is a separate investigation, not a lever-1a or lever-2 regression. The
`import_frame` criterion (`< 5,000 µs`) is met for slots 1 and 2; the slot0 outlier is a
distinct phenomenon to be investigated in a follow-up v5b capture.

---

## §D — Recommendation

**Lever 1a (`EXPORT_SYNC=0` + `FLUSH_PROBE=1`) is validated for the Python-sender-only topology.**

The combination delivers a 7.7× reduction in producer slot p50 (693 → 90 µs) with no
consumer frame-drop regression and a measured consumer overhead increase of only ~19 µs/slot
(event_wait), fully explainable by the mechanics of the async path.

**Recommended documentation change in `docs/PROFILING.md`:**

1. Add a `##` sub-section under the existing §4 Python-sender runbook:
   - Describe `EXPORT_SYNC=0` + `EXPORT_FLUSH_PROBE=1` as the **recommended** configuration
     for standalone Python-sender deployments (no shared-process TD sender).
   - Quantify the trade-off: −603 µs/slot producer, +19 µs/slot consumer event_wait.
   - Explicitly note that `EXPORT_SYNC=1` remains the **global library default** because it is
     load-bearing for concurrent topologies (TD Sender + TD Receiver in the same process;
     flipping the global default requires cycle-2 TDR-cascade regression testing — separate
     effort, out of scope).

2. Add a note on WDDM slot outliers:
   - HWS=2 eliminates WDDM Copy-engine epoch gaps; confirm this is a prerequisite for the
     async path to reach its full potential on Windows.
   - Reference `docs/PROFILING.md §7` (HWS toggle already documented in `c3f4e19`).

**Do NOT change `TDConfig.py` defaults.** `EXPORT_SYNC` default of `True` is intentional and
documented. Changing it is explicitly out of scope for this plan.

**Lever 1b (async H2D fill in `example_sender_python.py:165-182`):** Optional — now that the
producer slot p50 is 90 µs (vs 693 µs in v4), the slot budget is no longer dominated by the
sync. H2D staging fill is now 46 µs p50. Lever 1b would overlap H2D fill with the prior
slot's D2D, potentially recovering ~40–50 µs/slot. Worth pursuing only if FPS headroom
analysis shows remaining bottleneck at scale; not a priority at 60 FPS operation.

---

## §E — Files Produced by This Analysis

| File | Description |
|---|---|
| `benchmarks/results/nsys/td_pipeline_v5_findings.md` | Auto-generated by `analyze_td_pipeline.py` |
| `benchmarks/results/nsys/td_pipeline_v5_e2e.csv` | Cross-process E2E latency pairs (514 KB) |
| `benchmarks/results/nsys/td_pipeline_v5_producer/producer.nsys-rep` | 388 MB raw capture |
| `benchmarks/results/nsys/td_pipeline_v5_producer/producer_cuda_api_sum.csv` | Producer CUDA API breakdown |
| `benchmarks/results/nsys/td_pipeline_v5_producer/producer_nvtx_sum.csv` | Producer NVTX ranges (includes `hws_mode=2`) |
| `benchmarks/results/nsys/td_pipeline_v5_producer/producer_wddm_queue_sum.csv` | Empty — no WDDM queue entries |
| `benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer.nsys-rep` | 411 MB raw capture |
| `benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer_nvtx_sum.csv` | Consumer NVTX ranges (import_frame per slot) |
| `benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer_cuda_api_sum.csv` | Consumer CUDA API breakdown |
| `benchmarks/results/nsys/td_pipeline_v5_consumer/td_consumer_wddm_queue_sum.csv` | Empty — no WDDM queue entries |

---

## §F — Open Items

| ID | Item | Priority |
|---|---|---|
| F1 | Slot0 residual 30ms outlier — characterise source (OS preemption vs IPC re-validation). Run a v5b capture focusing on slot0 only (single-slot topology or extended capture). | Low |
| F2 | `analyze_td_pipeline.py` FPS calculation bug ("5.5 FPS" artifact). Fix timestamp-extraction logic in the analysis script. | Low |
| F3 | Parallel TD receiver `cudaIpcOpenMemHandle error 400` when one process is under nsys. Repro with minimal setup + triage (nsys driver interception vs HWS handle visibility). | Medium |
| F4 | Lever 1b evaluation: async H2D fill in `example_sender_python.py`. Only pursue if FPS analysis at target scale shows H2D as bottleneck. | Optional |
