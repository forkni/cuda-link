# Performance Improvement Plan

**Date**: 2026-06-09
**Scope**: full codebase — `src/cuda_link/` (canonical library), `td_exporter/` (TD mirror + TD-only engines), TD callback templates.
**Method**: line-level review of every per-frame hot path (`Exporter.export`, `Importer.get_frame*`, `TDSenderEngine.export_frame`, `TDReceiverEngine.import_frame`, `shm_protocol`, `activation_barrier`, `cuda_ipc_wrapper`), cross-checked against the project's own measurements (`docs/BENCHMARKS.md`, `docs/PROFILING.md` §0.5/§7/§8, `.profiling/*.json`) and against official CUDA / PyTorch / CuPy / Python / TouchDesigner documentation (linked per item).

---

## 1. Where the time actually goes

Per-frame cost model at 1080p RGBA float32, from the repo's measured baselines (RTX 4090, PCIe 4.0, Windows 11 WDDM):

| Cost bucket | Measured | Source |
|---|---:|---|
| WDDM blocking sync (`cudaStreamSynchronize`, producer, `EXPORT_SYNC=1`) | ~444–630 µs | PROFILING.md §0.5, §8 (v4 baseline) |
| WDDM API enqueue per submission (`cudaGraphLaunch`, memcpy enqueue, …) | ~30 µs each | PROFILING.md §0.5 |
| GPU D2D copy (actual kernel) | ~0.3–2 µs calculated; 22–367 µs wall incl. sync | PROFILING.md §0.5, BENCHMARKS.md |
| D2H PCIe copy (numpy consumers) | 0.18–5.7 ms (~22–24 GB/s, at bus limit) | BENCHMARKS.md |
| Cross-process IPC notification (publish → consumer detect) | ~136–286 µs | BENCHMARKS.md |
| Python-side per-frame overhead (struct codecs, spin loop, adapters) | ~200 µs aggregate, ≈4 % of frame | ADR-0006 §1 |

Two consequences drive the ranking below:

1. **The dominant fixable cost is WDDM submission/sync behaviour**, not GPU work and not Python. Anything that removes a blocking sync (~600 µs) or a WDDM submission (~30 µs) outranks any Python micro-optimisation. This matches NVIDIA's guidance that WDDM batches command buffers and that `cudaStreamQuery()` can be used to force submission ([CUDA Runtime API – Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html); [NVIDIA forums on WDDM queue flushing](https://forums.developer.nvidia.com/t/concurrent-kernels/28864)).
2. **D2H bandwidth is already at the bus limit** (~23 GB/s on PCIe 4.0 x16), so the only remaining win for numpy consumers is *hiding* the copy, not speeding it up ([CUDA C++ Best Practices Guide — Asynchronous Transfers and Overlapping](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation)).

---

## 2. Ranked opportunities (summary)

| # | Opportunity | Path affected | Expected gain | Effort | Risk |
|---|---|---|---:|---|---|
| P1 | Validate + flip TD Sender to async export (`EXPORT_SYNC=0` + flush probe), or auto-select by topology | TD Sender per frame | **−600 µs p50, −2.8 ms p99** (measured in v5 for the library path) | M | M (needs cycle-2 regression per PROFILING.md §8) |
| P2 | Detect `HwSchMode=0` at init and actively recommend HWS=2 | whole pipeline | ~500 µs producer sync; consumer outlier 36 ms → <5 ms (measured §7) | S | none (advisory only) |
| P3 | Fold `stream_wait_event` + `record_event` into the export CUDA graph (1 WDDM submission instead of 2–3) | Exporter graph path | ~30–60 µs/frame CPU + fewer batch-flush stalls | M | M (IPC-event-in-graph needs HW validation) |
| P4 | Skip redundant `cudaGraphExecMemcpyNodeSetParams1D` when (src, dst) unchanged | Exporter graph path | 1 driver call/frame (~µs, more under WDDM load) | S | low |
| P5 | Double-buffered pinned host buffer: overlap D2H with consumer compute (opt-in) | `get_frame_numpy()` | hides 0.18–5.7 ms/frame behind user code | M | M (adds 1 frame latency, opt-in) |
| P6 | Cache bound methods in `CTypesCUDAAdapter.__getattr__` | every CUDA call, both sides | ~0.1–0.2 µs × 4–8 calls/frame | S | none |
| P7 | Zero-copy `unpack_from` in `CheckerBarrier.read_state` (drop `bytes(buf[:64])`) | Exporter per frame | ~1 µs + 2 allocations/frame | S | none |
| P8 | Reuse per-call objects in `Importer._consume_frame` (backend instances, cached `c_void_p`) | all `get_frame*` | ~0.5–1 µs + allocations/frame | S | low |
| P9 | Gate `_HighResTimer` (winmm) to Python < 3.11; rely on CPython's 100 ns waitable timer on 3.11+ | Importer sleep-phase wait | 2 syscalls/frame + correct sleep floor on 3.11+ | S | low |
| P10 | TD steady-state fast path: cache format-resolution result + status strings | TD Sender + TD Receiver cook loop | ~5–15 µs/frame on TD main thread | S | low |
| P11 | Skip the Receiver's `cook(force=True)` when SHM `write_idx` is unchanged | TD Receiver cook loop | one full Script-TOP cook per idle frame | S | M (cook-context subtleties) |
| P12 | Optional `cuda.bindings` (NVIDIA cuda-python) adapter behind the existing `CudaPort` seam | every CUDA call | part of the ~200 µs Python slice | L | M (optional dep; keep ctypes default) |

S/M/L = small/medium/large. Items P1–P3 dominate; P6–P10 are cheap and additive; P5/P11/P12 are conditional on workload.

---

## 3. Detailed opportunities

### P1 — Bring the measured async-export win to the TD Sender (largest single number)

**Evidence.** `TDSenderConfig.export_sync` defaults to `True` (`td_exporter/TDConfig.py:25,41`), so every TD-produced frame blocks on `cudaStreamSynchronize` inside `Exporter.export` (`src/cuda_link/exporter.py:619-624`). The library exporter already defaults to the async path (`ExportPolicy.export_sync=False`, `src/cuda_link/_exporter_port.py:69,83`), and PROFILING.md §8 measured the switch on the same machine: producer slot p50 **693.7 µs → 90.6 µs (−87 %)**, p99 **2,997 µs → 225.5 µs**, with only +19 µs redistributed to consumer `event_wait`. The remaining blocker is explicitly procedural: *"Switching to async without full cycle-2 regression testing risks subtle TDR-cascade failures. Do not change TDConfig.py defaults"* (PROFILING.md §8).

**Plan.**
1. Run the cycle-2 regression described in PROFILING.md §8 (TD Sender + TD Receiver in one TD instance, repeated reconnect cycles, HWS=2) with `CUDALINK_EXPORT_SYNC=0`, `CUDALINK_EXPORT_FLUSH_PROBE=1`.
2. If it passes: flip the `TDSenderConfig` default. If only the shared-process topology fails: auto-select — keep blocking sync only when a TD Receiver coexists in-process (the extension already knows its mode; a process-local registry of active engines is enough), async otherwise.
3. The fallback already exists in the code: `export()` automatically blocks when a slot has no IPC event (`exporter.py:619`), and the flush probe (`cudaStreamQuery`) is the documented WDDM command-buffer kick ([CUDA Runtime API – cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)).

**Gain.** ~600 µs/frame producer CPU at any resolution (the sync cost is WDDM-bound, not size-bound). At 60 FPS that is ~3.6 % of the TD frame budget returned to the cook loop.

**Validation.** `nsys stats --report cuda_api_sum` must show zero `cudaStreamSynchronize` rows on the producer (PROFILING.md §8 verification step); soak ≥ 1 h with reconnect cycles; watch `flush_probe` average stays ~6 µs.

---

### P2 — Make the HWS (Hardware-accelerated GPU Scheduling) check active instead of passive

**Evidence.** PROFILING.md §7 measures HWS=2 cutting producer `cudaStreamSynchronize` from ~617 µs to ~50–100 µs and consumer outliers from ~36.5 ms to <5 ms. The exporter already *reads* `HwSchMode` (`exporter.py:94-103,237-240`) but only logs it at INFO and emits an NVTX range. Users on default-configured Windows (HWS off) silently leave the largest environmental win on the table; P1's async path explicitly lists HWS=2 as prerequisite #1.

**Plan.** When `_read_hws_mode()` returns `"0"`, emit a one-time `logger.warning` (and TD-side `set_info_status` note) with the §7 toggle instructions. Optionally expose `hws_mode` in `get_stats()` so dashboards surface it.

**Gain.** Zero code-path cost; converts a documented but buried environment lever into an actionable signal. This is also the precondition that makes P1 and P3 reach their measured numbers.

---

### P3 — One WDDM submission per exported frame: capture event-record (and source-sync wait) inside the graph

**Evidence.** The graphs path today is: update memcpy node → `stream_wait_event` (when ordering armed) → `cudaGraphLaunch` → separate `cudaEventRecord` (`exporter.py:566-590`). That is 2–3 WDDM submissions per frame. The infrastructure to do better already exists and is unused: `graph_exec_event_record_node_set_event` / `graph_exec_event_wait_node_set_event` (`src/cuda_link/cuda_graphs.py:275-312`, CUDA 11.4+), and `graph_get_nodes`' docstring already anticipates the 3-node shape `[EventWaitNode, MemcpyNode, EventRecordNode]` (`cuda_graphs.py:146-148`). The mixin's own `graph_launch` docstring states the intent: *"replaces N individual API calls (stream_wait_event, memcpy_async, record_event) with one batched WDDM submission"* (`cuda_graphs.py:120-134`) — but `_build_export_graphs` captures only the memcpy and enforces exactly 1 node (`exporter.py:377-380`).

**Plan.**
1. Bind `cudaEventRecordWithFlags` and capture per slot: `stream_wait_event(ipc_stream, source_sync_event)` → `memcpy_async` → `cudaEventRecordWithFlags(ipc_event[slot], ipc_stream, cudaEventRecordExternal)`. Per the CUDA Programming Guide, the `cudaEventRecordExternal` / `cudaEventWaitExternal` flags create event **nodes** (rather than capture-internal edges) during stream capture ([CUDA Programming Guide — CUDA Graphs / stream capture cross-stream dependencies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html); [Runtime API — Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)).
2. Accept node count 2 or 3 in `_build_export_graphs`; classify nodes by type; keep per-slot graphs so the event-record node is fixed per slot (no per-frame `SetEvent` call needed) and only the memcpy src needs updating (P4 makes even that conditional).
3. Keep the legacy path as automatic fallback (the disable-on-failure machinery at `exporter.py:583-586` already handles this) — important because **interprocess events inside graph capture must be validated on hardware**; the official docs do not explicitly bless `cudaEventInterprocess` events in event-record nodes, so step 0 is a 20-line probe script under `scripts/probe/` that captures, launches, and opens the event from a second process.

**Gain.** 1–2 fewer WDDM submissions/frame (~30 µs each per §0.5), and fewer chances for the deferred-submission accumulation that `flush_probe`/`PERSIST_STREAM` work around. Benefits both Python and TD senders once P1 lands (graphs are currently default-on for the library, off for TD — `TDConfig.py:28`).

**Validation.** nsys timeline: one `cudaGraphLaunch` and zero standalone `cudaEventRecord` per frame; cross-process consumer still sees events fire (soak with the standard IPC roundtrip sweep).

---

### P4 — Don't re-set graph memcpy params when nothing changed

**Evidence.** `export()` calls `graph_exec_memcpy_node_set_params_1d` unconditionally every frame (`exporter.py:570-577`). For Python senders exporting the same tensor every frame, and TD senders in steady state (TD's `cudaMemory()` typically returns a stable allocation between format changes — see the dtype-shrink discussion in `TDSender.py:507-553`), `(src, dst, count)` are identical frame-over-frame. The setter is CPU-only ([cudaGraphExecMemcpyNodeSetParams1D](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)) but still a ctypes→driver round-trip per frame.

**Plan.** Keep `_graph_last_src: list[int | None]` per slot; skip the setter when `gpu_ptr_int` matches. Reset on rebuild/close.

**Gain.** One driver call/frame on the graph path (~1–3 µs via ctypes, more when the driver is contended). Trivial diff, no behaviour change.

---

### P5 — Overlap the D2H copy with consumer compute (opt-in pipelined `get_frame_numpy`)

**Evidence.** `_NumpyBackend.materialize` is fully synchronous: `memcpy_async` immediately followed by `stream_synchronize` (`importer.py:673-702`). At 4K float32 that synchronously burns 5.7 ms of the consumer's frame budget even though the copy engine runs independently. The repo already proved bandwidth is bus-limited and multi-stream splitting does not help (BENCHMARKS.md, D2H stream table) — the remaining lever is *overlap*, the canonical pinned-memory + streams pattern ([CUDA C++ Best Practices Guide — Asynchronous and Overlapping Transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation)).

**Plan.** Add `ImportPolicy.d2h_pipelined: bool = False`. When enabled, `NumpyBuffers` allocates two pinned buffers; `get_frame_numpy()` enqueues the copy for the *current* slot, then synchronizes and returns the *previous* frame's buffer. First call returns `NO_FRAME`. Document the +1 frame latency; keep default off. (The pinned allocation ladder and `needs_rebuild` logic already generalise to N buffers.)

**Gain.** For consumers that do nontrivial CPU work per frame (OpenCV etc.), up to the full D2H time (0.18–5.7 ms) disappears from the critical path. No gain for trivial consumers — hence opt-in.

**Validation.** Extend the D2H benchmark to measure end-to-end consumer loop time with a synthetic 5 ms workload; verify with nsys that copy and CPU work overlap.

---

### P6 — `CTypesCUDAAdapter`: stop paying `__getattr__` on every CUDA call

**Evidence.** The production adapter forwards *every* method through `__getattr__` (`src/cuda_link/_cuda_adapters.py:49-56`). Per the Python data model, `__getattr__` runs only when normal lookup fails — which here is **every call**, since the adapter defines no methods ([Python data model — `object.__getattr__`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__)). The exporter makes 4–6 adapter calls per frame (`memcpy_async`/graph calls, `record_event`, `stream_query`, `check_sticky_error`, …), the importer 2–4. The importer's spin loop already hand-hoists `query = conn.cuda.query_event` (`importer.py:1165`) — evidence the cost is real enough to have been worked around once.

**Plan.** Cache on first miss:

```python
def __getattr__(self, name: str) -> Any:
    attr = getattr(self._api, name)
    object.__setattr__(self, name, attr)  # next lookup hits the instance dict
    return attr
```

**Gain.** ~0.1–0.2 µs per call × every per-frame CUDA call on both sides. Three lines; no interface change (bound methods keep delegating to the same singleton).

---

### P7 — Activation barrier: read the 64-byte segment without copying

**Evidence.** `CheckerBarrier.evaluate()` runs on every `export()` (`exporter.py:495`) and calls `read_state`, which does `_STRUCT.unpack(bytes(shm.buf[:SHM_SIZE]))` (`src/cuda_link/activation_barrier.py:72-74`) — a memoryview slice plus a `bytes` copy plus unpacking all 8 fields including the 32-byte reserved blob, per frame. `struct.unpack_from(buffer, offset)` reads directly from any object supporting the buffer protocol — `SharedMemory.buf` is a memoryview — with no copy ([`struct.unpack_from`](https://docs.python.org/3/library/struct.html#struct.unpack_from); [`multiprocessing.shared_memory`](https://docs.python.org/3/library/multiprocessing.shared_memory.html)).

**Plan.** Add a hot-path codec `_ST_STATE = struct.Struct("<IIIIQI")` (fields up to `barrier_skips`) and use `_ST_STATE.unpack_from(shm.buf, 0)` in `read_state`. The snapshot-vs-tearing comment still holds — `unpack_from` reads the same bytes the slice would. Keep the full-struct path for `increment`/`decrement` (cold).

**Gain.** ~1 µs + two allocations per exported frame. Also applies to `bump_skip` when frames are being skipped.

---

### P8 — Importer: stop allocating per-call objects on the frame path

**Evidence.**
- Every `get_frame*` call constructs a fresh backend object (`importer.py:1277,1288,1309`) — `_TorchBackend(self, stream)` etc. — plus an `ImportResult`.
- `_NumpyBackend.materialize` rebuilds `nb.buffer.ctypes.data_as(ctypes.c_void_p)` each frame (`importer.py:681`), a ctypes object construction; the buffer address is fixed for the life of `NumpyBuffers`.
- `acquire_slot` allocates an `AcquireResult` dataclass per call (`shm_protocol.py:425-459`).

**Plan.** Cache one backend instance per (type, stream) on the Importer (invalidate on close/reinit); precompute `buffer_ptr: c_void_p` in `NumpyBuffers.build`; consider making `AcquireResult` a `NamedTuple` (lighter construction, same field access; it is never mutated).

**Gain.** ~0.5–1 µs and several allocations per frame; reduces GC pressure in long-running consumer loops. Keep `ImportResult` as-is (public API).

---

### P9 — Windows timer hygiene: trust CPython ≥ 3.11, stop toggling winmm per wait

**Evidence.** `_wait_for_slot`'s sleep phase enters `_HighResTimer` per call — a `timeBeginPeriod(1)`/`timeEndPeriod(1)` syscall pair every frame that reaches phase 2 (`importer.py:79-99,1179-1186`). Since CPython 3.11, `time.sleep()` on Windows uses a high-resolution waitable timer (`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`) with 100 ns resolution, independent of the global timer period ([CPython issue #89592 / What's New in 3.11](https://github.com/python/cpython/issues/89592); [`time.sleep` docs](https://docs.python.org/3/library/time.html#time.sleep)). The winmm dance is only needed on 3.9/3.10.

**Plan.** Gate `_winmm` setup on `sys.version_info < (3, 11)`; on 3.11+ make `_HighResTimer` a no-op. Optionally, for ≤3.10, hold the period once for the Importer's lifetime instead of per-wait (the [Microsoft `timeBeginPeriod`](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod) contract is begin/end pairing, not per-sleep scoping).

**Gain.** Two syscalls per slow-path frame removed; on 3.11+ also a *better* sleep floor (100 ns-class timer vs 1 ms `timeBeginPeriod` floor), which tightens `wait_sleep` latency jitter in `get_frame()`.

---

### P10 — TD cook-loop micro-costs: steady-state fast path + cached status strings

**Evidence.** TouchDesigner's own optimization guidance is to minimise per-frame Python work in the cook loop ([Derivative — Optimize](https://docs.derivative.ca/Optimize)). Per frame, `TDSenderEngine.export_frame` currently: reads and string-converts `pixel_format_name`, runs `_resolve_frame_dtype` (dict lookups + arithmetic), the `_PIXEL_FMT_NAME_TO_DTYPE` override block, a 4-way geometry comparison, constructs a `GpuFrame`, and builds a status f-string for `set_info_status` (`td_exporter/TDSender.py:495-686`); `_write_status_par` dedupes the parameter write (`TDHost.py:334-338`) but the string is rebuilt every frame. `TDReceiverEngine.import_frame` likewise calls `td_format_string(self._format)` + f-string per frame (`TDReceiver.py:512-513`).

**Plan.**
- Sender: cache `(pixel_format_name, cm.width, cm.height, cm.size)`; when unchanged from the previous frame, skip the entire dtype/geometry resolution block (the cached `resolved_dtype`/`cm_channels` are still valid) and reuse the previous status string. The format-change path already logs transitions, so correctness is unchanged — changes still take the slow path.
- Receiver: recompute the status string only when `self._format` changes (it is rebuilt on connect/refresh anyway).

**Gain.** ~5–15 µs/frame of Python on the TD main thread (the most budget-constrained thread in the system), plus fewer transient strings per cook.

---

### P11 — Receiver: don't force-cook the Script TOP when there is no new frame

**Evidence.** `callbacks_template.py:onFrameStart` runs `import_buffer.cook(force=True)` every TD frame (`td_exporter/callbacks_template.py:39`); the cook then calls `import_frame`, which reads SHM and returns `False` on `NO_FRAME`. When the producer runs slower than TD (e.g. 30 FPS AI output into a 60 FPS TD project), half the force-cooks do nothing but still pay a full Script-TOP cook. Forced cooking is precisely what Derivative's optimization guide says to minimise ([Derivative — Optimize](https://docs.derivative.ca/Optimize), [Performance Monitor](https://docs.derivative.ca/Performance_Monitor)).

**Plan.** Expose a cheap `has_new_frame()` on the receiver engine — read `write_idx` (`struct.unpack_from(shm.buf, WRITE_IDX_OFFSET)`) and the shutdown flag, compare with `last_write_idx` — and call it in `onFrameStart` before force-cooking. Shutdown/version events must still reach `import_frame`, so cook anyway when shutdown/version differ (the same `acquire_slot` fields make this a ~3-line check).

**Gain.** One Script-TOP cook per idle frame. Zero effect when producer ≥ TD rate; large effect for slow producers. Validate the cook-context edge cases noted in ARCHITECTURE.md (the warning-badge probes show TD lifecycle callbacks are touchy) with the existing verification probes.

---

### P12 — Optional low-overhead CUDA bindings behind the existing `CudaPort` seam

**Evidence.** Every CUDA call goes through hand-rolled ctypes (`cuda_ipc_wrapper.py`), which costs roughly 1–3 µs per call (FFI conversion + errcheck). NVIDIA's official [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/overview.html) package provides Cython-level bindings for the same runtime/driver APIs with substantially lower per-call overhead, and is the vehicle NVIDIA maintains for exactly this use case ([CUDA Python overview](https://developer.nvidia.com/cuda/python)). ADR-0006 caps the value: the whole Python slice is ~200 µs (≈4 % of frame), so this is a tail item — but unlike a Rust rewrite it is *consistent* with ADR-0006's constraints: pure-Python wheel stays the default, `cuda-python` becomes a third optional accelerator exactly like `torch`/`cupy`/`numpy`, and the `CudaPort` Protocol (`_exporter_port.py:163`) is already the seam — a `CudaPythonAdapter` is a drop-in third implementation next to `CTypesCUDAAdapter`/`FakeCUDAAdapter`.

**Plan.** Only after P1–P9 land and if profiling still shows the ctypes seam visible: implement `CudaPythonAdapter` (auto-selected when `cuda.bindings` is importable, env-overridable), reusing the existing port test suite. Do **not** make it required.

**Gain.** Bounded by the ~200 µs Python slice; realistically tens of µs/frame. Lowest priority of the code items.

---

## 4. Examined and deliberately *not* recommended

These were checked during the review; recording them prevents re-litigating (per the ADR convention):

- **Multi-stream D2H** — already measured: no win at ≤1080p, a *regression* at 4K (6.82 vs 5.69 ms). Default `CUDALINK_D2H_STREAMS=1` is correct (BENCHMARKS.md).
- **Rust / native rewrite** — rejected with rationale in ADR-0006; nothing in this review contradicts it. P12 is the consistent alternative.
- **VMM API instead of legacy IPC** — rejected in ADR-0004; the linear-memory workload gets nothing from VMM.
- **Replacing the SHM poll with Windows named-event signalling** — would cut the 136–286 µs notify latency and consumer spin, but adds a second cross-process synchronisation primitive, a new failure mode (event handle lifetime across producer restarts), and protocol-version churn for a latency that ARCHITECTURE.md already classifies as imperceptible at frame rates. Revisit only if a sub-millisecond-latency consumer use case appears.
- **`cudaEventBlockingSync` consumer waits** — `cudaEventSynchronize` has no timeout, which is why the importer polls `cudaEventQuery`; the spin+sleep ladder with a deadline is the right shape for a consumer that must detect producer death. The spin window is already tunable (`CUDALINK_WAIT_SPIN_US`, `ImportPolicy.low_latency()`).
- **Pre-compiled struct codecs, NVTX shim, pointer-attribute cache, `_release_fence`** — all already implemented correctly (`shm_protocol.py:90-93`, `_nvtx.py` zero-cost-when-off, `exporter.py:542-563`, `shm_protocol.py:116-121`); no action.
- **`torch.as_tensor` / CuPy `UnownedMemory` zero-copy views** — built once per connection, correct per the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) and [CuPy interoperability docs](https://docs.cupy.dev/en/stable/user_guide/interoperability.html); the bfloat16 `<u2`-view trick matches the protocol's lack of a bf16 typestr.

---

## 5. Suggested execution order & measurement protocol

1. **Wave 1 (no-risk micro):** P6, P7, P8, P9, P10 — pure Python, each verifiable with the existing no-GPU pytest suite plus `CUDALINK_EXPORT_PROFILE=1` before/after (watch `unacc=` and `shm=` columns drop).
2. **Wave 2 (environment + defaults):** P2, then P1 behind the cycle-2 regression gate from PROFILING.md §8. Measure with `nsys stats --report cuda_api_sum` (producer must lose its `cudaStreamSynchronize` rows) and the IPC roundtrip sweep.
3. **Wave 3 (graph consolidation):** P3 probe script first (IPC event in captured graph, cross-process open), then P4, then the exporter changes. Measure submissions/frame on the nsys timeline.
4. **Wave 4 (conditional):** P5 if numpy consumers with real CPU workloads matter; P11 for slow-producer TD topologies; P12 only if post-Wave-3 profiles still show the ctypes seam.

Every wave: re-run the standard benchmark set (`bench_graphs`-equivalent isolated export, D2H sweep, IPC roundtrip) on the reference machine and update `docs/BENCHMARKS.md`, keeping the §0.5 napkin-math discipline — compare API-call wall time against calculated GPU time before attributing any regression to GPU work.

---

## References

- NVIDIA CUDA Runtime API: [Stream Management (`cudaStreamQuery`, `cudaStreamSynchronize`)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) · [Graph Management (`cudaGraphLaunch`, `cudaGraphExecMemcpyNodeSetParams1D`, `cudaGraphExecEventRecordNodeSetEvent`)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html) · [Event Management (`cudaEventCreateWithFlags`, interprocess/timing flags)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
- NVIDIA CUDA Programming Guide: [CUDA Graphs — stream capture, external event nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- NVIDIA CUDA C++ Best Practices Guide: [Asynchronous transfers & overlap with computation (pinned memory)](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation)
- NVIDIA cuda-python: [`cuda.bindings` overview](https://nvidia.github.io/cuda-python/cuda-bindings/latest/overview.html) · [CUDA Python product page](https://developer.nvidia.com/cuda/python)
- Python: [`struct.unpack_from` (buffer protocol, no copy)](https://docs.python.org/3/library/struct.html#struct.unpack_from) · [`object.__getattr__` data model](https://docs.python.org/3/reference/datamodel.html#object.__getattr__) · [`time.sleep` — 3.11 Windows 100 ns waitable timer](https://docs.python.org/3/library/time.html#time.sleep) · [CPython #89592](https://github.com/python/cpython/issues/89592) · [`multiprocessing.shared_memory`](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
- Microsoft: [`timeBeginPeriod`](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod)
- PyTorch: [`torch.cuda.Stream` / `current_stream`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.current_stream.html)
- CuPy: [Interoperability (`ExternalStream`, `UnownedMemory`, CUDA Array Interface)](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)
- TouchDesigner (Derivative): [Optimize](https://docs.derivative.ca/Optimize) · [Performance Monitor](https://docs.derivative.ca/Performance_Monitor) · [CUDAMemory Class](https://derivative.ca/UserGuide/CUDAMemory_Class)
- In-repo measurements: `docs/BENCHMARKS.md`, `docs/PROFILING.md` (§0.5 WDDM napkin math, §7 HWS, §8 async export A/B), `docs/adr/0004`, `docs/adr/0006`
