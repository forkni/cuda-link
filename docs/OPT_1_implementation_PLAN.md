     OPT-1: Remove stream_synchronize from Producer Hot Path

     Context

     Problem: cudaStreamSynchronize(ipc_stream) is the dominant cost in export_frame() — it blocks the CPU
     until the D2D memcpy + event record complete on the GPU. Cost: ~13us at 512x512, ~100us at 1080p,
     ~300-500us at 4K. This is 50-80% of export_frame() wall-clock time.

     History: Previously attempted and reverted. The sync exists to guarantee "pre-signaling" — the IPC event
      is already fired by the time write_idx is published to SharedMemory, so the consumer's query_event()
     returns True on the first call.

     Goal: Remove the sync to make export_frame() async (just enqueue + publish), then adapt the consumer to
     handle events that may not yet be signaled.

     Risk: HIGH — affects the synchronization contract between producer and consumer across process
     boundaries on Windows WDDM.

     ---
     Consumer Impact Analysis

     ┌────────────────────────┬──────────────────────────────┬───────────┬──────────────────────────────────┐
     │      Consumer API      │        Sync Mechanism        │ Affected? │              Reason              │
     ├────────────────────────┼──────────────────────────────┼───────────┼──────────────────────────────────┤
     │ get_frame(stream=X)    │ cudaStreamWaitEvent          │ No        │ GPU-side dep works regardless of │
     │                        │                              │           │  event timing                    │
     ├────────────────────────┼──────────────────────────────┼───────────┼──────────────────────────────────┤
     │ get_frame_cupy()       │ CuPy streamWaitEvent         │ No        │ Same GPU-side dep                │
     ├────────────────────────┼──────────────────────────────┼───────────┼──────────────────────────────────┤
     │ TD import_frame()      │ cudaStreamWaitEvent          │ No        │ Same GPU-side dep                │
     ├────────────────────────┼──────────────────────────────┼───────────┼──────────────────────────────────┤
     │ get_frame(stream=None) │ _wait_for_slot (query_event  │ Yes       │ Event may not be signaled on     │
     │                        │ poll)                        │           │ first poll                       │
     ├────────────────────────┼──────────────────────────────┼───────────┼──────────────────────────────────┤
     │ get_frame_numpy()      │ _wait_for_slot (query_event  │ Yes       │ Same — event may not be signaled │
     │                        │ poll)                        │           │                                  │
     └────────────────────────┴──────────────────────────────┴───────────┴──────────────────────────────────┘

     Key insight: Only the query_event polling paths are affected. All stream_wait_event paths are inherently
      correct because the GPU manages the dependency regardless of CPU-side event state.

     Consumer adaptation options (decided by Phase 0 measurement):

     - Option A — cudaEventSynchronize: The wrapper already has wait_event() (line 497 of
     cuda_ipc_wrapper.py) wrapping cudaEventSynchronize. Single blocking call, no CPU burn, no polling loop.
     If WDDM latency is acceptable on cross-process IPC events, this is the cleanest solution.
     - Option B — Hybrid spin-wait: Immediate query_event, then tight spin for ~500us, then fall back to
     sleep-based polling. Burns CPU briefly but avoids Windows time.sleep() granularity (~0.5-1ms minimum).

     ---
     Phase 0: Baseline + WDDM Empirical Measurement

     Purpose: Establish reference numbers AND determine which consumer adaptation to use.

     Step 0.1: Create feature branch

     git checkout development
     git checkout -b feat/opt-1-remove-stream-sync

     Step 0.2: Baseline measurements (NO code changes)

     Run these and record numbers:

     # Poll count verification (current: >=85% first-poll success)
     python benchmarks/_verify_improvements.py

     # E2E round-trip at 512x512 (60fps, 10s)
     python benchmarks/benchmark_roundtrip.py --resolution 512x512 --fps 60 --duration 10

     # E2E round-trip at 1080p (60fps, 10s)
     python benchmarks/benchmark_roundtrip.py --resolution 1080p --fps 60 --duration 10

     # Per-operation breakdown
     python benchmarks/profile_hotpath.py

     # Existing tests
     pytest tests/ -v

     Step 0.3: WDDM latency micro-benchmark

     File: benchmarks/_measure_wddm_event_latency.py (already created, needs timing fix)

     Bug to fix: The script starts producer and consumer simultaneously via p.start(); c.start(). On Windows
     spawn context, the producer subprocess takes ~300ms+ to start and create SharedMemory. The consumer's
     _wait_for_shm() polls but can exhaust retries before the SHM exists, causing CUDAIPCImporter to fail
     with "SharedMemory not found".

     Fix in _producer(): Add a ready signal after initialize() completes:
     exp.initialize()
     ready_q.put("ready")   # signal SHM is created

     Fix in _run_resolution(): Add a ready_q parameter, wait for producer readiness before starting consumer:
     ready_q: Queue = ctx.Queue()
     p = ctx.Process(target=_producer, args=(shm_name, cfg, num_frames, fps, pq, ready_q))
     p.start()
     ready_q.get(timeout=30)  # wait for SHM to be created
     c = ctx.Process(target=consumer_fn, args=(shm_name, cfg, num_frames, cq))
     c.start()

     Remove _wait_for_shm() calls from consumer functions (no longer needed since main process guarantees SHM
      exists before consumer starts).

     What the benchmark measures (3 strategies, each in a separate consumer process):
     1. A: cudaEventSynchronize — single blocking call on IPC event, no CPU burn
     2. B: cudaEventQuery tight spin — no sleep, measures pure GPU completion time
     3. C: cudaStreamWaitEvent + cudaStreamSynchronize — GPU-side wait + CPU block

     Critical: The producer deliberately skips stream_synchronize (raw CUDA ops after initialize()), so the
     IPC event may NOT be signaled when write_idx is published. This tests the exact OPT-1 post-change
     scenario.

     This is the critical experiment: If cudaEventSynchronize latency is < 1ms for 1080p, use Option A. If it
      shows the ~100-300ms WDDM overhead, use Option B.

     Step 0.3 Decision Gate

     ┌──────────────────────────────────────┬─────────────────────────────────────────────────────────┐
     │ cudaEventSynchronize latency (1080p) │                        Decision                         │
     ├──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
     │ < 1ms                                │ Option A — replace _wait_for_slot with wait_event()     │
     ├──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
     │ 1-10ms                               │ Option B — hybrid spin-wait                             │
     ├──────────────────────────────────────┼─────────────────────────────────────────────────────────┤
     │ > 10ms                               │ Option B — hybrid spin-wait (confirms the WDDM concern) │
     └──────────────────────────────────────┴─────────────────────────────────────────────────────────┘

     ---
     Phase 1: Remove Producer stream_synchronize (Python only)

     Step 1.1: Modify export_frame()

     File: src/cuda_link/cuda_ipc_exporter.py, line 449

     Remove self.cuda.stream_synchronize(self.ipc_stream). Update comment.

     Note: As of v0.7.3, _export_frame_fast and _export_frame_debug were merged into a single export_frame()
     method with a `debug = self.debug` local variable. There is only one stream_synchronize call to remove.

     Before:
     if self.ipc_events[slot]:
         self.cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)
     # Synchronize ipc_stream to ensure the D2D copy + event record have EXECUTED on
     # the GPU before publishing write_idx. Without this, the consumer's query_event()
     # may return False (event not yet signaled) even though write_idx is visible...
     # Cost: ~D2D GPU time per frame (~13us at 512x512, ~100us at 1080p).
     self.cuda.stream_synchronize(self.ipc_stream)
     self.write_idx += 1

     After:
     if self.ipc_events[slot]:
         self.cuda.record_event(self.ipc_events[slot], stream=self.ipc_stream)
     # stream_synchronize removed (OPT-1): write_idx is published while GPU work
     # may still be in flight. Consumer uses cudaEventSynchronize/query_event to
     # wait for the IPC event before accessing slot data.
     self.write_idx += 1

     Also remove the stream_synchronize timing from the debug branch (if debug: _t = ... self.total_sync_us +=
     ... lines surrounding the call).

     Step 1.2: Run tests + benchmarks (producer-only change)

     pytest tests/ -v  # All 85 tests must pass
     python benchmarks/_verify_improvements.py  # Poll counts will regress — EXPECTED
     python benchmarks/benchmark_roundtrip.py --resolution 512x512 --fps 60 --duration 10
     python benchmarks/benchmark_roundtrip.py --resolution 1080p --fps 60 --duration 10

     Record:
     - export_frame avg/p50/p95 (expect ~5-15x improvement)
     - get_frame_numpy avg/p50/p95 (expect slight regression from polling)
     - E2E latency (expect improvement if net positive)
     - Poll count avg/max (expect regression from 1 → 2+)
     - Skipped frames (must not increase)

     ---
     Phase 2: Consumer Adaptation

     If Option A (cudaEventSynchronize):

     File: src/cuda_link/cuda_ipc_importer.py, lines 522-555

     Replace _wait_for_slot implementation:

     def _wait_for_slot(self, slot: int) -> float:
         """Wait for producer to finish writing slot.

         Uses cudaEventSynchronize for efficient blocking wait on the IPC event.
         This replaced query_event polling after OPT-1 removed stream_synchronize
         from the producer — the event may not be signaled when write_idx is first
         visible.
         """
         wait_start = time.perf_counter()
         if self.ipc_events[slot]:
             self.cuda.wait_event(self.ipc_events[slot])
         elif TORCH_AVAILABLE:
             torch.cuda.synchronize()
         else:
             self.cuda.synchronize()
         return (time.perf_counter() - wait_start) * 1_000_000

     Note: timeout handling is lost with cudaEventSynchronize (it blocks indefinitely). Add a watchdog: check
      shutdown_flag before calling wait_event, and rely on producer setting the shutdown flag + recording a
     final event on cleanup. If producer crashes without signaling, wait_event blocks forever — this is an
     existing risk with stream_wait_event paths too (get_frame with stream, TD import_frame). Document this
     limitation.

     If Option B (hybrid spin-wait):

     File: src/cuda_link/cuda_ipc_importer.py, lines 522-555

     Replace _wait_for_slot implementation:

     def _wait_for_slot(self, slot: int) -> float:
         """Wait for producer to finish writing slot, with hybrid spin/sleep.

         After OPT-1 removed stream_synchronize from the producer, the IPC event
         may not yet be signaled when write_idx becomes visible. This method uses
         a three-phase wait: immediate check, tight spin (~500us), sleep polling.
         """
         wait_start = time.perf_counter()
         if self.ipc_events[slot]:
             ev = self.ipc_events[slot]
             # Phase 1: immediate check (handles pre-signaled / already-complete)
             if self.cuda.query_event(ev):
                 return (time.perf_counter() - wait_start) * 1_000_000

             # Phase 2: tight spin for up to 500us (covers D2D at most resolutions)
             spin_deadline = wait_start + 0.0005
             while time.perf_counter() < spin_deadline:
                 if self.cuda.query_event(ev):
                     return (time.perf_counter() - wait_start) * 1_000_000

             # Phase 3: sleep-based polling with timeout (4K or unusual delays)
             deadline = wait_start + self.timeout_ms / 1000
             while True:
                 if self.cuda.query_event(ev):
                     break
                 if time.perf_counter() >= deadline:
                     raise TimeoutError(
                         f"IPC event wait timed out after {self.timeout_ms}ms "
                         f"(slot={slot}) -- producer may have crashed"
                     )
                 time.sleep(0.0001)
         elif TORCH_AVAILABLE:
             torch.cuda.synchronize()
         else:
             self.cuda.synchronize()
         return (time.perf_counter() - wait_start) * 1_000_000

     Step 2.1: Update docstring/comments

     Update the comment at importer lines 864-870 to reflect the new contract:
     # CPU-side event wait + async D2H + synchronize.
     # After OPT-1 removed stream_synchronize from the producer, the IPC event
     # may not be pre-signaled when write_idx becomes visible. _wait_for_slot
     # handles this via [cudaEventSynchronize | hybrid spin-wait].

     Step 2.2: Run tests + benchmarks (producer + consumer change)

     pytest tests/ -v  # All 85 tests must pass
     python benchmarks/_verify_improvements.py  # Update expected behavior
     python benchmarks/benchmark_roundtrip.py --resolution 512x512 --fps 60 --duration 10
     python benchmarks/benchmark_roundtrip.py --resolution 1080p --fps 60 --duration 10
     python benchmarks/profile_hotpath.py

     Go/No-Go Gate: Compare against Phase 0 baseline:
     - export_frame must be significantly faster (>2x at 1080p)
     - get_frame_numpy must not regress >10% vs baseline
     - E2E latency must be same or better
     - Skipped frames must not increase
     - All 85 tests pass

     ---
     Phase 3: TD Extension Producer Change

     Step 3.1: Modify export_frame() in TD extension

     File: td_exporter/CUDAIPCExtension.py, line 782

     Remove self.cuda.stream_synchronize(self.ipc_stream). Update comment at lines 778-781.

     Note: The timestamp write at line 785 is currently INSIDE the if self.ipc_events[slot]: block. After
     removing the sync, this should still work — the timestamp is a CPU-side wall-clock write that doesn't
     depend on GPU completion.

     Step 3.2: Tests

     pytest tests/ -v  # All 85 tests pass (TD extension tests use mocks)

     Step 3.3: TD Integration Test (manual)

     Load CUDA_Link_Example.toe in TouchDesigner:
     - Sender mode → verify 60 FPS sustained for 2+ minutes
     - If receiver mode is testable (Python→TD), verify that too

     ---
     Phase 4: Update Verification Infrastructure

     Step 4.1: Update _verify_improvements.py

     The Improvement #2 test currently asserts >=85% first-poll success. After OPT-1:

     If Option A: Poll counts are no longer meaningful (using cudaEventSynchronize not query_event). Either
     skip the test or replace it with a wait_event latency measurement.

     If Option B: Poll counts will be >1 but bounded. Update assertion:
     - Replace pct1 >= 85 (85% first-poll) with latency-based check
     - Measure _wait_for_slot latency instead of poll counts
     - Assert average wait < 500us (covers spin window for most resolutions)

     Step 4.2: Create soak test script

     Create benchmarks/_soak_test.py:
     - Runs benchmark_roundtrip.py equivalent for 60+ seconds at 60 FPS
     - Reports: total frames, skipped frames, p99 latency, max latency
     - Pass criteria: <1% frame skip rate, p99 latency < 10ms
     - Run at 1080p (primary target resolution)

     python benchmarks/_soak_test.py --resolution 1080p --fps 60 --duration 60

     ---
     Phase 5: Final Validation

     Step 5.1: Full benchmark sweep

     ┌─────────────────────────┬────────────┬─────────────┬────────────────────────────────┐
     │        Benchmark        │ Resolution │  Duration   │          Key Metrics           │
     ├─────────────────────────┼────────────┼─────────────┼────────────────────────────────┤
     │ benchmark_roundtrip.py  │ 512x512    │ 10s @ 60fps │ export_us, latency_ms          │
     ├─────────────────────────┼────────────┼─────────────┼────────────────────────────────┤
     │ benchmark_roundtrip.py  │ 1080p      │ 10s @ 60fps │ export_us, latency_ms, skipped │
     ├─────────────────────────┼────────────┼─────────────┼────────────────────────────────┤
     │ benchmark_roundtrip.py  │ 3840x2160  │ 10s @ 30fps │ export_us, latency_ms, skipped │
     ├─────────────────────────┼────────────┼─────────────┼────────────────────────────────┤
     │ _soak_test.py           │ 1080p      │ 60s @ 60fps │ frame skip rate, p99 latency   │
     ├─────────────────────────┼────────────┼─────────────┼────────────────────────────────┤
     │ profile_hotpath.py      │ default    │ default     │ per-operation breakdown        │
     ├─────────────────────────┼────────────┼─────────────┼────────────────────────────────┤
     │ _verify_improvements.py │ 1080p      │ 60 frames   │ poll/wait behavior             │
     └─────────────────────────┴────────────┴─────────────┴────────────────────────────────┘

     Step 5.2: Test suite

     pytest tests/ -v  # All 85 tests pass

     Step 5.3: TD integration (if available)

     - Load CUDA_Link_Example.toe
     - Sender mode: 60 FPS for 2+ minutes
     - Check Textport for errors

     Step 5.4: Final go/no-go

     PASS criteria (all must be met):
     - export_frame avg at 1080p < 30us (was ~100-270us with sync)
     - get_frame_numpy avg at 1080p no worse than 1.1x baseline
     - E2E latency at 1080p same or better than baseline
     - Frame skip rate < 1% at 60fps soak test
     - 85/85 tests pass
     - TD integration at 60 FPS (if testable)

     If FAIL: git checkout development — feature branch stays for future reference.

     Step 5.5: Commit + update docs

     # Update README.md performance claims (remove "includes producer-side stream_synchronize" caveat)
     # Update ARCHITECTURE.md if sync contract is mentioned
     # Update docstrings in exporter/importer
     ./scripts/git/commit_enhanced.sh --no-venv "perf: remove stream_synchronize from producer hot path
     (OPT-1)"

     ---
     Rollback Plan

     At any phase, if results are unsatisfactory:
     git checkout development  # abandon feature branch

     The feature branch feat/opt-1-remove-stream-sync preserves all work for future reference.

     ---
     File Inventory

     ┌───────────────────────────────────────────┬─────────────────────────────────────────────────┬────────┐
     │                   File                    │                     Changes                     │ Phase  │
     ├───────────────────────────────────────────┼─────────────────────────────────────────────────┼────────┤
     │ src/cuda_link/cuda_ipc_exporter.py        │ Remove sync (line 449), update comments         │ Phase  │
     │                                           │ (export_frame merged method as of v0.7.3)       │ 1      │
     ├───────────────────────────────────────────┼─────────────────────────────────────────────────┼────────┤
     │ src/cuda_link/cuda_ipc_importer.py        │ Replace _wait_for_slot (lines 522-555), update  │ Phase  │
     │                                           │ comments (lines 864-870)                        │ 2      │
     ├───────────────────────────────────────────┼─────────────────────────────────────────────────┼────────┤
     │ td_exporter/CUDAIPCExtension.py           │ Remove sync (line 782), update comment (lines   │ Phase  │
     │                                           │ 778-781)                                        │ 3      │
     ├───────────────────────────────────────────┼─────────────────────────────────────────────────┼────────┤
     │ benchmarks/_measure_wddm_event_latency.py │ NEW — WDDM micro-benchmark                      │ Phase  │
     │                                           │                                                 │ 0      │
     ├───────────────────────────────────────────┼─────────────────────────────────────────────────┼────────┤
     │ benchmarks/_verify_improvements.py        │ Update Improvement #2 assertions                │ Phase  │
     │                                           │                                                 │ 4      │
     ├───────────────────────────────────────────┼─────────────────────────────────────────────────┼────────┤
     │ benchmarks/_soak_test.py                  │ NEW — sustained 60fps soak test                 │ Phase  │
     │                                           │                                                 │ 4      │
     ├───────────────────────────────────────────┼─────────────────────────────────────────────────┼────────┤
     │ README.md                                 │ Update performance claims                       │ Phase  │
     │                                           │                                                 │ 5      │
     └───────────────────────────────────────────┴─────────────────────────────────────────────────┴────────┘