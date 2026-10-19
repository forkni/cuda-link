# Block-by-block dual code review — 2026-10-19

Target: `development` @ `f3d6395` (working-tree changes were `.claude/` only and excluded).
Reviewers, run independently per block, findings only (no fixes, nothing posted):

- **Claude** — built-in `/code-review` at `medium` (recall mode: 8 angles, dedup, no verify pass; every item was checked against callers/tests before reporting).
- **agy** — Gemini `gemini-4-argon-medium` via the `agy-pr-review` skill's `code-review` prompt, adapted to whole-block input (full source per block instead of a PR diff). All 8 runs returned `status: SUCCESS` on the first attempt.

Verdict legend: **CONFIRMED** = the cited lines were read and the failure scenario holds; **PLAUSIBLE** = mechanism is real but the trigger depends on configuration or a path not exercised here; **DROPPED** = checked and found not to be a defect.
`td_exporter/*.py` mirrors of `src/cuda_link` (sync-generated) were excluded; cold block B9 (`_env`, `nvml_observer`, `_nvtx`, `_profile`, `_console`, `cuda_ipc_importer`) was skipped by decision.

## 1. Functional blocks and hot paths

| ID | Block | Files | Hot? |
| --- | --- | --- | --- |
| B1 | SHM protocol + activation barrier | `src/cuda_link/shm_protocol.py`, `activation_barrier.py` | hot (`publish_frame`, `acquire_slot`) |
| B2 | CUDA runtime wrapper + adapters | `cuda_ipc_wrapper.py`, `cuda_runtime_types.py`, `cuda_graphs.py`, `_cuda_adapters.py` | hot (ipc/event/memcpy calls) |
| B3 | Exporter (producer) | `exporter.py`, `_exporter_port.py`, `_doorbell.py` | hot (`Exporter.export`) |
| B4 | Importer (consumer) | `importer.py`, `_importer_port.py`, `_wait_backend.py` | hot (`get_frame` → `_wait_for_slot` → `_consume_frame`) |
| B5 | Native waiter | `_cpp/native_waiter.cpp`, `_native_loader.py` | hot (`wait_slot_impl`) |
| B6 | TD facade + host + config + bootstrap | `td_exporter/CUDAIPCExtension.py`, `TDHost.py`, `TDConfig.py`, `CUDALinkBootstrap.py` | cold |
| B7 | TD Sender | `td_exporter/TDSender.py` | hot (`TDSenderEngine.export_frame`) |
| B8 | TD Receiver | `td_exporter/TDReceiver.py` | hot (`TDReceiverEngine.import_frame`) |

Hot paths (verified with code-search + grep):

- **Send**: `TDSender.export_frame` (TDSender.py:510) → `Exporter.export` (exporter.py:584-764) → `shm_protocol.acquire_slot` (458) / `publish_frame` (398) → doorbell signal (exporter.py:721/798).
- **Receive (Python)**: `Importer.get_frame` (importer.py:1718) → `_wait_for_slot` (1519) → `WaitBackend.wait_slot` (_wait_backend.py:54) / `native_waiter.cpp:wait_slot_impl` (147-219) → `_consume_frame` (1653).
- **Receive (TD)**: `TDReceiver.import_frame` (TDReceiver.py:386) → `acquire_slot` → event wait → `TOPHandle.copy_cuda_memory` (TDReceiver.py:505/534 → TDHost.py:80).

## 2. Ranked cross-block defects

Ranked by hot-path impact × blast radius. "Both" = found independently by both reviewers.

| # | Finding | Location | Found by | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Recovered non-sticky CUDA errors become permanently fatal: `check_sticky_error` raises on any latched `cudaPeekAtLastError`, `cudaGetLastError` never bound; a failed graph launch that `export` already fell back from poisons every later frame | `cuda_ipc_wrapper.py:689`, `exporter.py:704` | Claude | CONFIRMED |
| 2 | CUDA errors inside the native waiter are reported as `TIMEOUT`; deadline is checked before the first event query, so `timeout_ms == 0` (and `int(0.5)`) never polls | `native_waiter.cpp:231`, `:192-195`, `importer.py:1551` | Both (agy B4#6, B5#1-2; Claude B4#7, B5#1-2) | CONFIRMED |
| 3 | TDSender reopen: `Exporter.open` on a geometry/dtype change is unguarded; a failure leaves a closed exporter with `_initialized=True` and every later cook raises | `TDSender.py:745` | Both | CONFIRMED |
| 4 | Importer `_reinitialize` keeps stale `_last_write_idx` and bypasses `_resolve_format`; the restarted producer's first frame is dropped and `ImportSpec` overrides are discarded. Same bug in TDReceiver `_refresh_on_version_change` | `importer.py:1861`, `TDReceiver.py:1206` | agy (B4#3, B8#1) | CONFIRMED |
| 5 | TDReceiver `_refresh_on_version_change` reopens handles destructively: all old handles closed first; a mid-loop failure double-closes via `cleanup()` and leaks the new ones | `TDReceiver.py:1148-1173` | Both | CONFIRMED |
| 6 | TDReceiver refresh never rebuilds the float16 staging buffers / CuPy views; float32→float16 switch skips every frame, float16 handle change reads closed pointers | `TDReceiver.py:1186` | agy (B8#2) | CONFIRMED |
| 7 | Importer `_apply_format_change` never rebuilds `self._cupy`; after `_reinitialize` CuPy arrays wrap unmapped pointers | `importer.py:1829` | agy (B4#2) | CONFIRMED |
| 8 | `_connect` builds Torch/CuPy buffers after the rollback block: a build failure leaks the whole `IPCConnection` on every reconnect retry | `importer.py:1148` | Both (Claude B4#1 open leak; agy B4#4) | CONFIRMED (bfloat16-under-CuPy trigger PLAUSIBLE) |
| 9 | Activation barrier: full-struct RMW in `bump_skip`/`increment`/`decrement` races; `HolderBarrier.acquire` has no held guard and TDSender re-runs `initialize()` every failing cook → stacked increments, exporters paused until the 5 s stale timeout | `activation_barrier.py:128`, `:337`, `TDSender.py:440/:504` | Both (Claude B1#1-2, B7#2; agy B7#2) | CONFIRMED |
| 10 | Consumer timed-out frame is consumed without retry (`_last_write_idx` advanced before the copy) | `importer.py:1675`, `TDReceiver.py:451` | Claude | CONFIRMED |
| 11 | TDReceiver event-less fallback synchronizes the receiver's own stream, giving no cross-process ordering → silent torn frames | `TDReceiver.py:473` | Claude | CONFIRMED |
| 12 | Reopen/refresh conditions ignore `extra_flags`/`flags`: RG ↔ Mono+Alpha switch at the same dims is invisible to both sender and receiver | `TDSender.py:721`, `TDReceiver.py:1209` | Both | CONFIRMED |
| 13 | Exporter: SHM segment reused without size check (→ `IndexError` in `set_shutdown`); version/shutdown written before metadata with no fence | `exporter.py:328`, `:334` | Both | CONFIRMED |
| 14 | Exporter caches an invalid `gpu_ptr` and sticks the error when `strict_device=False`; `int(frame.ptr)` TypeError escapes; ptr cache unbounded | `exporter.py:614-635` | Both | CONFIRMED |
| 15 | `_native_loader` sets the global function pointer before validation; a bad load breaks the live backend. Unknown status → `KeyError` | `_native_loader.py:60`, `:112` | Both / Claude | CONFIRMED |
| 16 | NVTX range leaked on non-timeout exceptions in `_consume_frame` | `importer.py:1669` | Both | CONFIRMED |
| 17 | `wait_for_doorbell` blocks even with `policy.doorbell=False` because the handle is force-opened for the native backend on Windows | `importer.py:1637`, `:1293` | agy (B4#7) | PLAUSIBLE (handle opened at 1293 when `want_native`; 1306 adds a guard whose effect was not traced) |
| 18 | `DtypeCodec.decode` maps unknown dtypes to float32; version change invisible while `last_version == 0`; `write_idx` uint32 overflow raises `struct.error` from `publish` | `shm_protocol.py:287`, `:483`, `:410` | Claude | CONFIRMED (overflow ~207 days at 240 fps) |
| 19 | Barrier/SHM magic and version never validated by readers | `activation_barrier.py:83/90`, `shm_protocol.py:90` | Both | CONFIRMED |
| 20 | TD status tint/script errors leak across mode switches (`TDReceiverEngine.cleanup` never calls `clear_status`); `set_warning_status` never clears prior script errors; status setters write COMP storage on every cook | `CUDAIPCExtension.py:526`, `TDHost.py:377` | agy (B6#3-4) / Claude (B6#1) | CONFIRMED |

## 3. Per-block results

### B1 — SHM protocol + activation barrier (Claude 10, agy 2)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| Barrier magic/version written but never validated (`activation_barrier.py:83/90`) | #8 | #1 | CONFIRMED |
| `bump_skip` full-struct RMW races the Sender decrement (:128) | #1 | – | CONFIRMED |
| `HolderBarrier.acquire` re-increments when already held (:337) | #2 | – (agy reported the consumer side as B7#2) | CONFIRMED |
| `DtypeCodec.decode` silently maps unknown dtypes to float32 (`shm_protocol.py:287`) | #3 | – | CONFIRMED |
| Version change invisible while `last_version == 0` (:483) | #4 | – | CONFIRMED |
| `write_idx` uint32 overflow → `struct.error` from publish (:410) | #5 | – | CONFIRMED, low |
| Timestamp torn/next-frame read (:490) | #6 | – | CONFIRMED, low |
| No overrun detection when the consumer lags > num_slots (:487) | #7 | – | PLAUSIBLE |
| `open_or_create` loses the create race with a second Sender (:69) | #9 | – | CONFIRMED, low |
| `FLAGS_BGRA` defined and masked but never set or inspected (:59) | – | #2 | CONFIRMED (dead protocol flag) |
| Test gap: double acquire, bump_skip vs decrement race | #10 | – | – |

Claude's B1 note that TDSender `set_version(_ipc_version)` could regress the version was **checked in B7 and dropped**: `initialize` seeds `_ipc_version` from `read_version` after `Exporter.open` bumps (TDSender.py:494), and the reopen `+= 1` matches the re-attached segment.

### B2 — CUDA runtime wrapper + adapters (Claude 9, agy 4)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| `restore_context` leaks the primary-ctx retain on failure; passes `c_int(device)` instead of `cu_dev` (`cuda_ipc_wrapper.py:821-825`) | #4 | #1 | CONFIRMED |
| Recovered non-sticky CUDA errors become permanently fatal (:689) | #1 | – | CONFIRMED |
| Hard-coded toolkit version list misses installed cudarts (:158) | #3 | – | CONFIRMED |
| `get_cuda_runtime` singleton init not thread-safe (:1464) | #7 | – | CONFIRMED, low |
| 1D graph node setter requires `c_void_p`; int ptr raises `AttributeError` (`cuda_graphs.py:246`) | #2 | – | CONFIRMED, latent |
| `_FakeIpcHandle` all-zero bytes treated as unwritten by importer.py:1271 / TDReceiver.py:816 (`_cuda_adapters.py:97`) | – | #2 | CONFIRMED (test-double fidelity) |
| Fake adapter cannot detect double free / foreign close (:201) | #5 | – | CONFIRMED |
| `__getattr__` recursion when `_api` unset (:54) | #6 | – | CONFIRMED, low |
| `cudart_event_query_fn_ptr` untested (:1009) | – | #3 | CONFIRMED (test gap) |
| Unused graph setters / 3D params / `check_error` (`cuda_graphs.py:262/:37`) | #8 | #4 | CONFIRMED (dead code) |
| Duplicate error formatting (:599) | #9 | – | CONFIRMED |

### B3 — Exporter (Claude 10, agy 6)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| Partial `open()` leak (`exporter.py:236`) | #1 | #1 | CONFIRMED |
| SHM reuse without size check → `IndexError` in `set_shutdown` (:328) | #2 | #4 | CONFIRMED |
| Invalid `gpu_ptr` cached + sticky error under `strict_device=False`; `int(frame.ptr)` TypeError; ptr cache cap (:614-635) | #3, #4 | #3 | CONFIRMED |
| Dead `_ts_offset` (:336) | #10 | #6 | CONFIRMED |
| Version/shutdown written before metadata, no fence (:334) | – | #2 | CONFIRMED |
| Non-positive dims not rejected (:134) | – | #5 | CONFIRMED |
| `cudaGetDevice` per frame (:518) | #5 | – | CONFIRMED, efficiency |
| `__del__` sleeps (:904) | #7 | – | CONFIRMED, low |
| Doorbell negative timeout (`_doorbell.py:150`) | #8 | – | CONFIRMED, low |
| Raises doc mismatch (:220) | #9 | – | doc |
| Test gap (export error paths) | #6 | – | – |

### B4 — Importer (Claude 10, agy 8)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| `_connect` leaks the connection when buffer build fails after rollback block (`importer.py:1148`) | #1 | #4 | CONFIRMED |
| NVTX range leaked on non-timeout exceptions (:1669/:1680) | #3 | #1 | CONFIRMED |
| `_reinitialize` skips validation / stale `_last_write_idx` / drops spec (:1852-1861) | #4 | #3 | CONFIRMED |
| `int(timeout_ms)` truncates sub-ms timeouts to 0 (:1551) | #7 | #6 | CONFIRMED |
| Back-buffer / stream allocation leak in `NumpyBuffers.build` (:496/:539) | #8 | #5 | CONFIRMED |
| Timed-out frame consumed without retry (:1675) | #2 | – | CONFIRMED |
| `_RetryState` never resets (:1413) | #5 | – | CONFIRMED, low |
| Per-frame `from_buffer` (:1544) | #6 | – | CONFIRMED, efficiency |
| `from_env` debug leftover (`_importer_port.py:123`) | #9 | – | low |
| `_apply_format_change` never rebuilds CuPy buffers (:1829) | – | #2 | CONFIRMED |
| `wait_for_doorbell` blocks with `policy.doorbell=False` (:1637) | – | #7 | PLAUSIBLE |
| Provisional re-resolve ignores partial `ImportSpec` (:1491) | – | #8 | CONFIRMED |
| Test gap (reconnect/leak paths) | #10 | – | – |

### B5 — Native waiter (Claude 9, agy 3)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| CUDA errors reported as `TIMEOUT` (`native_waiter.cpp:231`) | #1 | #1 | CONFIRMED |
| Deadline checked before query; `timeout 0` never polls (:192-195) | #2 | #2 | CONFIRMED |
| Global fn-ptr set before validation (`_native_loader.py:60`) | #3 | #3 | CONFIRMED |
| `READY_DOORBELL` returned without a doorbell wait (:207) | #4 | – | PLAUSIBLE |
| `KeyError` on unknown status (`_native_loader.py:112`) | #5 | – | CONFIRMED |
| Per-thread timer failure silent (:283) | #6 | – | CONFIRMED, low |
| `ImportError` always reported as "not built" (`_native_loader.py:52`) | #7 | – | CONFIRMED, low |
| Retired `write_idx` plumbing (:237) | #9 | – | dead code |
| Test gap (`tests/core/test_native_state_machine.py:117`) | #8 | – | – |

### B6 — TD facade + host + config + bootstrap (Claude 7, agy 5)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| `init_pace`, `persist_stream`, `nvml` parsed but never consumed (`TDConfig.py:40-44`) | #4 | #1 | CONFIRMED (nvml verified unconsumed in `td_exporter/`) |
| Status setters write COMP storage / `addScriptError` every call (`TDHost.py:377/389`) | #1 | – | CONFIRMED, efficiency |
| `set_warning_status` never clears prior script errors (`TDHost.py:377`) | – | #4 | CONFIRMED |
| `switch_mode` / `reconfigure_and_reinit` leak status tint: only `TDSenderEngine.cleanup` calls `clear_status` (`CUDAIPCExtension.py:526`) | – | #3 | CONFIRMED |
| Sibling mirrors suppress the unavailable notice even when `LIBRARY_READY` is False (:299) | – | #2 | PLAUSIBLE — silent fallback with mirrors present is documented intent (CUDALinkBootstrap docstring); only the "mirrors present but broken" edge is a gap |
| Dead `CUPY_AVAILABLE` / `cp` globals (:152) | – | #5 | CONFIRMED (no test patches them) |
| New `RealTOPHandle` per receive frame (:468/477/488) | #2 | – | CONFIRMED, efficiency |
| `export_sync` comment contradicts resolver (`TDConfig.py:50`) | #3 | – | doc |
| Facade `initialize()` drops `extra_flags` (:456) | #5 | – | CONFIRMED, low |
| Only `NameError` suppressed around `delayRef=op.TDResources` (:307) | #6 | – | CONFIRMED, low |
| Cached `par.Active` stale after par rebuild (`TDHost.py:275`) | #7 | – (agy dropped it as covered by tests) | PLAUSIBLE, low |
| `CUDALinkBootstrap.py` | none | none | read in full; rollback, rival detection and layer precedence match docstring + `tests/td/test_td_bootstrap.py` |

### B7 — TD Sender (Claude 9, agy 5)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| Unguarded `Exporter.open` on reopen wedges the sender (`TDSender.py:745`) | #1 | #1 | CONFIRMED |
| Failed `initialize` leaks a barrier increment per cook (:440/:504) | #2 | #2 | CONFIRMED |
| Reopen check ignores `extra_flags` (:721) | #3 | #3 | CONFIRMED |
| `ZeroDivisionError` in log f-string when `px == 0` (:706, :692) | #4 | #4 | CONFIRMED |
| No GPU-free tests for the above (:440) | #9 | #5 | – |
| Barrier held until stale timeout on non-published paths (:803) | #5 | – | CONFIRMED, low |
| Grow guard skips every frame silently (:784) | #6 | – | CONFIRMED, low |
| `num_slots == 0` → `ZeroDivisionError` per cook (:589) | #7 | – | CONFIRMED, low |
| `AttributeError` on None `ipc_stream` escapes `initialize` (:360) | #8 | – | PLAUSIBLE, low |

### B8 — TD Receiver (Claude 10, agy 4)

| Finding | Claude | agy | Verdict |
| --- | --- | --- | --- |
| Refresh failure double-closes old ptrs and leaks new (`TDReceiver.py:1148-1173`) | #2 | #3 | CONFIRMED |
| Refresh keeps stale slot layout / never re-reads `NUM_SLOTS` (:1095) | #1 | – | CONFIRMED |
| Refresh does not reset `last_write_idx` (:1206) | – | #1 | CONFIRMED |
| Refresh skips float16 buffer / CuPy view rebuild (:1186) | – | #2 | CONFIRMED |
| Refresh ignores `flags` for the format update (:1209) | – | #4 | CONFIRMED |
| Event-less fallback syncs the wrong stream → torn frames (:473) | #3 | – | CONFIRMED |
| `par.format` never restored to `rgba32float` (:886) | #4 | – | CONFIRMED |
| Frame consumed before the copy succeeds (:451) | #5 | – | CONFIRMED, low |
| Backoff not reset after shutdown cleanup (:1081) | #6 | – | CONFIRMED, low |
| `td.tdError` from `copyCUDAMemory` escapes `onCook` (:577) | #7 | – | PLAUSIBLE |
| Header reads duplicate SHMProtocol helpers (:676/704/782) | #8 | – | reuse |
| Dead stream-reuse branch in `_connect` (:795) | #9 | – | simplification |
| Test gap: refresh failure, stale layout, format restore | #10 | – | – |

## 4. Observations on the two reviewers

- **Overlap**: 19 findings were reported independently by both families, all CONFIRMED. The reviewers never contradicted each other; the one Claude note that would have (B1's `set_version` regression) was retracted after reading B7.
- **Claude** reported more (74 vs 37) and covered efficiency, reuse, simplification and test-coverage angles that agy's code-review prompt does not emit.
- **agy** found eight correctness defects Claude missed, concentrated in the two refresh/reinitialize paths (`importer.py:1829/1861/1491`, `TDReceiver.py:1186/1206/1209`) and the TD status lifecycle (`CUDAIPCExtension.py:526`, `TDHost.py:377`). Both reviewers missed each other's half of the same `_refresh_on_version_change` bug cluster, which makes that function the single best fix target.
- **Hot-path concentration**: of the top 20, items 1-3, 10-11 and 16 sit on the per-frame send/receive paths; the rest are reconnect/restart paths that run rarely but fail permanently.

## 5. Run record

| Block | Claude | agy | Notes |
|---|---|---|---|
| B1-B8 | reported via `ReportFindings`, notes in session scratchpad `claude-reviews/block-B<N>.md` | `SUCCESS` on first run, 1-9 min each | no reruns needed |

Raw agy output (`block-<N>.agy.json`, `.md`, prompt and context) and the Claude notes live in the session scratchpad and are not committed.
