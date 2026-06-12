# cuda-link — Phase C: R2 Win32 named-event doorbell (single consumer, opt-in, default OFF)

## Context

Phases R1/R4, A (CuPy `_event_to_int` + 30/60 fps bench), B (R3 f16 pinning), and D (adaptive
gpu-wait latch) are all **done** on `feat/r1-r4-wait-path-perf` (latest `013d910`, 764 tests green).
The one remaining item from the previous session's design (`humming-singing-sunbeam.md`) is **R2**.

**Problem it solves.** A Python consumer that has no new frame today spins a 1 ms poll-sleep
(`td_exporter/example_receiver_python.py:257`, `time.sleep(0.001)` on `NO_FRAME`). At 30/60 fps that
is pure wasted CPU and adds up to ~1 ms of notify latency between "producer published" and "consumer
noticed." R2 replaces that poll-sleep with a **kernel-level wake**: the producer signals a Win32
auto-reset named event right after it publishes a frame; the consumer blocks on that event instead of
sleeping. Expected: sub-300 µs notify latency and ~zero idle CPU while waiting.

**Scope / decisions (carried from prior session, re-confirmed):**
- **Single consumer.** Auto-reset event wakes exactly one waiter — this is a single-consumer
  optimisation, not a broadcast.
- **Opt-in, default OFF** behind `CUDALINK_DOORBELL=1` (read on *both* producer and consumer).
  Poll-sleep remains the default and the >1-consumer / non-Windows fallback.
- **Branch: `feat/r2-doorbell` based off `feat/r1-r4-wait-path-perf`** (carries A–D; rebase onto
  `development` for the clean PR later).

**Standing constraints (do not relearn the hard way):**
- Wrapper-mirror invariant: edit `src/cuda_link/*.py`, run `python scripts/sync_td_wrapper.py`, never
  hand-edit the `td_exporter/` twin; verify with `--check`.
- Commit via `./scripts/git/commit_enhanced.sh --no-venv --only <paths> "feat: ..."` — never raw git.
  Never stage `CLAUDE.md`/`MEMORY.md`/`.claude/`/`logs/`.
- cmd.exe env syntax (`SET VAR=value`), never PowerShell `$env:VAR`.
- `_resolve_export_sync(None) → True` (blocking) must never change.
- Multi-phase work stays on the feature branch — never edit `development` directly.

---

## Index verification (code-search, project=cuda-link, 2026-06-12)

All Phase C anchors re-confirmed against the live index — **line numbers shifted from the prior
plan because Phase D landed**; current values below:

| Anchor | Prior plan | **Current (verified)** |
|---|---|---|
| WinDLL/`os.name=="nt"` guard precedent | `cuda_ipc_wrapper.py:39–44` | `cuda_ipc_wrapper.py:39–44` ✓ (exact) |
| `publish_frame` call site (producer SetEvent) | `:697` | `src/cuda_link/exporter.py:697` ✓ |
| `set_shutdown` call site (shutdown SetEvent) | `:769` | `src/cuda_link/exporter.py:769` (in `_do_cleanup`) ✓ |
| `Exporter._do_cleanup` (CloseHandle) | `757–766` | `exporter.py:756–…` ✓ |
| `Exporter.__init__` (handle field default) | `133–188` | `exporter.py:133–189` ✓ |
| `IPCConnection` (consumer handle field) | `261–309` | `importer.py:261–309`; `close_ipc_handles` `:280`, `close` `:300` ✓ |
| `_open_ipc_slots` (OpenEventW) | `1145–1197` | `importer.py:1174–1226` (+29) |
| `_reinitialize` (reopen) | `1498–1542` | `importer.py:1550–1594` — reopens via `close_ipc_handles`→`_open_ipc_slots`, no extra edit |
| consumer poll-sleep on NO_FRAME | `example_receiver_python.py:255–257` | `:255–257` ✓ |
| `ExportPolicy` / `from_env` | `_exporter_port.py:60–92` | `:60–92` ✓ |
| `ImportPolicy` / `from_env` | `_importer_port.py:49–94` | `:49–108` ✓ |
| `sync_td_wrapper.PAIRS` | — | `scripts/sync_td_wrapper.py:47–64`; `NAMES` auto-derived `:80` |
| sync guard | `test_wrapper_sync.py:87–111` | `test_pairs_cover_all_mirrorable_modules` `:87` ✓ |
| README env-var guard | `test_all_env_vars_documented_in_readme` | `tests/support/test_env_var_docs.py:60` — scans `env_bool/int/str("CUDALINK_*")` in `src/`, requires a row under `### Performance Tuning (env vars)` |

**Greenfield confirmed:** no existing `doorbell` / `CUDALINK_DOORBELL` / `CreateEventW` scaffolding in
the repo.

**Two refinements over the prior design (resolved here, not a redesign):**

1. **The NO_FRAME sleep lives in the consumer *application loop*, not importer internals.** R2's wait
   replaces `example_receiver_python.py:257`. `_wait_for_slot` (`importer.py:1346`) is a *different*
   wait — it blocks on the per-slot IPC copy-complete event *after* a frame is already detected (the
   R1/D gpu-wait path) and is **untouched** by R2. So the importer **exposes a wait primitive** (it
   owns `shm_handle.buf` + `_last_write_idx`) and the example loop calls it.
2. **`_doorbell.py` has no relative imports → sync mode `byte_identical`** (env reads stay in the
   policy dataclasses). It must still be registered in `PAIRS` so `NAMES` gains the `_doorbell→Doorbell`
   key that `exporter.py`/`importer.py` (both `rewrite_relative`) need to rewrite
   `from ._doorbell import …` → `from Doorbell import …`.

---

## Implementation

### C1. New mirrored module — `src/cuda_link/_doorbell.py` (+ `td_exporter/Doorbell.py`)

Pure Win32/ctypes; `import ctypes, logging, os` only (no relative imports). Copy the guard pattern
from `cuda_ipc_wrapper.py:39–44`:

```python
if os.name == "nt":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # CreateEventW(lpEventAttributes, bManualReset, bInitialState, lpName) -> HANDLE
    _k32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    _k32.CreateEventW.restype  = ctypes.c_void_p
    # OpenEventW(dwDesiredAccess, bInheritHandle, lpName) -> HANDLE
    _k32.OpenEventW.argtypes   = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    _k32.OpenEventW.restype    = ctypes.c_void_p
    _k32.SetEvent.argtypes     = [ctypes.c_void_p]; _k32.SetEvent.restype = ctypes.c_int
    _k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k32.WaitForSingleObject.restype  = ctypes.c_uint32
    _k32.CloseHandle.argtypes  = [ctypes.c_void_p]; _k32.CloseHandle.restype = ctypes.c_int
else:
    _k32 = None

_EVENT_MODIFY_STATE = 0x0002
_SYNCHRONIZE        = 0x00100000
_WAIT_OBJECT_0      = 0x0
```

Public surface (all return `None` / `False` no-op when `_k32 is None`, so callers stay
platform-agnostic):
- `doorbell_event_name(shm_name: str) -> str` → `r"Local\cudalink_db_" + shm_name`. Document the
  `Global\` cross-session caveat and the ~260-char name ceiling.
- `create_doorbell(name) -> handle|None` — `CreateEventW(None, False, False, name)`: **auto-reset**
  (`bManualReset=False`), initial non-signaled. Producer side.
- `open_doorbell(name) -> handle|None` — `OpenEventW(_EVENT_MODIFY_STATE|_SYNCHRONIZE, False, name)`;
  returns None if it doesn't exist yet (consumer started first). Consumer side.
- `signal(handle) -> None` — `SetEvent`. `wait(handle, timeout_ms) -> bool` — returns
  `WaitForSingleObject(...) == _WAIT_OBJECT_0`. `close(handle) -> None` — `CloseHandle`.

Register in **`scripts/sync_td_wrapper.py` PAIRS** (`:47–64`):
`(_SRC / "_doorbell.py", _TD / "Doorbell.py", "byte_identical")`. Run `python scripts/sync_td_wrapper.py`.

### C2. Policy flags — `_exporter_port.py` + `_importer_port.py`

- `ExportPolicy`: add `doorbell: bool = False`; `from_env` → `doorbell=env_bool("CUDALINK_DOORBELL", default=False)`.
- `ImportPolicy`: add `doorbell: bool = False`; `from_env` → `doorbell=env_bool("CUDALINK_DOORBELL", default=False)`.
- Both reuse the **same** env var. Field docstring: note single-consumer + Windows-only + default-OFF.
- **README:** add a `CUDALINK_DOORBELL` row under `### Performance Tuning (env vars)` or
  `test_env_var_docs.py` fails.

### C3. Producer — `src/cuda_link/exporter.py`

- `__init__` (~`156–161` handle block): `self._doorbell = None`.
- In `open()`/`_initialize` (where `spec.shm_name` is bound and SHM is created): if
  `self._policy.doorbell`, `self._doorbell = _doorbell.create_doorbell(_doorbell.doorbell_event_name(self._spec.shm_name))`.
- `export()`: **after** `publish_frame(...)` at `:697` (inside the `shm_write` nvtx range, on the
  publish path only): `if self._doorbell: _doorbell.signal(self._doorbell)`. **NOT** on the
  `SKIPPED_BARRIER` (`:548`) or `FAILED` returns.
- `_do_cleanup`: **after** `set_shutdown(...)` at `:769`, `if self._doorbell: _doorbell.signal(self._doorbell)`
  (wake a blocked consumer so it observes shutdown); then `_doorbell.close(self._doorbell)` later in the
  same teardown. Cost: one ~1 µs `SetEvent` after the already-blocking export — TD Sender frame
  behaviour unchanged (verify with `profile_export.py`).

### C4. Consumer — `src/cuda_link/importer.py`

- `IPCConnection` (`:261–309`): add field `doorbell_handle: object = None`; in `close_ipc_handles`
  (`:280`) close it (`_doorbell.close(self.doorbell_handle); self.doorbell_handle = None`) alongside the
  event/handle teardown.
- `_open_ipc_slots` (`:1174–1226`): if `self._policy.doorbell`, open
  `_doorbell.open_doorbell(_doorbell.doorbell_event_name(self._spec.shm_name))` and pass it into the
  returned `IPCConnection(... doorbell_handle=...)`. `_reinitialize` (`:1550`) re-runs `_open_ipc_slots`
  so the handle reopens automatically — **no extra edit**.
- New method `Importer.wait_for_doorbell(self, timeout_ms: float) -> bool` — the lost-wakeup-safe
  primitive, reusing existing state:
  ```python
  conn = self._conn
  h = getattr(conn, "doorbell_handle", None) if conn else None
  if h is None:
      return False                       # disabled / not opened / non-Windows -> caller polls
  cur = read_write_idx(conn.shm_handle.buf)
  if cur != 0 and cur != self._last_write_idx:
      return True                        # frame already waiting -- no block
  return _doorbell.wait(h, int(timeout_ms))   # auto-reset; bounded timeout caps any missed signal
  ```
  Reuses `read_write_idx` (`shm_protocol.py:335`) and `self._last_write_idx` (`:948`/`:1324`) — the
  same new-frame test `get_frame` already uses. Caller re-invokes `get_frame()` after a wake to
  re-check, so a spurious/missed signal costs at most one `timeout_ms` slice.

### C5. Consumer app loop — `td_exporter/example_receiver_python.py`

NO_FRAME branch (`:255–257`): replace the bare `time.sleep(0.001)` with:
```python
elif result.outcome is ImportOutcome.NO_FRAME:
    no_frame_count += 1
    if not importer.wait_for_doorbell(2.0):   # ~2 ms cap; returns False when disabled/non-Win
        time.sleep(0.001)
```
(`wait_for_doorbell` returns False immediately when the handle is None → existing poll-sleep path,
zero behaviour change with `CUDALINK_DOORBELL` unset.)

### C6. Mirror + tests

- `python scripts/sync_td_wrapper.py` (touches `_doorbell.py`, `exporter.py`, `importer.py`,
  `_exporter_port.py`, `_importer_port.py`); `python scripts/sync_td_wrapper.py --check` clean.
  `example_receiver_python.py` is **not** mirrored — edit directly.
- New `tests/support/test_doorbell.py`:
  - **Pure (CI-safe everywhere):** `doorbell_event_name("foo") == r"Local\cudalink_db_foo"`.
  - **Importer primitive (GPU-free, `FakeCUDAAdapter` via `make_connected_importer`):**
    `wait_for_doorbell` returns `False` when `doorbell_handle is None`; returns `True` immediately
    after writing an advanced `write_idx` into the connection's buffer (no real handle needed for the
    early-return path).
  - **Real kernel32 round-trip, `@pytest.mark.skipif(os.name != "nt", ...)`:** `create_doorbell` →
    `open_doorbell` second handle → `signal` → `wait(..., 1000)` returns `True`; `wait` on an
    unsignaled event with a short timeout returns `False`; `close` both. No GPU.

---

## Verification

1. `python -m pytest tests/ -q --ignore=tests/requires_cuda` — green incl. new doorbell tests
   (Windows runs the kernel32 round-trip; other CI skips it). Expect ~767+ passed.
2. `python scripts/sync_td_wrapper.py --check` and `pytest tests/support/test_wrapper_sync.py` —
   mirror intact across all five edited canonical modules + new `_doorbell.py`/`Doorbell.py`.
3. `pytest tests/support/test_env_var_docs.py` — `CUDALINK_DOORBELL` documented in README.
4. **Real end-to-end (Windows, two processes):** run `example_sender_python` and
   `example_receiver_python` both with `SET CUDALINK_DOORBELL=1`; confirm consumer idle CPU drops to
   ~0 while waiting and notify latency improves vs the 1 ms poll. Then run **without** the env var and
   confirm identical behaviour to today (poll-sleep default).
5. `python scripts/profiling/profile_export.py` — producer per-frame export time unchanged within
   noise (the added `SetEvent` is ~1 µs after the already-blocking publish).

## Sequencing

1. (Already done) Created `feat/r2-doorbell` off `feat/r1-r4-wait-path-perf`.
2. C1 module + PAIRS entry + `sync_td_wrapper.py` run → confirm `test_wrapper_sync` green.
3. C2 policy flags + README row.
4. C3 producer, C4 consumer, C5 example loop.
5. C6 sync + tests; full verification.
6. Commit with `./scripts/git/commit_enhanced.sh --no-venv --only <paths>
   "feat: R2 Win32 named-event doorbell (single consumer, opt-in CUDALINK_DOORBELL)"`.
