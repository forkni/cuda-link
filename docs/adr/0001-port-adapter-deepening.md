# ADR-0001: Port + Adapter deepening template

**Status**: Accepted  
**Date**: 2026-05-20  
**Applies to**: `src/cuda_link/exporter.py`, `src/cuda_link/importer.py`, and any future module that owns a long-lived GPU or IPC resource.

---

## Context

Before v1.5.0, the Exporter and Importer were monolithic classes (`CUDAIPCExporter`, `CUDAIPCImporter`) that directly invoked the CUDA runtime via ctypes on every operation. This made them:

- **Untestable without a GPU** — any test that exercised real code paths needed a live CUDA device.
- **Leaky at their interface** — callers needed to know internal resource lifecycle (when handles were valid, which CUDA calls had been made).
- **Hard to change** — touching the GPU resource lifecycle required understanding the whole class.

The specific symptom that forced action: the only way to unit-test construction was `object.__new__(CUDAIPCExporter)` followed by 25+ hand-populated private attributes — a fragile bypass that broke whenever the class changed.

## Decision

Apply the **Port + Adapters + value-object** template to every module that owns a long-lived GPU or IPC resource:

1. **Extract a `*Port` Protocol** (`_exporter_port.py`, `_importer_port.py`) that declares the CUDA operations the module needs, as a `@runtime_checkable` `typing.Protocol`. This becomes the seam.

2. **Move all value objects into the port file**: frozen dataclasses for Spec, Policy, Outcome, and Result. These hold all the information a caller needs to express intent; the module reads them once at `open()` time and never consults `os.environ` again per-frame.

3. **Write a deep module** (`exporter.py`, `importer.py`) whose `open(spec, policy, cuda=<Port>)` factory constructs the module fully or raises — no half-initialised state. The module is a context manager; `close()` is idempotent.

4. **Provide two adapters** in `_cuda_adapters.py`:
   - `CTypesCUDAAdapter` — satisfies the Port using the real `CUDARuntimeAPI` ctypes wrapper. Used in production.
   - `FakeCUDAAdapter` — satisfies the Port with in-process fakes (no GPU required). Used in all no-GPU tests. Returns `_FakeIpcHandle` objects with `.internal` / `.reserved` attributes matching the 64-byte `cudaIpcMemHandle_t` / `cudaIpcEventHandle_t` shape.

5. **Collapse the legacy class to a deprecation shim** (`cuda_ipc_exporter.py`, `cuda_ipc_importer.py`) that re-exports the new API and emits a `DeprecationWarning` once per process via a `_warn_once()` helper.

6. **Write tests against the seam**, not the implementation. Tests call `Exporter.open(FrameSpec(…), policy=ExportPolicy.for_testing(), cuda=FakeCUDAAdapter())`. This exercises real construction, the real context-manager lifecycle, and the real per-frame path — without a GPU.

## Consequences

**Positive**:
- All construction and export logic is testable without a GPU; the test suite runs on any machine.
- The interface a caller must understand shrinks from "a class with 25+ private attributes and a 4-step initialisation sequence" to "`open()`, `export()`, context manager".
- Bug fixes are local: the Port makes it obvious which CUDA operations the module uses; the adapter absorbs any ctypes-layer changes.

**Negative / trade-offs**:
- Every canonical `src/cuda_link/` module that uses this template uses relative imports (`from ._exporter_port import …`), which prevents byte-identical mirroring into `td_exporter/` without a transform step. See ADR-0002.
- The Spec/Policy split adds a pair of extra dataclasses per module. Callers who previously passed everything as keyword arguments to `__init__` now build two frozen objects. Migration guides are provided.

## Evidence

- v1.5.0: `feat: extract Exporter module` — commit `92fa384`; `refactor: deprecate CUDAIPCExporter` — commit `ac93b67`.
- v1.5.x: `feat: deepen CUDAIPCImporter with Exporter template` — commit `6ff8c45`.
- 307 no-GPU tests pass (current). 0 tests use `object.__new__` bypass.

## Template checklist (for future deepening)

When applying this template to a new module `foo`:

- [ ] `src/cuda_link/_foo_port.py` — `FooSpec`, `FooPolicy`, `FooOutcome`, `FooResult[T]`, `FooPort` Protocol
- [ ] `src/cuda_link/foo.py` — `Foo` class; `Foo.open(spec, policy, cuda=…)` factory; context manager
- [ ] `src/cuda_link/_cuda_adapters.py` — extend `FakeCUDAAdapter` if new CUDA methods are needed
- [ ] `src/cuda_link/cuda_ipc_foo.py` — deprecation shim with re-exports + `_warn_once()`
- [ ] `src/cuda_link/__init__.py` — export new symbols
- [ ] `tests/test_foo_port.py` — contract tests for `FooPort` against `FakeCUDAAdapter`
- [ ] `tests/test_foo.py` — construction + lifecycle + per-frame tests via `FakeCUDAAdapter`
- [ ] `tests/test_foo_deprecation.py` — shim emits warning exactly once
- [ ] `docs/MIGRATION_v<N>.md` — before/after migration guide
- [ ] `CHANGELOG.md` — entry under `[Unreleased]`

---

## Resolution note (2026-06-10): Async export and source-buffer lifetime

**Context:** In 1.10.0, `export_frame()` was made async by default (`CUDALINK_EXPORT_SYNC`
unset → async).  This surfaced a **source-buffer lifetime race** when the TD Sender
integrated with StreamDiffusion (CUDA 719, confirmed by `/diagnose` Loop A).

**Root cause:** `Exporter.export()` reads the caller's source pointer *directly* — there
is no staging buffer.  For the Python `Exporter` API the caller owns a persistent source
buffer and passes `producer_stream`, so async export is correct.  For the **TD Sender**,
the source is TD's cook-scoped `TOP` texture (`cm.ptr`) — an **externally-owned, transient**
pointer reclaimed by TD the instant the cook returns.  Async export lets TD reclaim the
source while the queued IPC-stream D2D copy is still executing → reads freed memory → 719.

**Critical distinction — ordering vs. lifetime:**
- `record_source_sync` / `producer_stream` / `_arm_same_stream_ordering` are **pre-copy
  ordering** primitives: they guarantee the source is fully *written* before the copy
  starts.  They do **not** guarantee the source outlives the queued read.
- `CUDALINK_EXPORT_SYNC=1` (post-copy `stream_synchronize`) is the **source-lifetime
  guard**: it blocks the CPU until the D2D finishes, so the source pointer is provably
  safe to release when `export()` returns.  The two primitives are not substitutes.

**Fix (v1.10.1):** The TD Sender defaults to blocking export (`None` → sync via
`TDSenderEngine._resolve_export_sync`).  Explicit `CUDALINK_EXPORT_SYNC=0` opts into async
for callers with a guaranteed-stable source.  `Exporter._do_cleanup` now synchronizes the
IPC stream (STEP 1b) before destroying GPU resources — closes the same race on geometry/
dtype-change reopens and any explicit `close()` call under async export.

**Architectural implication for this template:** When applying the Exporter template to new
modules, document whether the source buffer is **caller-owned + persistent** (async export
safe) or **externally-owned + transient** (blocking export required).  If an adapter sits
between an external resource manager and `Exporter.export()` (as TDSender does for TD's
`cudaMemory()` textures), the adapter MUST guarantee source-buffer lifetime before returning
control to the resource manager — the Exporter itself cannot do this without a post-copy sync.
