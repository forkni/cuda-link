# Migrating to cuda-link v1.6.0

> **Migration window closed in v1.7.0.**
> `CUDAIPCExporter` was removed in v1.7.0 as scheduled. This guide documents the migration
> path for reference. If you are still on the old API, apply the changes below and upgrade
> to v1.7.0 or later.

## Overview

v1.6.0 introduces `Exporter` — a deep, testable module that replaces the inline CUDA IPC
logic in `CUDAIPCExporter`. `CUDAIPCExporter` was **deprecated in v1.6.0** and **removed in
v1.7.0**.

---

## Quick migration

### 1. Import path

```python
# Before (v1.5.x and earlier)
from cuda_link import CUDAIPCExporter
from cuda_link.cuda_ipc_exporter import CUDAIPCExporter  # also worked

# After (v1.6.0+)
from cuda_link import Exporter, FrameSpec, ExportPolicy, GpuFrame, FrameOutcome
```

### 2. Construction

```python
# Before
exp = CUDAIPCExporter(shm_name="my_shm", height=1080, width=1920)
exp.initialize()  # required separate call

# After — single call, raises on failure (no half-states)
exp = Exporter.open(
    FrameSpec(shm_name="my_shm", height=1080, width=1920),
)
```

### 3. Exporting a frame

```python
# Before
ok: bool = exp.export_frame(gpu_ptr=ptr, size=size)

# After
from cuda_link import GpuFrame, FrameOutcome

outcome: FrameOutcome = exp.export(GpuFrame(ptr=ptr, size=size))
ok = outcome != FrameOutcome.FAILED
```

### 4. Source-stream synchronization

```python
# Before
exp.record_source_sync(producer_stream_handle)

# After — same method, same signature
exp.record_source_sync(producer_stream_handle)
```

### 5. Cleanup

```python
# Before
exp.cleanup()  # or relies on __del__

# After — idempotent; safe to call multiple times
exp.close()

# Context manager (unchanged)
with Exporter.open(FrameSpec(...)) as exp:
    exp.export(GpuFrame(ptr=ptr, size=size))
```

---

## Policy configuration

The new `ExportPolicy` dataclass replaces environment-variable-only configuration:

```python
# Before (env vars only)
# CUDALINK_EXPORT_SYNC=0
# CUDALINK_USE_GRAPHS=1

# After — pass explicitly; env vars still work via ExportPolicy.from_env()
policy = ExportPolicy(
    export_sync=False,
    use_graphs=True,
    flush_probe=True,
    strict_device=False,
    barrier_enabled=True,
    high_priority_stream=True,
)

# Or from environment (old behaviour)
policy = ExportPolicy.from_env()

exp = Exporter.open(FrameSpec(...), policy=policy)
```

Named presets:

```python
ExportPolicy.for_testing()   # export_sync=False, use_graphs=False, flush_probe=False, …
ExportPolicy.low_latency()   # high_priority_stream=True, flush_probe=True, export_sync=False
```

---

## FrameOutcome vs bool

`export()` returns a `FrameOutcome` enum instead of a plain `bool`:

| Outcome | Meaning |
|---|---|
| `PUBLISHED` | Frame written to shared memory successfully |
| `SKIPPED_BARRIER` | Activation barrier active — no consumer connected |
| `SKIPPED_NOT_READY` | Ring slot still held by consumer — frame dropped |
| `FAILED` | CUDA or SHM error — exporter may need to be recreated |

```python
from cuda_link import FrameOutcome

outcome = exp.export(GpuFrame(ptr=ptr, size=size))
if outcome == FrameOutcome.PUBLISHED:
    frames_sent += 1
elif outcome == FrameOutcome.SKIPPED_BARRIER:
    pass  # normal during consumer startup
elif outcome == FrameOutcome.FAILED:
    logger.error("export failed — check GPU/SHM state")
```

---

## Unit testing without a GPU

The `FakeCudaAdapter` lets you test export logic without hardware:

```python
from cuda_link import ExportPolicy, FrameSpec, GpuFrame, FrameOutcome
from cuda_link._cuda_adapters import FakeCudaAdapter
from cuda_link.exporter import Exporter

def test_my_export_logic():
    fake = FakeCudaAdapter(device=0)
    exp = Exporter.open(
        FrameSpec(shm_name="test_shm", height=8, width=8, channels=4,
                  dtype="uint8", num_slots=2, device=0),
        policy=ExportPolicy.for_testing(),
        cuda=fake,
    )
    try:
        outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=8*8*4))
        assert outcome != FrameOutcome.FAILED
        assert len(fake.allocations) == 2  # two ring-buffer slots
    finally:
        exp.close()
```

---

## Strict device checking

`strict_device=True` raises `ValueError` instead of logging on device mismatches:

```python
policy = ExportPolicy(strict_device=True)
# or: CUDALINK_STRICT_DEVICE=1

exp = Exporter.open(FrameSpec(..., device=0), policy=policy)
exp.export(GpuFrame(ptr=wrong_device_ptr, size=size))
# → raises ValueError("belongs to device 1, but exporter is bound to device 0")
```

---

## `CUDAIPCExporter` backwards compatibility

If you cannot migrate immediately, `CUDAIPCExporter` still works via a shim:

```python
# Unchanged — emits DeprecationWarning at import time
exp = CUDAIPCExporter(shm_name="my_shm", height=1080, width=1920)
exp.initialize()
exp.export_frame(ptr, size)
exp.cleanup()
```

The shim delegates every call to an inner `Exporter` instance. All env-var flags
(`CUDALINK_EXPORT_SYNC`, `CUDALINK_USE_GRAPHS`, etc.) are read at construction time via
`ExportPolicy.from_env()` — identical to the v1.5.0 behaviour.

**Removal timeline:** `CUDAIPCExporter` will be removed in v1.7.0.
