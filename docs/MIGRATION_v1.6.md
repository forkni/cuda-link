# Migrating to cuda-link v1.6.0

> No breaking API changes. All public Python interfaces are backwards-compatible.
> The one change that may require action is the `CUDALINK_EXPORT_SYNC` default flip.

---

## `CUDALINK_EXPORT_SYNC` default changed: `1` → `0`

**Who is affected**: Python-side `Exporter` callers only. The TD Sender is unaffected
(its `TDConfig.export_sync` default remains `True` for TDR-cascade safety).

### What changed

`ExportPolicy.export_sync` now defaults to `False`. When `sync=False`, the
`Exporter` relies on the CUDA IPC event for cross-process GPU ordering instead of a
blocking `cudaStreamSynchronize`. The non-blocking `flush_probe`
(`cudaStreamQuery`, already default-on) replaces it for WDDM driver submission.

### When to set `CUDALINK_EXPORT_SYNC=1`

Set the env var or pass `ExportPolicy(export_sync=True)` when:

- Your topology runs a **concurrent TD Sender and Python Exporter in the same process**
  (prevents TDR-cascade on first-frame settle).
- You use `CUDALINK_USE_GRAPHS=0` (legacy stream path, no IPC event recorded).
- You observe stale frames or incorrect ordering in your pipeline after upgrading.

### No action needed if

- You use the `Exporter` in a standalone Python process without a co-located TD Sender.
- You already pass `ExportPolicy(export_sync=False)` explicitly.
- You use only `Importer` / `get_frame*()` (consumer side; unaffected).

---

## New public APIs (additions, no migration needed)

### `DtypeCodec` backend accessors

```python
from cuda_link.shm_protocol import DtypeCodec

# New in v1.6.0 — expose per-dtype backend representations
typestr  = DtypeCodec.typestr("float32")   # "<f4"  (CAI typestr)
np_name  = DtypeCodec.numpy_name("float32") # "float32"  (None for bfloat16)
cp_name  = DtypeCodec.cupy_name("float32")  # "float32"  (None if unsupported)
```

### `Importer.from_connection` — `last_write_idx` parameter

Advanced callers that adopt an `IPCConnection` out-of-band can now specify the
initial consumed-frame index at construction time:

```python
importer = Importer.from_connection(
    spec, policy, conn, fmt,
    last_write_idx=5,   # treat frames 0-5 as already consumed
)
```

### `ImporterCudaPort` = `CudaPort`

`ImporterCudaPort` is now a public alias for `CudaPort` (the Protocol was unified
to the full union of all methods). Existing code that imports `ImporterCudaPort`
continues to work without change.

---

## Deprecation reminder (scheduled for v1.8.0)

`CUDAIPCImporter` — the legacy shim wrapping `Importer` — emits a
`DeprecationWarning` once per process. Migrate to `Importer.open()`:

```python
# Before
from cuda_link import CUDAIPCImporter
imp = CUDAIPCImporter(shm_name="my_shm")
imp.connect()

# After
from cuda_link import Importer, ImportSpec
imp = Importer.open(ImportSpec(shm_name="my_shm"))
```

Both the Exporter and Importer APIs changed in v1.5.0; see the v1.5 release notes in
`CHANGELOG.md` for the full before/after reference.
