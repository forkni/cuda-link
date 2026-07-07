# CUDA-Link Examples

Standalone, runnable scripts that teach how to integrate CUDA-Link into your own
Python programs. Every script runs end-to-end on a single machine with nothing
else open: **01** walks the full API surface in a single process via
`FakeCUDAAdapter`, and every other example spawns its own **demo producer
process** that stands in for TouchDesigner. Every script is heavily commented —
the comments are the documentation; run order below is the reading order.

**Start with 01 and 02.** 01 walks the whole API surface without any GPU;
02 introduces the one non-negotiable pattern (separate OS processes for CUDA
IPC) that every later script builds on. After those two, jump to whichever
script matches your use case.

These are the runnable counterparts of the embedded-code workflows in
[docs/INTEGRATION_EXAMPLES.md](../docs/INTEGRATION_EXAMPLES.md) — the doc
explains, these execute.

## Index

| Script | Teaches | Consumes via | Hardware | Standalone? |
|---|---|---|---|---|
| [01_fake_adapter_tour.py](01_fake_adapter_tour.py) | Full API surface, ring-buffer slot cycling, outcome enums, stats, shutdown handshake — single process via `FakeCUDAAdapter` | `get_frame_numpy()` (fake pixels) | **none** | yes |
| [02_two_process_roundtrip.py](02_two_process_roundtrip.py) | The mandatory `multiprocessing("spawn")` pattern; real GPU roundtrip with pixel verification | `get_frame_numpy()` — CPU readback IS the proof | NVIDIA GPU¹ | yes |
| [03_td_to_pytorch_pipeline.py](03_td_to_pytorch_pipeline.py) | **Zero-copy `get_frame()`** → torch tensors that live in the ring buffer; the lifetime contract; FPS/latency monitoring | **zero-copy `get_frame()`** | NVIDIA GPU + CUDA torch | yes² |
| [04_td_to_opencv_numpy.py](04_td_to_opencv_numpy.py) | `get_frame_numpy()` CPU path (your own copy, no lifetime contract); RGBA→BGRA; optional cv2 display | `get_frame_numpy()` — the copy IS the lesson | NVIDIA GPU¹ | yes² |
| [05_multi_stream_and_dynamic_resolution.py](05_multi_stream_and_dynamic_resolution.py) | N streams = N SHM names = N Importers; `shape=None` auto-detect; following a producer resolution change | **zero-copy `get_frame()`**³ | NVIDIA GPU¹ | yes |
| [06_graceful_shutdown_and_reconnect.py](06_graceful_shutdown_and_reconnect.py) | The production template: SHUTDOWN vs RECONNECTING vs TIMEOUT, reconnect loop, Ctrl+C / console-close cleanup | **zero-copy `get_frame()`**³ | NVIDIA GPU¹ | yes |
| [07_python_to_td_exporter_and_benchmark.py](07_python_to_td_exporter_and_benchmark.py) | The producer side: `Exporter`/`GpuFrame`/`record_source_sync` ordering rule; p50/p95/p99 export cost | — (produces: `export()` straight from a GPU tensor) | NVIDIA GPU¹ | yes |
| [08_low_cpu_doorbell_consumer.py](08_low_cpu_doorbell_consumer.py) | `doorbell=True` policies + `wait_for_doorbell(2000.0)` — block instead of spin-poll | **zero-copy `get_frame()`**³ | NVIDIA GPU¹ | yes |

¹ Degrades to `FakeCUDAAdapter` (prints why; protocol runs, pixels are synthetic/zero). Force it with `--force-fake`.
² Can attach to a live TouchDesigner sender instead of the demo producer: `--no-demo-producer --shm-name <name>`.
³ Zero-copy when CUDA-enabled torch is installed; falls back automatically to `get_frame_numpy()` (CPU-only torch, no torch, or `--force-fake`). Each script prints which path it took; loop structure and outcome handling are identical on both.

**The zero-copy path is the point of the library.** **03** demonstrates it in
depth on the consumer side (`get_frame()` returns a torch tensor that *lives in
the ring buffer* — no per-frame copy, no allocation — plus the lifetime contract
that comes with that) and **07** on the producer side (`export()` takes a raw GPU
pointer; one device-to-device copy into the ring, never through host memory).
05, 06, and 08 consume zero-copy by default too³. `get_frame_numpy()` — a
device-to-host copy per frame — is the deliberate subject only of 04, and of
02, whose pixel proof needs CPU readback.

`_common.py` holds only non-pedagogical plumbing (hardware probes, unique SHM
names, the demo producer, the torch/cupy preflight). Everything worth learning —
`open()` calls, specs, policies, outcome handling — is repeated in full in each
numbered script on purpose.

## Install

From the repo root (a plain `pip install cuda-link` works the same way):

```bash
pip install -e .                  # core (numpy comes in via the demo producers' needs)
pip install -e ".[torch]"         # + PyTorch: zero-copy consumption in 03/05/06/08, torch render path in 07
pip install -e ".[all]"           # + torch, numpy, cupy, nvml
pip install opencv-python         # for 04's display window — deliberately NOT a cuda-link extra
```

The torch wheel must be **CUDA-enabled** (`torch.cuda.is_available()` → `True`).
A CPU-only wheel cannot run the zero-copy path — 05/06/08 detect that and fall
back to numpy; 03 refuses to run, because zero-copy IS its lesson. See
Troubleshooting for why a CPU-only wheel is worse than no wheel at all here.

## Common CLI flags

Documented once here; every script supports `--help`.

| Flag | Meaning |
|---|---|
| `--frames N` | Bounded run length. Examples stop after N frames; production consumers loop until `SHUTDOWN`. |
| `--force-fake` | Use `FakeCUDAAdapter` even if a real GPU is present (protocol only, no real pixels). |
| `--shm-name NAME` | Attach to a live producer (e.g. a TouchDesigner sender) instead of generating a unique name. |
| `--no-demo-producer` | (03, 04) Don't spawn the demo producer; requires `--shm-name`. |
| `--width / --height / --fps` | Geometry and rate of the demo producer where it applies. |

By default every run generates a **unique SHM name** (`prefix_<pid>_<random>`), so
parallel runs never collide. When you pass `--shm-name`, collision avoidance is
on you — two producers on one name corrupt each other's ring.

## Troubleshooting

### `Importer.open()` crashes with a CUDA error even though I only wanted numpy frames

This is the one real landmine. `Importer.open()` **eagerly** builds per-slot
zero-copy torch/cupy GPU views whenever those packages are merely *importable* —
it calls `torch.as_tensor(wrapper, device="cuda")` regardless of which adapter
you passed or which `get_frame*()` method you intend to call. Two setups break:

- **CPU-only torch installed** → `device="cuda"` raises, even though your code
  never touches torch.
- **`FakeCUDAAdapter`** → the fake's fabricated IPC "pointers" must never reach
  real torch/cupy.

Fix: flip the importer module's availability flags **before** `open()`, in
*every process* that opens an Importer (module state doesn't cross a spawn):

```python
import cuda_link.importer as importer_module
importer_module.TORCH_AVAILABLE = False   # and/or CUPY_AVAILABLE
```

The examples do this automatically via `_common.make_importer_open_safe()` and
print a `NOTE:` line when the flags were flipped.

### `cudaIpcOpenMemHandle` fails with error 201 (invalid device context)

Producer and consumer are in the **same process**. CUDA IPC on Windows requires
separate OS processes — see example 02 for the spawn pattern. Threads are not
enough; neither is `fork` semantics (Windows always spawns anyway).

### Consumer hangs at startup / `no producer appeared`

The producer needs ~1-2 s to load CUDA and allocate the ring before the SHM
name exists. The examples poll for the SHM (`_common.wait_for_shm`) and then
settle for 0.3 s before opening — copy that ordering. If you attached with
`--shm-name`, check the producer really uses that exact name.

### `TIMEOUT` outcome mid-run

The producer died or hung without closing cleanly (a clean close yields
`SHUTDOWN` instead). Recovery is the same reconnect loop either way — see
example 06.

### Doorbell never fires (example 08 reports 0 wakes)

Both sides must opt in (`ExportPolicy(doorbell=True)` **and**
`ImportPolicy(doorbell=True)`), and it is Windows-only. When unavailable,
`wait_for_doorbell()` returns `False` immediately and the loop degrades to
1 ms polling — functional, just not low-CPU.
