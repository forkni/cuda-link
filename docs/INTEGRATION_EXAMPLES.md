# Integration Examples

Complete workflows for common CUDA IPC use cases.

> **Current API:** Consumer side — `from cuda_link import Importer, ImportSpec, ImportOutcome`.
> Call `Importer.open(ImportSpec(...))` and check `result.outcome is ImportOutcome.NEW_FRAME`
> before reading `result.frame`.  Producer side — `from cuda_link import Exporter, FrameSpec, GpuFrame`;
> call `Exporter.open(FrameSpec(...))` and `exporter.export(GpuFrame(...))`.
> The TouchDesigner COMP facade is `CUDAIPCExtension` (Sender or Receiver mode).

> **▶ Runnable versions:** every workflow below has a standalone, heavily-commented script in
> [`examples/`](../examples/README.md) that spawns its own demo producer — no TouchDesigner
> needed to run them. Each example heading links to its script.

---

## Example 1: TouchDesigner → PyTorch AI Pipeline (Zero-Copy)

> ▶ Runnable: [examples/03_td_to_pytorch_pipeline.py](../examples/03_td_to_pytorch_pipeline.py)

### Use Case
Real-time AI inference (style transfer, object detection, etc.) on TouchDesigner video feed.

### TouchDesigner Setup

1. **Network Layout**:
```
Movie File In TOP → CUDAIPCExtension (Mode=Sender)
                    (Ipcmemname="ai_input")
```

2. **Parameters**:
   - `Mode`: `Sender`
   - `Ipcmemname`: `"ai_input"`
   - `Active`: ON
   - `Numslots`: 3
   - Resolution: 512x512 (or your model's input size)

### Python Script (AI Inference Loop)

```python
import torch
from cuda_link import Importer, ImportSpec, ImportOutcome

# Initialize importer — waits up to 5 s for the TD sender to appear
with Importer.open(ImportSpec(
    shm_name="ai_input",
    shape=(512, 512, 4),   # RGBA; None to auto-detect
    dtype="float32",
    timeout_ms=5000.0,
)) as importer:

    # Load your AI model
    model = torch.jit.load("style_transfer_model.pt").cuda()
    model.eval()

    fps_counter = 0
    import time
    start_time = time.time()

    while True:
        # Get frame (zero-copy GPU tensor)
        result = importer.get_frame()

        if result.outcome is ImportOutcome.SHUTDOWN:
            print("Producer shut down — exiting")
            break
        if result.outcome is not ImportOutcome.NEW_FRAME:
            continue  # NO_FRAME (idle) or RECONNECTING (format change)

        input_tensor = result.frame  # torch.Tensor on GPU, shape (512, 512, 4)

        # Preprocess (convert RGBA → RGB, normalize)
        rgb = input_tensor[:, :, :3]           # Drop alpha
        normalized = (rgb - 0.5) / 0.5        # [-1, 1] range

        # Run inference
        with torch.no_grad():
            output = model(normalized.unsqueeze(0))  # Add batch dim

        # Postprocess
        result_t = (output.squeeze(0) * 0.5) + 0.5  # [0, 1] range

        fps_counter += 1
        if fps_counter % 60 == 0:
            elapsed = time.time() - start_time
            print(f"FPS: {fps_counter / elapsed:.1f}")
```

**Performance**: 60 FPS achievable at 512x512; throughput is limited by model inference, not IPC overhead. See [docs/BENCHMARKS.md](BENCHMARKS.md) for IPC timings.

> **Tip (R1 GPU-side wait):** Set `CUDALINK_TORCH_GPU_WAIT=1` (or `CUDALINK_TORCH_GPU_WAIT_ADAPTIVE=1`
> for auto-enable) to replace the host-side IPC event sync with a GPU-resident `cudaStreamWaitEvent`.
> Gain: ~50–200 µs/frame depending on WDDM batch timing; safe alongside CUDA Graphs.

---

## Example 2: TouchDesigner → OpenCV Processing (Numpy)

> ▶ Runnable: [examples/04_td_to_opencv_numpy.py](../examples/04_td_to_opencv_numpy.py)

### Use Case
Traditional computer vision (edge detection, feature tracking, etc.) on TouchDesigner feed.

### TouchDesigner Setup

Same as Example 1, but use a different `Ipcmemname`:

```
Camera TOP → CUDAIPCExtension (Mode=Sender)
             (Ipcmemname="cv_input")
```

### Python Script (OpenCV Processing)

```python
import cv2
from cuda_link import Importer, ImportSpec, ImportOutcome

with Importer.open(ImportSpec(
    shm_name="cv_input",
    shape=(720, 1280, 4),  # 720p RGBA
    dtype="uint8",         # OpenCV expects uint8
)) as importer:

    while True:
        result = importer.get_frame_numpy()

        if result.outcome is ImportOutcome.SHUTDOWN:
            break
        if result.outcome is not ImportOutcome.NEW_FRAME:
            continue

        frame = result.frame  # numpy.ndarray (720, 1280, 4), uint8

        # Convert RGBA → BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Apply edge detection
        edges = cv2.Canny(bgr, 100, 200)

        # Display
        cv2.imshow("Edges", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
```

**Performance**: 60 FPS achievable at 720p; IPC is not the bottleneck. See [docs/BENCHMARKS.md](BENCHMARKS.md) for IPC timings.

---

## Example 3: Multiple Texture Streams (Main + ControlNet)

> ▶ Runnable: [examples/05_multi_stream_and_dynamic_resolution.py](../examples/05_multi_stream_and_dynamic_resolution.py) `--part a`

### Use Case
AI pipeline with two inputs: main image + control signal (depth map, edges, etc.).

### TouchDesigner Setup

```
Camera TOP → CUDAIPCExtension (Mode=Sender, Ipcmemname="main_input")

Edge Detection TOP → CUDAIPCExtension (Mode=Sender, Ipcmemname="controlnet_input")
```

### Python Script (Dual-Stream AI)

```python
import torch
from cuda_link import Importer, ImportSpec, ImportOutcome

# Initialize both importers
main_importer = Importer.open(ImportSpec(
    shm_name="main_input",
    shape=(512, 512, 4),
    dtype="float32",
))
cn_importer = Importer.open(ImportSpec(
    shm_name="controlnet_input",
    shape=(512, 512, 4),
    dtype="float32",
))

# Load ControlNet model
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny").to("cuda")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
).to("cuda")

try:
    while True:
        main_result = main_importer.get_frame()
        cn_result = cn_importer.get_frame()

        if (main_result.outcome is ImportOutcome.SHUTDOWN
                or cn_result.outcome is ImportOutcome.SHUTDOWN):
            break
        if (main_result.outcome is not ImportOutcome.NEW_FRAME
                or cn_result.outcome is not ImportOutcome.NEW_FRAME):
            continue  # one stream idle or reconnecting

        main_frame = main_result.frame  # torch.Tensor on GPU
        cn_frame = cn_result.frame

        # Preprocess
        main_rgb = main_frame[:, :, :3]
        cn_rgb = cn_frame[:, :, :3]

        # Run ControlNet inference
        with torch.no_grad():
            output = pipe(
                prompt="high quality, detailed",
                image=main_rgb,
                control_image=cn_rgb,
                num_inference_steps=20,
            ).images[0]

        output.save("controlnet_output.png")
finally:
    main_importer.close()
    cn_importer.close()
```

**Performance**: ~2-5 FPS (limited by Stable Diffusion inference, not IPC).

---

## Example 4: Dynamic Resolution Handling

> ▶ Runnable: [examples/05_multi_stream_and_dynamic_resolution.py](../examples/05_multi_stream_and_dynamic_resolution.py) `--part b`

### Use Case
Source TOP resolution changes at runtime (user resizes window, switches camera, etc.).

### TouchDesigner Setup

```
Select TOP → CUDAIPCExtension (Mode=Sender)
(Resolution changes dynamically based on Select TOP input)
```

### Python Script (Auto-Reinitialize)

```python
from cuda_link import Importer, ImportSpec, ImportOutcome

# shape=None lets the importer auto-detect resolution from the sender's SHM
with Importer.open(ImportSpec(shm_name="dynamic_input")) as importer:

    frame_count = 0

    while True:
        result = importer.get_frame()

        if result.outcome is ImportOutcome.SHUTDOWN:
            print("Producer shut down — exiting")
            break
        if result.outcome is ImportOutcome.RECONNECTING:
            print("Format/resolution change detected — re-initializing")
            continue
        if result.outcome is not ImportOutcome.NEW_FRAME:
            continue  # Idle — no new frame yet

        frame = result.frame  # torch.Tensor on GPU
        print(f"Frame {frame_count}: shape={frame.shape}")
        frame_count += 1
```

**Note**: The importer **automatically detects** version changes and re-opens IPC handles.
During re-initialization it returns `ImportOutcome.RECONNECTING` for one or more frames;
steady-state resumes at `NEW_FRAME` with the new shape automatically.

---

## Example 5: Graceful Shutdown Handling

> ▶ Runnable: [examples/06_graceful_shutdown_and_reconnect.py](../examples/06_graceful_shutdown_and_reconnect.py)

### Use Case
Cleanly shut down Python process when TouchDesigner exits.

### TouchDesigner Setup

Ensure the `CUDAIPCExtension` COMP's Active parameter triggers shutdown signalling on exit
(built-in — the component writes the shutdown flag when `Active` is toggled Off or on TD exit).

### Python Script (Shutdown Detection)

```python
import signal
import sys
from cuda_link import Importer, ImportSpec, ImportOutcome

importer = Importer.open(ImportSpec(
    shm_name="clean_shutdown",
    shape=(512, 512, 4),
    dtype="float32",
))

# Register Ctrl+C handler
def signal_handler(sig, frame):
    print("Ctrl+C detected, cleaning up...")
    importer.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main loop
try:
    while True:
        result = importer.get_frame()

        if result.outcome is ImportOutcome.SHUTDOWN:
            print("Producer shutdown detected (TD exited)")
            break  # Exit gracefully

        if result.outcome is not ImportOutcome.NEW_FRAME:
            continue

        frame = result.frame
        # ... process frame ...
finally:
    importer.close()

print("Clean shutdown complete")
```

**Key**: The importer **automatically detects** the shutdown flag and returns
`ImportOutcome.SHUTDOWN` from `get_frame()`. No polling or manual SHM reading needed.

---

## Example 6: Benchmarking IPC Performance

> ▶ Runnable: [examples/07_python_to_td_exporter_and_benchmark.py](../examples/07_python_to_td_exporter_and_benchmark.py) (producer-side percentiles)

### Use Case
Measure IPC overhead for your specific hardware.

> **Recommended**: For a full IPC roundtrip sweep with statistical rigor (avg, p50, p95, p99,
> CSV + JSON export), run `python benchmarks/bench_sweep.py` (full 16-cell, ~12 min) or
> `python benchmarks/bench_sweep.py --quick` (1 cell, ~1 min). See
> [docs/BENCHMARKS.md](BENCHMARKS.md) for pre-measured results.

The manual script below is useful for quick ad-hoc profiling of the consumer side against a
live TD sender.

### Python Script

```python
import time
import statistics
from cuda_link import Importer, ImportSpec, ImportOutcome

with Importer.open(ImportSpec(
    shm_name="benchmark",
    shape=(1080, 1920, 4),
    dtype="float32",
)) as importer:

    # Warmup
    for _ in range(30):
        importer.get_frame()

    # Benchmark
    frame_times = []
    for _ in range(500):
        start = time.perf_counter()
        result = importer.get_frame()
        elapsed = (time.perf_counter() - start) * 1_000_000  # µs
        if result.outcome is ImportOutcome.NEW_FRAME:
            frame_times.append(elapsed)

    # Statistics
    print(f"IPC get_frame() overhead ({len(frame_times)} frames):")
    print(f"  Mean:   {statistics.mean(frame_times):.1f} µs")
    print(f"  Median: {statistics.median(frame_times):.1f} µs")
    print(f"  P95:    {sorted(frame_times)[int(len(frame_times)*0.95)]:.1f} µs")
    print(f"  P99:    {sorted(frame_times)[int(len(frame_times)*0.99)]:.1f} µs")
```

**Expected output** (PyTorch zero-copy mode, `get_frame()`, 1080p):

The call returns a pre-mapped GPU tensor; overhead is the ring-buffer poll and `cudaStreamWaitEvent` enqueue. Reference timings: IPC notify p50 ~136 µs at 1080p f32 (see [docs/BENCHMARKS.md](BENCHMARKS.md)).

**Expected output** (numpy D2H copy mode, `get_frame_numpy()`, 1080p float32):

Dominated by GPU-to-CPU D2H transfer. `get_frame_numpy()` p50 ~5000 µs at 1080p f32 (two-process context); ~1320 µs isolated. See [docs/BENCHMARKS.md](BENCHMARKS.md) for full tables.

---

## Example 7: Python → TouchDesigner (AI Output Display)

> ▶ Runnable: [examples/07_python_to_td_exporter_and_benchmark.py](../examples/07_python_to_td_exporter_and_benchmark.py)

### Use Case

AI pipeline generates frames (e.g., diffusion model output) in Python and sends them back to
TouchDesigner for display, compositing, or recording. This is the **reverse direction** — Python
is the producer, TD is the consumer.

### Python Side (Producer: `Exporter`)

```python
import torch
from cuda_link import Exporter, FrameSpec, GpuFrame

HEIGHT, WIDTH = 512, 512

# Initialize exporter once (at startup)
exporter = Exporter.open(
    FrameSpec(
        shm_name="ai_output_ipc",  # Must match TD Receiver's Ipcmemname parameter
        height=HEIGHT,
        width=WIDTH,
        channels=4,                # BGRA/RGBA
        dtype="uint8",             # uint8 is typical for display
        num_slots=2,               # Double-buffer (default)
    )
)

# AI inference loop
while running:
    # --- Your model generates output_tensor: (H, W, 4) uint8 BGRA on GPU ---
    with torch.no_grad():
        output_tensor = model(input_frame)  # shape: (512, 512, 4), dtype=uint8, on CUDA

    # Export to TD.
    # producer_stream arms ordering on the non-blocking IPC stream —
    # omitting it risks torn/gray frames when kernels run on a different stream.
    exporter.export(GpuFrame(
        ptr=output_tensor.data_ptr(),
        size=output_tensor.nelement() * output_tensor.element_size(),
        producer_stream=torch.cuda.current_stream().cuda_stream,
    ))

exporter.close()
```

Or use it as a context manager for automatic cleanup:

```python
with Exporter.open(FrameSpec(shm_name="ai_output_ipc", height=512, width=512)) as exporter:
    while running:
        exporter.export(GpuFrame(
            ptr=tensor.data_ptr(),
            size=tensor.nbytes,
            producer_stream=torch.cuda.current_stream().cuda_stream,
        ))
```

> **Python `Exporter` export-sync default:** The Python `Exporter` API defaults to **async** export
> (`CUDALINK_EXPORT_SYNC=0`) — safe when the source buffer is caller-owned and persistent (e.g. a
> fixed GPU tensor in your ring buffer). Set `CUDALINK_EXPORT_SYNC=1` to force blocking if your
> source pointer is transient.

### TouchDesigner Side (Consumer: `CUDAIPCExtension` in Receiver mode)

1. **Add `TOXES/CUDAIPCLink_v1.12.1.tox`** (or build from `td_exporter/CUDAIPCExtension.py`)
2. **Set Mode parameter** to `Receiver`
3. **Set `Ipcmemname`** to `"ai_output_ipc"` (must match Python's `shm_name`)
4. **Add a Script TOP** as the import target
5. **Wire**: Script TOP → your display chain

In the Script TOP's `onCook` callback:
```python
def onCook(scriptOp):
    ext.CUDAIPCExtension.import_frame(scriptOp)
```

**TouchDesigner Network**:
```
Script TOP (receives AI frames via IPC)
    → Composite TOP
    → Out TOP
```

### Performance

| Metric | Value | Notes |
|--------|-------|-------|
| `export_frame()` p50, 512×512 f32 (standalone, EXPORT_SYNC=1) | 24 µs | see [BENCHMARKS.md](BENCHMARKS.md) |
| `export_frame()` p50, 1080p f32 (standalone, EXPORT_SYNC=1) | 106 µs | see [BENCHMARKS.md](BENCHMARKS.md) |
| `export_frame()` p50, 1080p uint8, async+graphs (Python Exporter default) | 13.9 µs | −31.3 µs (69%) vs blocking 45.2 µs baseline |
| Max theoretical FPS (1080p f32, async+graphs) | ~71 000 FPS | D2D-only isolated |
| Practical FPS | Limited by AI model inference | |

Full IPC roundtrip timings (with concurrent consumer): [docs/BENCHMARKS.md](BENCHMARKS.md).

**Real-world example** (StreamDiffusion SDXL + ControlNet + V2V):
- AI inference: ~32ms/frame (31 FPS)
- IPC export overhead: small fraction of inference time
- TD display: 60 FPS locked (reads latest available frame)

---

## Example 8: Low-CPU Doorbell Consumer (R2)

> ▶ Runnable: [examples/08_low_cpu_doorbell_consumer.py](../examples/08_low_cpu_doorbell_consumer.py)

### Use Case

High-frequency TD → Python channel where the Python consumer must stay idle without burning CPU
between frames.  `CUDALINK_DOORBELL=1` replaces the poll-sleep with a blocking Win32 named-event
wait — measured CPU usage: ~3-4% → ~0.3-1% at 60 FPS.

### Requirements

- Windows only (Win32 `CreateEventW`)
- `CUDALINK_DOORBELL=1` set on **both** TD Sender and Python consumer before either process starts
- Single consumer per channel only

### TouchDesigner Setup

Set `CUDALINK_DOORBELL=1` in the environment before launching TD (system env or a launch wrapper),
or add it to TD's own `CUDALINK_*` env block so it propagates to child processes.

### Python Script (Doorbell Consumer)

```python
import os
os.environ["CUDALINK_DOORBELL"] = "1"  # must be set before importing cuda_link

from cuda_link import Importer, ImportSpec, ImportOutcome

with Importer.open(ImportSpec(
    shm_name="td_input",
    shape=(1080, 1920, 4),
    dtype="uint8",
)) as importer:

    while True:
        result = importer.get_frame()

        if result.outcome is ImportOutcome.SHUTDOWN:
            break

        if result.outcome is ImportOutcome.NO_FRAME:
            # Block until the producer fires SetEvent — replaces poll-sleep.
            # Returns immediately if a new frame arrived between get_frame() and this call
            # (lost-wakeup guard: re-checks write_idx before blocking).
            importer.wait_for_doorbell(2000.0)  # timeout in MILLISECONDS
            continue

        if result.outcome is ImportOutcome.RECONNECTING:
            continue

        # ImportOutcome.NEW_FRAME — process normally
        process(result.frame)
```

**Performance** (TD Sender → Python subprocess, 60 FPS):
- CPU idle: ~0.3-1% (vs ~3-4% poll baseline)
- Notify latency p95: 0.02–0.10 ms (vs 0.04–1.80 ms poll baseline), ~10× tighter
- Teardown: IPC handles close in ~0.6 ms (no 2 s hang, no orphaned handles)

> **Note**: `TDReceiver.py` (in-TD COMP on the cook thread) cannot use `wait_for_doorbell` — it
> runs in a blocking cook context and must return immediately on `NO_FRAME` by design.

---

## Common Patterns

### Pattern 1: Error Recovery

```python
import time
from cuda_link import Importer, ImportSpec, ImportOutcome

def make_importer():
    return Importer.open(ImportSpec(shm_name="my_channel", timeout_ms=5000.0))

importer = make_importer()

while True:
    try:
        result = importer.get_frame()

        if result.outcome is ImportOutcome.SHUTDOWN:
            break  # Producer shut down cleanly
        if result.outcome is not ImportOutcome.NEW_FRAME:
            continue  # Idle or reconnecting

        process(result.frame)

    except Exception as e:
        print(f"Error: {e}")
        importer.close()
        time.sleep(1)
        importer = make_importer()  # Reconnect
```

### Pattern 2: FPS Limiting (Consumer-Side)

```python
import time
from cuda_link import Importer, ImportSpec, ImportOutcome

target_fps = 30
frame_interval = 1.0 / target_fps

with Importer.open(ImportSpec(shm_name="my_channel")) as importer:
    while True:
        loop_start = time.time()

        result = importer.get_frame()
        if result.outcome is ImportOutcome.NEW_FRAME:
            process(result.frame)

        # Sleep to maintain target FPS
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
```

### Pattern 3: Consume Latest Frame Only

The importer's ring buffer already surfaces the **latest available frame** by design —
calling `get_frame()` when the producer has advanced several slots skips stale intermediate
frames automatically. No manual `write_idx` polling is needed.

```python
from cuda_link import Importer, ImportSpec, ImportOutcome

with Importer.open(ImportSpec(shm_name="my_channel")) as importer:
    while True:
        # Always returns the most recent frame the producer has written
        result = importer.get_frame()
        if result.outcome is ImportOutcome.NEW_FRAME:
            process(result.frame)
```

---

## Troubleshooting Common Issues

### Issue: "SharedMemory not found"

**Cause**: Python started before TD exporter initialized.

**Solution**: `ImportSpec.timeout_ms` (default 5 000 ms) provides automatic retry with
exponential backoff — the importer polls until the sender appears or the timeout elapses.
Increase `timeout_ms` if TD takes longer to initialize:

```python
importer = Importer.open(ImportSpec(shm_name="ai_input", timeout_ms=30_000.0))
```

### Issue: Black frames or stale data

**Cause**: TD's source TOP not cooking.

**Solution**: Check TD's performance monitor, ensure source TOP is active and has valid data.

### Issue: High latency (>50ms)

**Cause**: GPU sync fallback to CPU, or AI model bottleneck.

**Solution**:
1. Enable IPC events in TD (`Numslots`=3)
2. Profile AI model: `torch.cuda.synchronize()` before/after inference
3. Check GPU utilization: `nvidia-smi` should show ~90%+ for real-time AI

---

**Last Updated**: 2026-07-12
**Version**: 1.12.1
