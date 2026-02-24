# Integration Examples

Complete workflows for common CUDA IPC use cases.

---

## Example 1: TouchDesigner → PyTorch AI Pipeline (Zero-Copy)

### Use Case
Real-time AI inference (style transfer, object detection, etc.) on TouchDesigner video feed.

### TouchDesigner Setup

1. **Network Layout**:
```
Movie File In TOP → CUDAIPCExporter
                    (Ipcmemname="ai_input")
```

2. **Parameters**:
   - `Ipcmemname`: `"ai_input"`
   - `Active`: ON
   - `Numslots`: 3
   - Resolution: 512x512 (or your model's input size)

### Python Script (AI Inference Loop)

```python
import torch
from cuda_link import CUDAIPCImporter

# Initialize importer
importer = CUDAIPCImporter(
    shm_name="ai_input",
    shape=(512, 512, 4),  # RGBA
    dtype="float32",
    debug=False
)

# Load your AI model
model = torch.jit.load("style_transfer_model.pt").cuda()
model.eval()

# Inference loop
fps_counter = 0
import time
start_time = time.time()

while True:
    if not importer.is_ready():
        break

    # Get frame (zero-copy, < 5μs)
    input_tensor = importer.get_frame()  # Shape: (512, 512, 4)

    # Preprocess (convert RGBA → RGB, normalize)
    rgb = input_tensor[:, :, :3]  # Drop alpha
    normalized = (rgb - 0.5) / 0.5  # [-1, 1] range

    # Run inference
    with torch.no_grad():
        output = model(normalized.unsqueeze(0))  # Add batch dim

    # Postprocess (denormalize, add alpha channel back)
    result = (output.squeeze(0) * 0.5) + 0.5  # [0, 1] range

    # Save result (or send back to TD via another IPC channel)
    # torch.save(result, "output.pt")  # Example

    fps_counter += 1
    if fps_counter % 60 == 0:
        elapsed = time.time() - start_time
        print(f"FPS: {fps_counter / elapsed:.1f}")

importer.cleanup()
```

**Performance**: ~60 FPS at 512x512, ~25 FPS at 1080p (model-dependent).

---

## Example 2: TouchDesigner → OpenCV Processing (Numpy)

### Use Case
Traditional computer vision (edge detection, feature tracking, etc.) on TouchDesigner feed.

### TouchDesigner Setup

Same as Example 1, but use different `Ipcmemname`:

```
Camera TOP → CUDAIPCExporter
             (Ipcmemname="cv_input")
```

### Python Script (OpenCV Processing)

```python
import cv2
import numpy as np
from cuda_link import CUDAIPCImporter

# Initialize importer
importer = CUDAIPCImporter(
    shm_name="cv_input",
    shape=(720, 1280, 4),  # 720p RGBA
    dtype="uint8",         # OpenCV expects uint8
    debug=False
)

while True:
    if not importer.is_ready():
        break

    # Get frame as numpy array (D2H copy, varies by resolution — see README performance table)
    frame = importer.get_frame_numpy()  # Shape: (720, 1280, 4), uint8

    # Convert RGBA → BGR for OpenCV
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Apply edge detection
    edges = cv2.Canny(bgr, 100, 200)

    # Display
    cv2.imshow("Edges", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
importer.cleanup()
```

**Performance**: ~60 FPS at 720p (OpenCV Canny is fast).

---

## Example 3: Multiple Texture Streams (Main + ControlNet)

### Use Case
AI pipeline with two inputs: main image + control signal (depth map, edges, etc.).

### TouchDesigner Setup

```
Camera TOP → CUDAIPCExporter_Main
             (Ipcmemname="main_input")

Edge Detection TOP → CUDAIPCExporter_CN
                     (Ipcmemname="controlnet_input")
```

### Python Script (Dual-Stream AI)

```python
import torch
from cuda_link import CUDAIPCImporter

# Initialize both importers
main_importer = CUDAIPCImporter(
    shm_name="main_input",
    shape=(512, 512, 4),
    dtype="float32"
)

cn_importer = CUDAIPCImporter(
    shm_name="controlnet_input",
    shape=(512, 512, 4),
    dtype="float32"
)

# Load ControlNet model
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny").to("cuda")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
).to("cuda")

while True:
    if not main_importer.is_ready() or not cn_importer.is_ready():
        break

    # Get both frames (parallel, zero-copy)
    main_frame = main_importer.get_frame()
    cn_frame = cn_importer.get_frame()

    # Preprocess
    main_rgb = main_frame[:, :, :3]
    cn_rgb = cn_frame[:, :, :3]

    # Run ControlNet inference
    with torch.no_grad():
        output = pipe(
            prompt="high quality, detailed",
            image=main_rgb,
            control_image=cn_rgb,
            num_inference_steps=20
        ).images[0]

    # Save or display result
    output.save("controlnet_output.png")

main_importer.cleanup()
cn_importer.cleanup()
```

**Performance**: ~2-5 FPS (limited by Stable Diffusion inference, not IPC).

---

## Example 4: Dynamic Resolution Handling

### Use Case
Source TOP resolution changes at runtime (user resizes window, switches camera, etc.).

### TouchDesigner Setup

```
Select TOP → CUDAIPCExporter
(Resolution changes dynamically based on Select TOP input)
```

### Python Script (Auto-Reinitialize)

```python
from cuda_link import CUDAIPCImporter

# Start with initial resolution
importer = CUDAIPCImporter(
    shm_name="dynamic_input",
    shape=(720, 1280, 4),  # Initial guess
    dtype="float32",
    debug=True
)

frame_count = 0

while True:
    if not importer.is_ready():
        break

    frame = importer.get_frame()

    # Check if resolution changed (version changed triggers auto-reinit)
    if frame is None:
        print("Resolution changed, importer auto-reinitialized")
        continue

    # Process frame
    print(f"Frame {frame_count}: shape={frame.shape}")
    frame_count += 1

importer.cleanup()
```

**Note**: The importer **automatically detects** version changes and re-opens IPC handles. No manual code needed.

---

## Example 5: Graceful Shutdown Handling

### Use Case
Cleanly shut down Python process when TouchDesigner exits.

### TouchDesigner Setup

Ensure `CUDAIPCExporter` has `callbacks` Execute DAT with `onExit()` defined (see TOX Build Guide).

### Python Script (Shutdown Detection)

```python
from cuda_link import CUDAIPCImporter
import signal
import sys

# Initialize importer
importer = CUDAIPCImporter(
    shm_name="clean_shutdown",
    shape=(512, 512, 4),
    dtype="float32"
)

# Register signal handlers
def signal_handler(sig, frame):
    print("Ctrl+C detected, cleaning up...")
    importer.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main loop
while True:
    frame = importer.get_frame()

    if frame is None:
        print("Producer shutdown detected (TD exited)")
        break  # Exit gracefully

    # Process frame
    # ...

# Automatic cleanup via shutdown flag detection
print("Clean shutdown complete")
```

**Key**: The importer **automatically detects** the shutdown flag at byte 596 (for 3 slots: `20 + 3×192`) and returns `None` from `get_frame()`.

---

## Example 6: Benchmarking IPC Performance

### Use Case
Measure IPC overhead for your specific hardware.

### Python Script

```python
import time
from cuda_link import CUDAIPCImporter

importer = CUDAIPCImporter(
    shm_name="benchmark",
    shape=(1080, 1920, 4),
    dtype="float32",
    debug=False
)

# Warmup
for _ in range(100):
    importer.get_frame()

# Benchmark
frame_times = []
for _ in range(1000):
    start = time.perf_counter()
    frame = importer.get_frame()
    elapsed = (time.perf_counter() - start) * 1_000_000  # μs
    frame_times.append(elapsed)

# Statistics
import statistics
print(f"IPC get_frame() overhead:")
print(f"  Mean:   {statistics.mean(frame_times):.1f} μs")
print(f"  Median: {statistics.median(frame_times):.1f} μs")
print(f"  P95:    {sorted(frame_times)[int(len(frame_times)*0.95)]:.1f} μs")
print(f"  P99:    {sorted(frame_times)[int(len(frame_times)*0.99)]:.1f} μs")

importer.cleanup()
```

**Expected output** (good hardware, PyTorch zero-copy mode):
```
IPC get_frame() overhead:
  Mean:   3.5 μs
  Median: 2.8 μs
  P95:    6.0 μs
  P99:    9.0 μs
```

---

## Example 7: Python → TouchDesigner (AI Output Display)

### Use Case

AI pipeline generates frames (e.g., diffusion model output) in Python and sends them back to
TouchDesigner for display, compositing, or recording. This is the **reverse direction** — Python
is the producer, TD is the consumer.

### Python Side (Producer: `CUDAIPCExporter`)

```python
import torch
from cuda_link import CUDAIPCExporter

HEIGHT, WIDTH = 512, 512

# Initialize exporter once (at startup)
exporter = CUDAIPCExporter(
    shm_name="ai_output_ipc",   # Must match TD Receiver's Ipcmemname parameter
    height=HEIGHT,
    width=WIDTH,
    channels=4,                 # BGRA/RGBA
    dtype="uint8",              # uint8 is typical for display
    num_slots=2,                # Double-buffer (default)
    debug=False,
)
exporter.initialize()

# AI inference loop
while running:
    # --- Your model generates output_tensor: (H, W, 4) uint8 BGRA on GPU ---
    with torch.no_grad():
        output_tensor = model(input_frame)  # shape: (512, 512, 4), dtype=uint8, on CUDA

    # Export to TD: ~10-20μs overhead at 512x512 (async D2D memcpy + event record)
    exporter.export_frame(
        gpu_ptr=output_tensor.data_ptr(),
        size=output_tensor.nelement() * output_tensor.element_size(),
    )

exporter.cleanup()
```

Or use it as a context manager for automatic cleanup:

```python
with CUDAIPCExporter(shm_name="ai_output_ipc", height=512, width=512) as exporter:
    exporter.initialize()
    while running:
        exporter.export_frame(gpu_ptr=tensor.data_ptr(), size=tensor.nbytes)
```

### TouchDesigner Side (Consumer: `CUDAIPCExtension` in Receiver mode)

1. **Add the CUDAIPCExporter TOX** (or build from `td_exporter/CUDAIPCExtension.py`)
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

| Metric | Value |
|--------|-------|
| Python export overhead | ~10-20μs per frame (512x512) |
| TD import overhead | <5μs per frame |
| Total IPC overhead | ~15-25μs per frame (512x512) |
| Maximum theoretical FPS | ~10,000 (IPC-limited) |
| Practical FPS | Limited by AI model inference |

**Real-world example** (StreamDiffusion SDXL + ControlNet + V2V):
- AI inference: ~32ms/frame (31 FPS)
- IPC export: ~20μs per frame (0.06% overhead)
- TD display: 60 FPS locked (reads latest available frame)

---

## Common Patterns

### Pattern 1: Error Recovery

```python
while True:
    try:
        frame = importer.get_frame()
        if frame is None:
            break  # Producer shutdown

        # Process frame
        process(frame)

    except Exception as e:
        print(f"Error: {e}")
        # Attempt recovery
        importer.cleanup()
        time.sleep(1)
        importer = CUDAIPCImporter(...)  # Reinitialize
```

### Pattern 2: FPS Limiting (Consumer-Side)

```python
import time

target_fps = 30
frame_interval = 1.0 / target_fps

while True:
    loop_start = time.time()

    frame = importer.get_frame()
    # Process frame...

    # Sleep to maintain target FPS
    elapsed = time.time() - loop_start
    sleep_time = frame_interval - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
```

### Pattern 3: Frame Dropping (Consume Latest Only)

```python
# Read current write_idx, always use the latest frame
import struct

while True:
    # Get latest write_idx
    write_idx = struct.unpack("<I", bytes(importer.shm_handle.buf[12:16]))[0]

    # Always read most recent slot
    if write_idx > 0:
        latest_slot = (write_idx - 1) % importer.num_slots
        frame = importer.tensors[latest_slot]  # Skip intermediate frames

        # Process frame...
```

---

## Troubleshooting Common Issues

### Issue: "SharedMemory not found"

**Cause**: Python started before TD exporter initialized.

**Solution**:
```python
import time
from multiprocessing.shared_memory import SharedMemory

# Retry logic
max_retries = 10
for i in range(max_retries):
    try:
        shm = SharedMemory(name="ai_input")
        print("✓ Connected")
        break
    except FileNotFoundError:
        print(f"Waiting for TD... ({i+1}/{max_retries})")
        time.sleep(0.5)
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

**Last Updated**: 2026-02-09
**Version**: 1.0.0
