# Quick Benchmark Guide: CUDA IPC Testing

## Overview

This guide shows how to test the CUDA IPC texture transfer performance using TouchDesigner as the sender and Python as the consumer.

**Workflow**: Two separate processes
- **TD sender**: `benchmark.toe` (you create) → sends GPU textures via CUDA IPC
- **Python consumer**: `benchmark_comparison.py` (from cmd) → receives and measures latency

---

## Step 1: Create benchmark.toe in TouchDesigner

### Minimal Setup

1. **Create a new .toe file** → save as `benchmark.toe`

2. **Add source texture**:
   - Create a **Noise TOP** (or any test pattern)
   - Set resolution: 1920x1080 (start with HD)
   - Set pixel format: RGBA 8-bit

3. **Add CUDAIPCLink component**:
   - Option A: Import existing `.tox` from `TOXES/` folder (if built)
   - Option B: Copy/paste from `CUDA_IPC_Comp.2.toe` main project
   - Option C: Build manually following `docs/TOX_BUILD_GUIDE.md`

4. **Connect**: `noise1` → `CUDAIPCLink`

5. **Configure CUDAIPCLink parameters**:
   ```
   Mode:       Sender
   Active:     ON          ← CRITICAL: must be ON to export frames
   Debug:      ON          ← Enables verbose performance logging (timestamps always written)
   Ipcmemname: cuda_ipc_handle  ← matches Python script default
   Numslots:   3
   ```

   **⚠️ IMPORTANT**: Verify `Active` is ON before running the benchmark. If Active=OFF, the sender is idle and SharedMemory won't be created.

6. **Verify sender is running**:
   - Check textport output: should see "Sender initialized" messages
   - Frame counter should increment every frame
   - No errors in textport

---

## Step 2: Run Python Benchmark Script

### Basic Usage

Open a command prompt and run:

```bash
cd C:\Users\INTER\Documents\INTER_TECH\COMPONENTS\CUDA_IPC
python benchmarks/benchmark_comparison.py --frames 600 --csv results_1080p.csv
```

**Expected output**:
```
============================================================
CUDA IPC Benchmark Comparison
============================================================
SharedMemory name: cuda_ipc_handle
Warmup frames: 60
Measurement frames: 600
============================================================

Initializing CUDAIPCImporter...
✅ Connected to sender (resolution: 1920x1080, format: 4 channels, dtype: float32)
   Ring buffer: 3 slots

Warming up (60 frames)...
✅ Warmup complete

Collecting measurements (600 frames)...
  Progress: 100/600 frames (17%)
  Progress: 200/600 frames (33%)
  ...
  Progress: 600/600 frames (100%)

✅ Raw data exported to: results_1080p.csv

============================================================
BENCHMARK RESULTS
============================================================

Configuration:
  shm_name: cuda_ipc_handle
  resolution: 1920x1080
  format: 4 channels
  dtype: float32
  num_slots: 3

Measurement Summary:
  Total frames requested: 600
  Successful frames: 600
  Frame skips detected: 0
  Total time: 10.05 seconds
  Average FPS: 59.7

End-to-End Latency (Producer Timestamp → Consumer):
  Average:     0.55 ms
  Median (p50): 0.56 ms
  p95:         1.05 ms
  p99:         1.17 ms
  Min:         0.01 ms
  Max:         1.22 ms

Frame Processing Time (get_frame() execution):
  Average:     0.12 ms (119 us)
  Median (p50): 0.02 ms (24 us)
  p95:         0.58 ms (576 us)
  p99:         0.64 ms (640 us)
  Min:         0.01 ms (12 us)
  Max:         0.78 ms (777 us)

============================================================

✅ Benchmark complete!
```

---

## Step 3: Test Retry Logic (Optional)

The script can wait for TD to start:

```bash
# Start Python FIRST (before TD)
python benchmarks/benchmark_comparison.py --max-retries 30 --retry-interval 1.0

# You'll see:
#   Waiting for TD sender... (1/30)
#   Waiting for TD sender... (2/30)
#   ...

# Now start TouchDesigner and open benchmark.toe → set Active=ON
# The Python script will connect automatically
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--frames` | 600 | Number of frames to measure |
| `--warmup` | 60 | Warmup frames before measurement |
| `--shm-name` | `cuda_ipc_handle` | SharedMemory name (must match TD parameter) |
| `--max-retries` | 30 | Max connection attempts to wait for TD |
| `--retry-interval` | 1.0 | Seconds between retry attempts |
| `--csv` | None | Export raw per-frame data to CSV file |

### Examples

**Quick test (100 frames)**:
```bash
python benchmarks/benchmark_comparison.py --frames 100 --warmup 10
```

**Long test with CSV export**:
```bash
python benchmarks/benchmark_comparison.py --frames 1000 --csv results.csv
```

**Custom SharedMemory name**:
```bash
python benchmarks/benchmark_comparison.py --shm-name my_cuda_ipc --frames 600
```

---

## Troubleshooting

### Issue: "Failed to initialize importer"

**Causes**:
1. TouchDesigner sender not running
2. SharedMemory name mismatch
3. Active=OFF in TD component

**Fix**:
- Verify TD is running and benchmark.toe is open
- Check `Ipcmemname` parameter matches `--shm-name` arg
- Ensure `Active=ON` in TD

### Issue: "End-to-end latency" shows 0.0 ms

**Cause**: Producer timestamp is zero (stale SharedMemory from previous session)

**Fix**: Restart the TD sender to ensure fresh SharedMemory initialization

### Issue: Frame skips detected

**Causes**:
1. Python consumer slower than TD producer (60 FPS)
2. GPU busy with other tasks

**Check**:
- Look at "Frame Processing Time" p99 value
- If p99 > 16ms, the consumer can't keep up with 60 FPS
- Try lowering TD FPS or increasing Python consumer efficiency

---

## CSV Output Format

When `--csv results.csv` is used, the script exports per-frame data:

| Column | Description |
|--------|-------------|
| `frame` | Frame number (1-based) |
| `end_to_end_latency_ms` | Producer timestamp → consumer (ms) |
| `frame_processing_time_ms` | `get_frame()` execution time (ms) |
| `write_idx` | Ring buffer write index at time of frame |

Use this for charting latency over time, detecting patterns, etc.

---

## Performance Targets

Expected results for 1920x1080 RGBA float32 at 60 FPS:

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| End-to-end latency (avg) | 0.5-1.5 ms | Includes TD framework overhead |
| Frame processing time (avg) | 0.02-0.15 ms (20-150us) | Pure Python hot-path |
| Frame skips | 0 | Should be zero at 60 FPS |
| Throughput | 60 FPS | Should match TD sender FPS |

If values differ significantly, check GPU load, other applications, and TD cook times.

---

## Next Steps

After verifying CUDA IPC works, you can:

1. **Test different resolutions**:
   ```bash
   # 4K test
   python benchmarks/benchmark_comparison.py --frames 600 --csv results_4k.csv
   ```
   Update TD source resolution to 3840x2160

2. **Test different formats**:
   - Change TD Noise TOP to RGBA 32-bit float
   - Re-run benchmark
   - Compare latency vs 8-bit

3. **Compare against TD-native solutions** (future work):
   - Memshare TOP (D2H + H2D double copy)
   - TouchOut/TouchIn TOP (TCP/IP)
   - See `.claude/plans/reflective-prancing-quail.md` for full comparison plan

---

## Files Reference

- **Benchmark script**: `benchmarks/benchmark_comparison.py`
- **Plan document**: `.claude/plans/reflective-prancing-quail.md`
- **Main project**: `CUDA_IPC_Comp.2.toe`
- **TOX build guide**: `docs/TOX_BUILD_GUIDE.md`
