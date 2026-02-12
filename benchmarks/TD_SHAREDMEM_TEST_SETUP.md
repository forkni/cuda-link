# TD Shared Mem Out TOP - Quick Test Setup

## Step 1: Create Test Project in TouchDesigner

1. **Create a new .toe file** or use your existing benchmark.toe

2. **Add a Noise TOP** (or use any existing source):
   - Create: Add Operator → TOP → Generator → Noise
   - Set resolution: 1920x1080 (or any size)
   - Set pixel format: RGBA 32-bit float

3. **Add Shared Mem Out TOP**:
   - Create: Add Operator → TOP → Export → Shared Mem Out
   - Connect: `noise1` → `sharedmemout1`
   - **Configure parameters**:
     ```
     Name:         test_shm
     Memtype:      Global
     Downloadtype: Delayed(Fast)
     ```
   - Make sure the Shared Mem Out TOP is **cooking** (check its viewer - should show the noise)

4. **Verify TD is running at 60 FPS**:
   - Check bottom-right corner of TD window
   - Should show "60 FPS" or similar

## Step 2: Run Python Test Script

Open a command prompt:

```bash
cd C:\Users\INTER\Documents\INTER_TECH\COMPONENTS\CUDA_IPC
python benchmarks/test_td_sharedmem_reader.py
```

### Expected Output

```
============================================================
TDSharedMemReader Test
============================================================
SharedMemory name: test_shm
Frames to read: 10
============================================================

Connecting to TD Shared Mem Out TOP...
[TDSharedMemReader] Opened mutex: TouchSHMtest_shmMutex
[TDSharedMemReader] Info segment not found (resize not supported)
[TDSharedMemReader] Opened data mapping: TouchSHMtest_shm
[TDSharedMemReader] Connected: 1920x1080, format=R32G32B32A32_SFLOAT, dtype=float32, channels=4
✅ Connected!

Metadata:
  Resolution: 1920x1080
  Pixel format: R32G32B32A32_SFLOAT (enum value: 109)
  NumPy dtype: float32
  Channels: 4
  Shape: (1080, 1920, 4)
  Data size: 33,177,600 bytes

Reading 10 frames...

Frame 1/10: shape=(1080, 1920, 4), dtype=float32, read_time=1.23ms
  First frame pixel stats: min=0.000, max=1.000, mean=0.501, std=0.289
Frame 2/10: shape=(1080, 1920, 4), dtype=float32, read_time=1.18ms
Frame 3/10: shape=(1080, 1920, 4), dtype=float32, read_time=1.21ms
...
Frame 10/10: shape=(1080, 1920, 4), dtype=float32, read_time=1.19ms

============================================================
Test Summary
============================================================
Successful reads: 10/10

Frame Read Times:
  Average: 1.20 ms
  Median:  1.19 ms
  Min:     1.15 ms
  Max:     1.25 ms
  Std:     0.03 ms

✅ Test complete!
```

### Expected Read Times

For 1920x1080 RGBA float32 (32 MB per frame):

| Operation | Expected Time |
|-----------|---------------|
| Mutex lock + read + unlock | ~1-2ms |
| Compare to CUDA IPC | ~750-1000x slower |

The D2H GPU readback has already happened inside TD (when Shared Mem Out TOP cooks), so this measures the CPU SharedMemory read + mutex overhead.

## Step 3: Test with Custom Name

```bash
# In TD: Change Shared Mem Out TOP "name" parameter to "my_test"
python benchmarks/test_td_sharedmem_reader.py --name my_test
```

## Troubleshooting

### Error: "Failed to connect"

**Check 1**: TD is running and .toe file is open
**Check 2**: Shared Mem Out TOP exists with matching name parameter
**Check 3**: Shared Mem Out TOP is **cooking** (not bypassed)
- Right-click the TOP → check that "Viewer Active" is ON
- The TOP should have a colored dot (cooking indicator)

**Check 4**: Name matches exactly (case-sensitive)
- TD parameter: `test_shm`
- Python script: `--name test_shm`

**Check 5**: Check Windows file mappings (advanced)
```cmd
# In PowerShell as Administrator:
Get-WmiObject -Class Win32_Process | Where-Object {$_.Name -eq "TouchDesigner.exe"}
```

### Error: "Mutex lock timeout"

The TD process is holding the mutex. This can happen if:
- TD is frozen/crashed (restart TD)
- Another Python process is already connected (close other scripts)

### Read returns None every time

This is expected if you're reading faster than TD produces frames (60 FPS).
The test script paces at ~60 FPS (`time.sleep(0.016)`), so you should get successful reads.

If all reads are None:
- Check that TD is actually running (not paused)
- Check TD FPS counter (bottom-right)

### Pixel values are all zeros or garbage

The Shared Mem Out TOP might not have written data yet. Wait a second and re-run the script.

## Next Steps

Once this test works, you can run the full benchmark:

```bash
# Full benchmark (600 frames)
python benchmarks/benchmark_sharedmem.py --frames 600 --warmup 60

# With CSV export
python benchmarks/benchmark_sharedmem.py --frames 600 --csv results.csv

# With timestamp channel for end-to-end latency (requires Execute DAT setup)
python benchmarks/benchmark_sharedmem.py --frames 600 --timestamp-shm cuda_ipc_benchmark_ts
```

### Optional: End-to-End Latency Measurement

To measure end-to-end latency (producer timestamp → consumer timestamp):

1. **Add Execute DAT in TouchDesigner**:
   - Create: Add Operator → DAT → Execute
   - Copy/paste the content from `td_exporter/benchmark_timestamp.py`
   - Set: DAT Execute → Active = ON
   - Set: Callbacks → onFrameEnd = ON (all others OFF)

2. **Run benchmark with timestamp channel**:
   ```bash
   python benchmarks/benchmark_sharedmem.py --frames 600 --timestamp-shm cuda_ipc_benchmark_ts
   ```

Without the timestamp channel, the benchmark will still measure frame processing time.
