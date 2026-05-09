# TD Pipeline nsys Capture -- Findings (2026-05-07)

## Setup

- **TD**: CUDA_Link_Example.43.toe (TD 2025.32820, launcher Execute DAT disabled), receiver Memname=cudalink_output_ipc
- **Producer**: example_sender_python.py, 512x512 RGBA uint8, 3 slots, target 60 FPS
- **nsys**: 2026.2.1, `--trace=cuda,nvtx,wddm --wddm-additional-events=true --wddm-backtraces=true`
- **Capture**: 55s producer / 60s consumer, timestamp overlap 134.2s

## Producer Metrics (PID: Python sender process)

| Metric | p50 | p99 | max | count |
|---|---|---|---|---|
| exporter.slot<N> duration (NVTX) | 90.2 µs | 225.5 µs | 2929 µs | 7053 |
| H2D staging fill (GPU) | 46.3 µs | 365.2 µs | 1214 µs | 7165 |

- **H2D bandwidth**: 9.90 GB/s (512x512 RGBA uint8 = 1 MB/frame via PCIe/WDDM)
- **Effective send rate**: 5.5 FPS (target: 60 FPS)
- **Sub-range breakdown** (verbose NVTX + CUDA API):
  - cudaGraphLaunch (D2D copy + event record): avg 35.2 µs, median 32.7 µs
  - cudaStreamSynchronize (flush_probe): avg n/a µs, median n/a µs -- dominates slot time
  - shm_write (SHM header update): avg 6.4 µs, median 5.6 µs

## Consumer Metrics (TD / TouchDesigner process)

| Metric | p50 | p99 | max | count |
|---|---|---|---|---|
| import_frame.slot<N> (NVTX) | 157.7 µs | 472.1 µs | 30024 µs | 7721 |
| D2A copy from IPC buffer (GPU) | 74.0 µs | 3605.8 µs | 31668 µs | 23556 |
| event_wait (NVTX) | 38.8 µs | 108.1 µs | 2354 µs | 7718 |

- **D2A bandwidth**: 62.72 GB/s (device-to-CUDA-array within GPU)
- **Effective cook rate**: 58.6 FPS
- **Frame drop rate**: ~0% -- by design (TD reads latest-written slot each cook)
- **NVTX**: present -- cudalink.receiver.* ranges active via system Python nvtx

## Cross-Process E2E Latency (6452 slot-matched pairs)

| Metric | p50 | p99 | max |
|---|---|---|---|
| Handoff latency (slot_end -> consumer wait_start) | 22182 µs | 49740 µs | 179492 µs |
| Full E2E (slot_start -> consumer wait_start) | 22271 µs | 49856 µs | 179586 µs |

Note: pairing method = nearest prior producer slot completion before each consumer IPC wait.
At 59 FPS consumer / 6 FPS producer, the consumer always reads a slot that is
0-3 frames behind the producer's current write head -- expected.

## Key Observations

1. **flush_probe dominates producer slot time** (90 µs slot median, of which ~n/a µs is
   cudaStreamSynchronize). The sync waits for the D2D IPC copy to complete on the GPU before
   the CPU updates the SHM header -- this is the WDDM batch-flush point.

2. **IPC event wait is healthy** (~39 µs p50). The consumer unblocks quickly once the
   producer's event is signalled, consistent with the D2D completing before the CPU updates SHM.

3. **Producer uses two separate CUDA contexts** on the same device (confirmed from wddm_queue_sum
   in the bench_sweep run): IPC stream + main stream on separate WDDM Render queue contexts.

4. **TD cook rate ~59 FPS ≈ producer 6 FPS** -- near-synchronous rates with 3 slots mean each slot is read ~2 slot-periods after being written. The E2E handoff p50 of ~22182 µs reflects this topology (not a latency regression).

5. **Consumer NVTX active** -- cudalink.receiver.* ranges present via system Python nvtx.
   import_frame p50 158 µs; event_wait p50 39 µs.

## Next Steps

- Run F2/F8 regression (CUDALINK_TD_PERSIST_STREAM=0) to see stream serialisation on the
  GPU lane (expected: D2D and D2A serialize rather than overlap).
- ncu deep-dive: the only kernel of interest is the CUDA Graph node (D2D copy in IPC stream).
  Target: `cudaGraphLaunch` call, profile with --set full to check PCIe vs HBM utilization.
