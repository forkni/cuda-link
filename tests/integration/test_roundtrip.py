"""
Round-trip integration tests: Exporter -> SharedMemory -> CUDAIPCImporter.

Tests the Python->TD code path: Python Exporter writes IPC handles to
SharedMemory; Python CUDAIPCImporter (standing in for TD Receiver) reads them.

Windows requires separate OS processes for CUDA IPC (cudaIpcOpenMemHandle returns
error 201 if called in the same process that created the handle). All tests here
use multiprocessing.Process with the "spawn" context.

NOTE on timing: producer and consumer run in lockstep through multiprocessing Events
(see _Sync), never through fixed sleeps. The producer signals `ready` once its exporter
is open, then waits for the consumer's `frame_read` after every export, so the consumer
sees each frame no matter how long its own CUDA start-up takes.

NOTE on isolation: each producer points cuda_link.activation_barrier.SHM_NAME at a
per-test segment before opening its exporter. The production segment is shared with
any live TD session, and a TD sender holding it would make these exports skip.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import time
from collections.abc import Callable
from typing import NamedTuple

import pytest

pytest.importorskip("numpy")

_SYNC_TIMEOUT_S = 20.0  # upper bound on any single wait; a healthy pair never comes near it

# ---------------------------------------------------------------------------
# Synchronisation between the two spawned processes
# ---------------------------------------------------------------------------


class _Sync(NamedTuple):
    """Events shared by one producer/consumer pair (pickled into both at spawn)."""

    ready: object  # producer: exporter is open, IPC handles are in SharedMemory
    frame_read: object  # consumer: the latest exported frame was read
    done: object  # consumer: finished, the producer may close


def _barrier_name(shm_name: str) -> str:
    return f"{shm_name}_barrier"


# ---------------------------------------------------------------------------
# Worker functions — all at module level so they can be pickled by spawn
# ---------------------------------------------------------------------------


def _worker_producer(
    shm_name: str,
    dtype: str,
    fill_value: int,
    num_frames: int,
    wait_for_done: bool,
    sync: _Sync,
    rq: object,
) -> None:
    """Exports num_frames 8x8x4 GPU frames, one per consumer read.

    With wait_for_done the exporter stays open until the consumer reports it is finished.
    Without it, close() follows the last read at once: that is the shutdown under test.
    """
    try:
        import cuda_link.activation_barrier as activation_barrier
        from cuda_link import FrameSpec, GpuFrame
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
        from cuda_link.exporter import Exporter

        activation_barrier.SHM_NAME = _barrier_name(shm_name)

        cuda = get_cuda_runtime()
        exporter = Exporter.open(FrameSpec(shm_name=shm_name, height=8, width=8, channels=4, dtype=dtype, num_slots=2))

        data_size = exporter.data_size
        src_ptr = cuda.malloc(data_size)

        # Fill source buffer with known pattern via H2D copy
        host_buf = (ctypes.c_uint8 * data_size)(*([fill_value] * data_size))
        cuda.memcpy(
            dst=src_ptr,
            src=ctypes.c_void_p(ctypes.addressof(host_buf)),
            count=data_size,
            kind=1,  # cudaMemcpyHostToDevice
        )

        sync.ready.set()
        for i in range(num_frames):
            exporter.export(GpuFrame(ptr=int(src_ptr.value), size=data_size))
            if not sync.frame_read.wait(_SYNC_TIMEOUT_S):
                raise TimeoutError(f"consumer did not read frame {i + 1} of {num_frames}")
            sync.frame_read.clear()

        if wait_for_done and not sync.done.wait(_SYNC_TIMEOUT_S):
            raise TimeoutError("consumer never reported done")

        cuda.free(src_ptr)
        exporter.close()
        rq.put(("OK", num_frames))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _connect_importer(shm_name: str, shape: tuple | None, dtype: str | None, sync: _Sync) -> object:
    """Waits for the producer's exporter, then connects a CUDAIPCImporter to it."""
    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    if not sync.ready.wait(_SYNC_TIMEOUT_S):
        raise TimeoutError(f"producer never opened '{shm_name}'")

    importer = CUDAIPCImporter(shm_name=shm_name, shape=shape, dtype=dtype)
    importer.connect()
    if not importer.is_ready():
        raise RuntimeError("CUDAIPCImporter not ready after connect()")
    return importer


def _worker_consumer_frames(
    shm_name: str, dtype: str, expected_fill: int, num_frames: int, sync: _Sync, rq: object
) -> None:
    """Reads num_frames frames and counts those whose pixels differ from expected_fill."""
    try:
        import numpy as np

        importer = _connect_importer(shm_name, (8, 8, 4), dtype, sync)

        frames_received = 0
        mismatches = 0
        deadline = time.perf_counter() + _SYNC_TIMEOUT_S
        while frames_received < num_frames and time.perf_counter() < deadline:
            frame = importer.get_frame_numpy()
            if frame is None:
                time.sleep(0.001)
                continue
            frames_received += 1
            if not np.all(frame == expected_fill):
                mismatches += 1
            sync.frame_read.set()

        importer.cleanup()
        rq.put(("OK", {"frames": frames_received, "mismatches": mismatches}))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))
    finally:
        sync.done.set()


def _worker_consumer_shutdown(shm_name: str, sync: _Sync, rq: object) -> None:
    """Reads frames from a live producer until it sees the producer shut down."""
    try:
        importer = _connect_importer(shm_name, (8, 8, 4), "uint8", sync)

        frames_received = 0
        shutdown_detected = False
        deadline = time.perf_counter() + _SYNC_TIMEOUT_S
        while time.perf_counter() < deadline:
            frame = importer.get_frame_numpy()
            if frame is not None:
                frames_received += 1
                sync.frame_read.set()
            elif not importer.is_ready():
                shutdown_detected = True
                break
            else:
                time.sleep(0.001)

        importer.cleanup()
        rq.put(("OK", {"frames": frames_received, "shutdown_detected": shutdown_detected}))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))
    finally:
        sync.done.set()


def _worker_consumer_auto_detect(shm_name: str, sync: _Sync, rq: object) -> None:
    """Connects with shape=None and dtype=None to test metadata auto-detection."""
    try:
        importer = _connect_importer(shm_name, None, None, sync)

        detected_shape = importer.shape
        detected_dtype = importer.dtype

        frame = None
        deadline = time.perf_counter() + _SYNC_TIMEOUT_S
        while frame is None and time.perf_counter() < deadline:
            frame = importer.get_frame_numpy()
            if frame is None:
                time.sleep(0.001)
        if frame is not None:
            sync.frame_read.set()

        importer.cleanup()
        rq.put(
            (
                "OK",
                {
                    "shape": detected_shape,
                    "dtype": detected_dtype,
                    "got_frame": frame is not None,
                },
            )
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))
    finally:
        sync.done.set()


# ---------------------------------------------------------------------------
# Helpers to launch and collect results
# ---------------------------------------------------------------------------


def _run_pair(
    producer_args: tuple,
    consumer_target: Callable,
    consumer_args: tuple,
    timeout_s: float = 45.0,
) -> list:
    """Launch producer + consumer as separate spawn processes, return collected results."""
    ctx = multiprocessing.get_context("spawn")
    rq = ctx.Queue()
    sync = _Sync(ready=ctx.Event(), frame_read=ctx.Event(), done=ctx.Event())

    producer = ctx.Process(target=_worker_producer, args=producer_args + (sync, rq))
    consumer = ctx.Process(target=consumer_target, args=consumer_args + (sync, rq))

    producer.start()
    consumer.start()

    producer.join(timeout=timeout_s)
    consumer.join(timeout=timeout_s)

    if producer.is_alive():
        producer.terminate()
        pytest.fail("Producer process timed out")
    if consumer.is_alive():
        consumer.terminate()
        pytest.fail("Consumer process timed out")

    results = []
    while not rq.empty():
        results.append(rq.get_nowait())
    return results


def _consumer_result(results: list, num_frames: int) -> dict:
    """Checks both processes succeeded and returns the consumer's result dict."""
    errors = [r for r in results if r[0] == "ERROR"]
    assert not errors, f"Process errors: {errors}"
    assert ("OK", num_frames) in results, f"Producer did not finish all {num_frames} frames: {results}"

    consumer = [r[1] for r in results if r[0] == "OK" and isinstance(r[1], dict)]
    assert len(consumer) == 1, f"Expected one consumer result, got: {results}"
    return consumer[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_exporter_importer(temp_shm_name: str) -> None:
    """Every frame the Exporter sends through a 2-slot ring reaches CUDAIPCImporter."""
    num_frames = 5

    results = _run_pair(
        producer_args=(temp_shm_name, "uint8", 42, num_frames, True),
        consumer_target=_worker_consumer_frames,
        consumer_args=(temp_shm_name, "uint8", 42, num_frames),
    )

    stats = _consumer_result(results, num_frames)
    assert stats["frames"] == num_frames


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_data_integrity(temp_shm_name: str) -> None:
    """GPU data written by exporter is correctly received by importer (fill pattern check)."""
    num_frames = 3
    fill_value = 0xAB  # 171 decimal

    results = _run_pair(
        producer_args=(temp_shm_name, "uint8", fill_value, num_frames, True),
        consumer_target=_worker_consumer_frames,
        consumer_args=(temp_shm_name, "uint8", fill_value, num_frames),
    )

    stats = _consumer_result(results, num_frames)
    assert stats["frames"] == num_frames
    assert stats["mismatches"] == 0, (
        f"Data integrity failure: {stats['mismatches']}/{stats['frames']} frames had pixel values ≠ {fill_value}"
    )


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_shutdown_detection(temp_shm_name: str) -> None:
    """CUDAIPCImporter detects shutdown flag set by Exporter.close()."""
    num_frames = 2

    results = _run_pair(
        producer_args=(temp_shm_name, "uint8", 0, num_frames, False),
        consumer_target=_worker_consumer_shutdown,
        consumer_args=(temp_shm_name,),
    )

    stats = _consumer_result(results, num_frames)
    assert stats["frames"] == num_frames
    assert stats["shutdown_detected"], "Consumer did not detect producer shutdown"


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_metadata_auto_detect(temp_shm_name: str) -> None:
    """CUDAIPCImporter with shape=None correctly auto-detects shape/dtype from metadata."""
    results = _run_pair(
        producer_args=(temp_shm_name, "float32", 0, 1, True),
        consumer_target=_worker_consumer_auto_detect,
        consumer_args=(temp_shm_name,),
    )

    data = _consumer_result(results, 1)
    assert data["shape"] == (8, 8, 4), f"Expected (8,8,4), got {data['shape']}"
    assert data["dtype"] == "float32", f"Expected 'float32', got {data['dtype']}"
    assert data["got_frame"], "Auto-detected importer never received a frame"
