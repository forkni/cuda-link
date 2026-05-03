"""
Round-trip integration tests: CUDAIPCExporter -> SharedMemory -> CUDAIPCImporter.

Tests the Python->TD code path: Python CUDAIPCExporter writes IPC handles to
SharedMemory; Python CUDAIPCImporter (standing in for TD Receiver) reads them.

Windows requires separate OS processes for CUDA IPC (cudaIpcOpenMemHandle returns
error 201 if called in the same process that created the handle). All tests here
use multiprocessing.Process with the "spawn" context.

NOTE on timing: The producer spawn process loads CUDA libraries and allocates GPU
buffers, which takes ~1-2 seconds on first start. Consumers therefore wait for
SharedMemory to appear using a cheap polling loop (no CUDA involvement) before
creating the full CUDAIPCImporter.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import time
from collections.abc import Callable

import pytest

# ---------------------------------------------------------------------------
# Shared helper: wait for SharedMemory to appear (cheap, no CUDA)
# ---------------------------------------------------------------------------


def _wait_for_shm(shm_name: str, timeout_s: float = 20.0) -> bool:
    """Poll until named SharedMemory exists. Returns True if found within timeout."""
    from multiprocessing.shared_memory import SharedMemory

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            return True
        except FileNotFoundError:
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Worker functions — all at module level so they can be pickled by spawn
# ---------------------------------------------------------------------------


def _worker_producer_basic(
    shm_name: str,
    height: int,
    width: int,
    channels: int,
    dtype: str,
    fill_value: int,
    num_frames: int,
    num_slots: int,
    rq: object,
) -> None:
    """Exports GPU frames via CUDAIPCExporter."""
    try:
        from cuda_link.cuda_ipc_exporter import CUDAIPCExporter
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

        cuda = get_cuda_runtime()
        exporter = CUDAIPCExporter(
            shm_name=shm_name,
            height=height,
            width=width,
            channels=channels,
            dtype=dtype,
            num_slots=num_slots,
            debug=False,
        )
        ok = exporter.initialize()
        if not ok:
            rq.put(("ERROR", "exporter.initialize() returned False"))
            return

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

        for _ in range(num_frames):
            exporter.export_frame(gpu_ptr=int(src_ptr.value), size=data_size)
            time.sleep(0.05)  # 50ms between frames

        time.sleep(2.0)  # Grace period so consumer can drain
        cuda.free(src_ptr)
        exporter.cleanup()
        rq.put(("OK", num_frames))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer_basic(
    shm_name: str, height: int, width: int, channels: int, dtype: str, num_frames: int, rq: object
) -> None:
    """Reads frames via CUDAIPCImporter."""
    try:
        from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, CUDAIPCImporter

        if not NUMPY_AVAILABLE:
            rq.put(("SKIP", "numpy not available"))
            return

        # Wait cheaply for producer's SharedMemory to appear (no CUDA needed yet)
        if not _wait_for_shm(shm_name, timeout_s=20.0):
            rq.put(("ERROR", f"SharedMemory '{shm_name}' never appeared within 20s"))
            return

        # Short extra delay so producer can write the CIPD magic and IPC handles
        time.sleep(0.3)

        importer = CUDAIPCImporter(shm_name=shm_name, shape=(height, width, channels), dtype=dtype)
        if not importer.is_ready():
            rq.put(("ERROR", "CUDAIPCImporter not ready after SharedMemory appeared"))
            return

        frames_received = 0
        deadline = time.perf_counter() + 20.0
        while frames_received < num_frames and time.perf_counter() < deadline:
            frame = importer.get_frame_numpy()
            if frame is not None:
                frames_received += 1
            else:
                time.sleep(0.01)

        importer.cleanup()
        rq.put(("OK", frames_received))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer_verify(
    shm_name: str, height: int, width: int, channels: int, dtype: str, expected_fill: int, num_frames: int, rq: object
) -> None:
    """Reads frames and verifies pixel values match expected fill."""
    try:
        import numpy as np

        from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, CUDAIPCImporter

        if not NUMPY_AVAILABLE:
            rq.put(("SKIP", "numpy not available"))
            return

        if not _wait_for_shm(shm_name, timeout_s=20.0):
            rq.put(("ERROR", f"SharedMemory '{shm_name}' never appeared"))
            return

        time.sleep(0.3)

        importer = CUDAIPCImporter(shm_name=shm_name, shape=(height, width, channels), dtype=dtype)
        if not importer.is_ready():
            rq.put(("ERROR", "CUDAIPCImporter not ready"))
            return

        frames_received = 0
        mismatches = 0
        deadline = time.perf_counter() + 20.0
        while frames_received < num_frames and time.perf_counter() < deadline:
            frame = importer.get_frame_numpy()
            if frame is not None:
                frames_received += 1
                if not np.all(frame == expected_fill):
                    mismatches += 1
            else:
                time.sleep(0.01)

        importer.cleanup()
        rq.put(("OK", {"frames": frames_received, "mismatches": mismatches}))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_producer_shutdown(shm_name: str, num_frames: int, rq: object) -> None:
    """Exports frames then cleanly shuts down (sets shutdown flag)."""
    try:
        from cuda_link.cuda_ipc_exporter import CUDAIPCExporter
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

        cuda = get_cuda_runtime()
        exporter = CUDAIPCExporter(shm_name=shm_name, height=8, width=8, channels=4, dtype="uint8", num_slots=2)
        exporter.initialize()

        src_ptr = cuda.malloc(exporter.data_size)
        for _ in range(num_frames):
            exporter.export_frame(gpu_ptr=int(src_ptr.value), size=exporter.data_size)
            time.sleep(0.05)

        # Stay alive long enough for consumer to connect, read at least 1 frame,
        # then set the shutdown flag via cleanup()
        time.sleep(4.0)
        cuda.free(src_ptr)
        exporter.cleanup()
        rq.put(("OK", num_frames))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer_shutdown(shm_name: str, rq: object) -> None:
    """Connects to live producer, reads frames, then waits for shutdown detection."""
    try:
        from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, CUDAIPCImporter

        if not NUMPY_AVAILABLE:
            rq.put(("SKIP", "numpy not available"))
            return

        if not _wait_for_shm(shm_name, timeout_s=20.0):
            rq.put(("ERROR", "SharedMemory never appeared"))
            return

        time.sleep(0.3)

        importer = CUDAIPCImporter(shm_name=shm_name, shape=(8, 8, 4), dtype="uint8")
        if not importer.is_ready():
            rq.put(("ERROR", "Importer not ready when connecting to live producer"))
            return

        # Poll for frames and watch for shutdown
        shutdown_detected = False
        poll_deadline = time.perf_counter() + 20.0
        while time.perf_counter() < poll_deadline:
            frame = importer.get_frame_numpy()
            if frame is None and not importer.is_ready():
                shutdown_detected = True
                break
            time.sleep(0.05)

        importer.cleanup()
        rq.put(("OK", {"shutdown_detected": shutdown_detected}))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_producer_float32(shm_name: str, num_frames: int, rq: object) -> None:
    """Exports float32 frames for metadata auto-detect test."""
    try:
        from cuda_link.cuda_ipc_exporter import CUDAIPCExporter
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

        cuda = get_cuda_runtime()
        exporter = CUDAIPCExporter(shm_name=shm_name, height=8, width=8, channels=4, dtype="float32", num_slots=2)
        exporter.initialize()

        src_ptr = cuda.malloc(exporter.data_size)
        for _ in range(num_frames):
            exporter.export_frame(gpu_ptr=int(src_ptr.value), size=exporter.data_size)
            time.sleep(0.05)

        time.sleep(2.0)
        cuda.free(src_ptr)
        exporter.cleanup()
        rq.put(("OK", num_frames))
    except Exception as e:  # noqa: BLE001
        import traceback

        rq.put(("ERROR", f"{e}\n{traceback.format_exc()}"))


def _worker_consumer_auto_detect(shm_name: str, rq: object) -> None:
    """Connects with shape=None to test metadata auto-detection."""
    try:
        from cuda_link.cuda_ipc_importer import NUMPY_AVAILABLE, CUDAIPCImporter

        if not NUMPY_AVAILABLE:
            rq.put(("SKIP", "numpy not available"))
            return

        if not _wait_for_shm(shm_name, timeout_s=20.0):
            rq.put(("ERROR", "SharedMemory never appeared"))
            return

        time.sleep(0.3)

        importer = CUDAIPCImporter(shm_name=shm_name, shape=None, dtype=None)
        if not importer.is_ready():
            rq.put(("ERROR", "Importer with shape=None not ready"))
            return

        detected_shape = importer.shape
        detected_dtype = importer.dtype

        frame = importer.get_frame_numpy()
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


# ---------------------------------------------------------------------------
# Helper to launch and collect results
# ---------------------------------------------------------------------------


def _run_pair(
    producer_target: Callable,
    producer_args: tuple,
    consumer_target: Callable,
    consumer_args: tuple,
    timeout_s: float = 45.0,
) -> list:
    """Launch producer + consumer as separate spawn processes, return collected results."""
    ctx = multiprocessing.get_context("spawn")
    rq = ctx.Queue()

    producer = ctx.Process(target=producer_target, args=producer_args + (rq,))
    consumer = ctx.Process(target=consumer_target, args=consumer_args + (rq,))

    # Both start together — consumer uses _wait_for_shm() to handle timing
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_exporter_importer(temp_shm_name: str) -> None:
    """CUDAIPCExporter (producer) sends GPU frames; CUDAIPCImporter reads them."""
    num_frames = 5

    results = _run_pair(
        producer_target=_worker_producer_basic,
        producer_args=(temp_shm_name, 8, 8, 4, "uint8", 42, num_frames, 2),
        consumer_target=_worker_consumer_basic,
        consumer_args=(temp_shm_name, 8, 8, 4, "uint8", num_frames),
    )

    skips = [r for r in results if r[0] == "SKIP"]
    if skips:
        pytest.skip(skips[0][1])

    errors = [r for r in results if r[0] == "ERROR"]
    assert not errors, f"Process errors: {errors}"

    ok_results = [r for r in results if r[0] == "OK"]
    assert ok_results, "No OK results received from any process"

    # The consumer reports its frame count as an int
    consumer_oks = [r[1] for r in ok_results if isinstance(r[1], int)]
    if consumer_oks:
        assert consumer_oks[-1] >= 1, f"Consumer received 0 frames (expected ≥1 of {num_frames})"


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_data_integrity(temp_shm_name: str) -> None:
    """GPU data written by exporter is correctly received by importer (fill pattern check)."""
    num_frames = 3
    fill_value = 0xAB  # 171 decimal

    results = _run_pair(
        producer_target=_worker_producer_basic,
        producer_args=(temp_shm_name, 8, 8, 4, "uint8", fill_value, num_frames, 2),
        consumer_target=_worker_consumer_verify,
        consumer_args=(temp_shm_name, 8, 8, 4, "uint8", fill_value, num_frames),
    )

    skips = [r for r in results if r[0] == "SKIP"]
    if skips:
        pytest.skip(skips[0][1])

    errors = [r for r in results if r[0] == "ERROR"]
    assert not errors, f"Process errors: {errors}"

    ok_dicts = [r for r in results if r[0] == "OK" and isinstance(r[1], dict)]
    if ok_dicts:
        stats = ok_dicts[-1][1]
        assert stats["frames"] >= 1, "Consumer received 0 frames"
        assert stats["mismatches"] == 0, (
            f"Data integrity failure: {stats['mismatches']}/{stats['frames']} frames had pixel values ≠ {fill_value}"
        )


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_shutdown_detection(temp_shm_name: str) -> None:
    """CUDAIPCImporter detects shutdown flag set by CUDAIPCExporter.cleanup()."""
    results = _run_pair(
        producer_target=_worker_producer_shutdown,
        producer_args=(temp_shm_name, 5),
        consumer_target=_worker_consumer_shutdown,
        consumer_args=(temp_shm_name,),
        timeout_s=45.0,
    )

    skips = [r for r in results if r[0] == "SKIP"]
    if skips:
        pytest.skip(skips[0][1])

    errors = [r for r in results if r[0] == "ERROR"]
    assert not errors, f"Process errors: {errors}"

    ok_dicts = [r for r in results if r[0] == "OK" and isinstance(r[1], dict)]
    if ok_dicts:
        assert ok_dicts[-1][1]["shutdown_detected"], "Consumer did not detect producer shutdown"


@pytest.mark.requires_cuda
@pytest.mark.slow
def test_roundtrip_metadata_auto_detect(temp_shm_name: str) -> None:
    """CUDAIPCImporter with shape=None correctly auto-detects shape/dtype from metadata."""
    results = _run_pair(
        producer_target=_worker_producer_float32,
        producer_args=(temp_shm_name, 5),
        consumer_target=_worker_consumer_auto_detect,
        consumer_args=(temp_shm_name,),
    )

    skips = [r for r in results if r[0] == "SKIP"]
    if skips:
        pytest.skip(skips[0][1])

    errors = [r for r in results if r[0] == "ERROR"]
    assert not errors, f"Process errors: {errors}"

    ok_dicts = [r for r in results if r[0] == "OK" and isinstance(r[1], dict)]
    if ok_dicts:
        data = ok_dicts[-1][1]
        assert data["shape"] == (8, 8, 4), f"Expected (8,8,4), got {data['shape']}"
        assert data["dtype"] == "float32", f"Expected 'float32', got {data['dtype']}"
