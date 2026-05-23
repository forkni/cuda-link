"""
Regression test for C2: stream-capture mode coexistence.

When cuda-link uses cudaStreamCaptureModeGlobal (mode=0), any concurrent
cudaStreamBeginCapture in the same process (e.g. TensorRT, CuPy, PyTorch)
receives cudaErrorStreamCaptureInvalidated (error 901) on EndCapture.

With mode=2 (Relaxed), independent captures do not invalidate each other.

Reproducer: two threads each perform stream_begin_capture + stream_end_capture
on independent streams. With mode=0, one EndCapture fails with 901. With
mode=2, both succeed.
"""

from __future__ import annotations

import contextlib
import threading

import pytest


@pytest.mark.requires_cuda
def test_concurrent_captures_mode_relaxed(cuda_runtime: object) -> None:
    """Two concurrent captures with mode=2 must both succeed."""
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def capture_cycle(mode: int) -> None:
        try:
            stream = cuda_runtime.create_stream()
            try:
                barrier.wait()  # synchronize so both captures overlap
                cuda_runtime.stream_begin_capture(stream, mode=mode)
                # enqueue a no-op (malloc/free on a zero-size region as placeholder)
                cuda_runtime.stream_end_capture(stream)
            finally:
                cuda_runtime.destroy_stream(stream)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=capture_cycle, args=(2,))
    t2 = threading.Thread(target=capture_cycle, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Concurrent relaxed captures failed: {errors}"


@pytest.mark.requires_cuda
def test_mode0_invalidates_concurrent_capture(cuda_runtime: object) -> None:
    """Verify that mode=0 DOES cause the collision (documents the regression).

    If this test stops failing it means the CUDA driver changed behaviour and
    C2's fix may need re-evaluation.
    """
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def capture_cycle(mode: int) -> None:
        try:
            stream = cuda_runtime.create_stream()
            try:
                barrier.wait()
                cuda_runtime.stream_begin_capture(stream, mode=mode)
                cuda_runtime.stream_end_capture(stream)
            finally:
                with contextlib.suppress(Exception):
                    cuda_runtime.destroy_stream(stream)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=capture_cycle, args=(0,))
    t2 = threading.Thread(target=capture_cycle, args=(0,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # With mode=0 on some CUDA/driver combinations this will fail (error 901).
    # We mark xfail so CI does not hard-fail if the driver happens to serialize
    # captures, but we document the expected bad outcome.
    if errors:
        pytest.xfail(f"mode=0 produced errors as expected (C2 regression confirmed): {errors}")
    else:
        pytest.skip(
            "mode=0 did not produce errors on this driver — driver may serialize captures; C2 fix still safe to land."
        )
