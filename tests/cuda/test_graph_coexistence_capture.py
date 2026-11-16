"""
Regression test for C2: stream-capture mode coexistence.

When cuda-link uses cudaStreamCaptureModeGlobal (mode=0), any concurrent
cudaStreamBeginCapture in the same process (e.g. TensorRT, CuPy, PyTorch)
receives cudaErrorStreamCaptureInvalidated (error 901) on EndCapture.

With mode=2 (Relaxed), independent captures do not invalidate each other.

Reproducer: two threads each perform stream_begin_capture + stream_end_capture
on independent streams. With mode=0, one EndCapture fails with 901. With
mode=2, both succeed.

Only the mode=2 half is tested. A mode=0 companion used to sit here as a
strict xfail that skipped whenever the driver serialized the captures. It
could only xfail or skip, never fail, so it pinned nothing and was removed
in H2. Whether mode=0 collides depends on the driver; the contract
cuda-link owns is that mode=2 never does.
"""

from __future__ import annotations

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
