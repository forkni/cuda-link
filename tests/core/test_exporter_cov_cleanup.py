"""
Coverage tests for Exporter._do_cleanup()'s 7-step shutdown branches, plus the
context-manager / is_ready() / __del__ surface.

All tests use FakeCUDAAdapter — no GPU required. Tests that leave a real
SharedMemory segment un-unlinked (by design, to exercise a failure path) use
the shared_memory_cleanup fixture as a safety net so nothing leaks.
"""

from __future__ import annotations

import gc
import time
import uuid
from collections.abc import Generator
from unittest.mock import patch

import pytest

from cuda_link import ExportPolicy, FrameSpec
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.exporter import Exporter

_H = _W = _C = 4
_DATA_SIZE = _H * _W * _C


def _spec(*, num_slots: int = 1) -> FrameSpec:
    return FrameSpec(
        shm_name=f"test_cov_cleanup_{uuid.uuid4().hex[:8]}",
        height=_H,
        width=_W,
        channels=_C,
        dtype="uint8",
        num_slots=num_slots,
        device=0,
    )


# ---------------------------------------------------------------------------
# _is_cuda_context_valid() exception (768-769) cascades through _do_cleanup's
# GPU-gated steps (779, 801->806, 810->815, 815->821, 821->827, 833->837, 837->859)
# ---------------------------------------------------------------------------


def test_do_cleanup_cuda_context_invalid_skips_gpu_teardown_steps() -> None:
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(num_slots=2), policy=ExportPolicy.for_testing(), cuda=fake)
    with (
        patch.object(fake, "peek_last_error", side_effect=RuntimeError("simulated destroyed context")),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp.close()
    assert exp._closed is True
    messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
    assert any("CUDA context already destroyed" in m for m in messages)


# ---------------------------------------------------------------------------
# _do_cleanup() — early return when never initialized (775-776)
# ---------------------------------------------------------------------------


def test_do_cleanup_early_return_when_never_initialized() -> None:
    exp = Exporter(_spec(), ExportPolicy.for_testing(), FakeCUDAAdapter())
    exp.close()
    assert exp._closed is True
    assert exp._initialized is False


# ---------------------------------------------------------------------------
# _do_cleanup() — shm_handle=None mid-cleanup skips STEP1 and STEP4 (782->801, 827->833)
# ---------------------------------------------------------------------------


def test_do_cleanup_shm_handle_none_mid_cleanup_skips_shm_steps() -> None:
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=fake)
    exp.shm_handle = None  # simulate an already-released SHM handle
    exp.close()  # STEP7 still unlinks the real segment by name regardless
    assert exp._closed is True


# ---------------------------------------------------------------------------
# _do_cleanup() STEP1 — set_shutdown()/zero-write failures are logged, not raised
# (786-787, 795-796)
# ---------------------------------------------------------------------------


class _RaisingBuf:
    def __setitem__(self, key: object, value: object) -> None:
        raise OSError("simulated SHM write failure")


class _FakeShmHandle:
    def __init__(self) -> None:
        self.buf = _RaisingBuf()

    def close(self) -> None:
        pass


def test_do_cleanup_shm_write_failures_are_logged_not_raised() -> None:
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=fake)
    original_shm = exp.shm_handle
    exp.shm_handle = _FakeShmHandle()
    original_shm.close()  # release the real OS handle; STEP7 still unlinks by name

    with patch("cuda_link.exporter.logger") as mock_logger:
        exp.close()

    messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
    assert any("Could not write shutdown signal" in m for m in messages)
    assert any("Could not zero IPC handles" in m for m in messages)


# ---------------------------------------------------------------------------
# _do_cleanup() STEP6 — a None dev_ptrs entry is skipped in the free loop (840->839)
# ---------------------------------------------------------------------------


def test_do_cleanup_dev_ptrs_none_entry_skipped_in_free_loop() -> None:
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(num_slots=2), policy=ExportPolicy.for_testing(), cuda=fake)
    exp.dev_ptrs[0] = None  # simulate a slot whose allocation was already freed
    exp.close()
    assert exp._closed is True


# ---------------------------------------------------------------------------
# _do_cleanup() STEP6 — cudaFree thread exceeding the 0.5s deadline logs a warning (856)
# ---------------------------------------------------------------------------


def _slow_free(_ptr: object) -> None:
    time.sleep(0.7)  # comfortably longer than _do_cleanup's 0.5s join deadline


def test_do_cleanup_free_thread_timeout_logs_warning() -> None:
    fake = FakeCUDAAdapter(device=0)
    exp = Exporter.open(_spec(num_slots=1), policy=ExportPolicy.for_testing(), cuda=fake)
    with (
        patch.object(fake, "free", side_effect=_slow_free),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp.close()
    messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
    assert any("timed out" in m for m in messages)


# ---------------------------------------------------------------------------
# _do_cleanup() STEP7 — unlink failure (not FileNotFoundError) is logged (866-867)
# ---------------------------------------------------------------------------


def test_do_cleanup_unlink_failure_logs_warning(shared_memory_cleanup: list[str]) -> None:
    fake = FakeCUDAAdapter(device=0)
    spec = _spec()
    shared_memory_cleanup.append(spec.shm_name)  # safety net: our patch prevents the real unlink
    exp = Exporter.open(spec, policy=ExportPolicy.for_testing(), cuda=fake)

    with (
        patch("cuda_link.exporter.SharedMemory", side_effect=PermissionError("simulated: access denied")),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp.close()

    messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
    assert any("Could not unlink SharedMemory" in m for m in messages)


# ---------------------------------------------------------------------------
# Context manager (__enter__/__exit__) + is_ready() (890, 893, 905)
# ---------------------------------------------------------------------------


def test_context_manager_and_is_ready() -> None:
    fake = FakeCUDAAdapter(device=0)
    with Exporter.open(_spec(), policy=ExportPolicy.for_testing(), cuda=fake) as exp:
        assert exp.is_ready() is True
    assert exp._closed is True


# ---------------------------------------------------------------------------
# __del__ closes an unclosed Exporter (897)
# ---------------------------------------------------------------------------


@pytest.fixture
def _gc_isolated() -> Generator[None, None, None]:
    """Force a collection before/after so __del__ timing is deterministic."""
    gc.collect()
    yield
    gc.collect()


def test_del_closes_unclosed_exporter(shared_memory_cleanup: list[str], _gc_isolated: None) -> None:
    fake = FakeCUDAAdapter(device=0)
    spec = _spec()
    shared_memory_cleanup.append(spec.shm_name)  # safety net in case __del__ timing is unreliable
    exp = Exporter.open(spec, policy=ExportPolicy.for_testing(), cuda=fake)
    assert exp._closed is False

    with patch("cuda_link.exporter.logger") as mock_logger:
        del exp
        gc.collect()

    messages = [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]
    assert any("Exporter closed" in m for m in messages)
