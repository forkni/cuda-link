"""
pytest configuration and shared fixtures for CUDA IPC tests.
"""

from __future__ import annotations

import time as _time
import uuid
from collections.abc import Generator
from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

# Fake doubles — available via pythonpath = ["td_exporter"] in pyproject.toml
from _td_fakes import FakeCUDAMemoryRef, FakeTDHost, FakeTOPHandle  # noqa: F401


@pytest.fixture
def cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA is available

    Raises:
        pytest.skip if CUDA runtime not available
    """
    try:
        from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

        CUDARuntimeAPI()
        return True
    except (RuntimeError, OSError) as e:
        pytest.skip(f"CUDA runtime not available: {e}")


@pytest.fixture
def cuda_runtime(cuda_available: bool) -> CUDARuntimeAPI:
    """Provide initialized CUDA runtime.

    Args:
        cuda_available: Fixture ensuring CUDA is present

    Returns:
        CUDARuntimeAPI instance
    """
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

    return get_cuda_runtime()


@pytest.fixture
def shared_memory_cleanup() -> Generator[list[str], None, None]:
    """Track and cleanup SharedMemory objects after test.

    Yields:
        List to append SharedMemory names to

    Cleanup is performed automatically after test completes.
    """
    names = []
    yield names

    # Cleanup
    from multiprocessing.shared_memory import SharedMemory

    for name in names:
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass  # Already cleaned up


@pytest.fixture
def temp_shm_name() -> str:
    """Generate a unique SharedMemory name for testing.

    Returns:
        Unique SharedMemory name string
    """
    return f"test_cuda_ipc_{uuid.uuid4().hex[:8]}"


# FakeTDHost / FakeTOPHandle / FakeCUDAMemoryRef imported from _td_fakes above.
# They are re-exported here so existing tests that do
#   ``from conftest import FakeTDHost, FakeTOPHandle``
# continue to work without change.


# ---------------------------------------------------------------------------
# FakeShmAdapter — satisfies BarrierShmPort structurally; no real SHM needed
# ---------------------------------------------------------------------------


@_dataclass
class FakeShmAdapter:
    """In-process fake satisfying BarrierShmPort (structural).

    Tests construct the scenario they want and pass it as `shm=` to
    CheckerBarrier.  No real SharedMemory is touched.

    Use `raise_on_*` fields to simulate I/O failures at specific callpoints.
    `last_change_ns` defaults to None; `attach()` sets it to time.monotonic_ns()
    on first call, simulating a freshly-written segment.  Pass an explicit
    value to simulate a stale segment (e.g. ``last_change_ns=0``).
    """

    active_count: int = 0
    last_change_ns: int | None = None
    barrier_skips: int = 0
    attached: bool = False
    raise_on_attach: type[Exception] | None = None
    raise_on_read: type[Exception] | None = None
    raise_on_bump: type[Exception] | None = None

    @property
    def is_attached(self) -> bool:
        return self.attached

    def attach(self, *, create: bool) -> None:
        if self.raise_on_attach is not None:
            raise self.raise_on_attach("simulated")
        self.attached = True
        if self.last_change_ns is None:
            self.last_change_ns = _time.monotonic_ns()

    def read_state(self) -> tuple[int, int, int]:
        if self.raise_on_read is not None:
            raise self.raise_on_read("simulated")
        return (self.active_count, self.last_change_ns or 0, self.barrier_skips)

    def bump_skip(self) -> None:
        if self.raise_on_bump is not None:
            raise self.raise_on_bump("simulated")
        self.barrier_skips += 1

    def close(self) -> None:
        self.attached = False

    # --- HolderShmPort methods (satisfies HolderBarrier seam structurally) ---

    def open_and_increment(self, pid: int) -> int:
        self.attached = True
        if self.last_change_ns is None:
            self.last_change_ns = _time.monotonic_ns()
        self.active_count += 1
        return self.active_count

    def decrement(self, pid: int) -> int:
        self.active_count = max(0, self.active_count - 1)
        return self.active_count
