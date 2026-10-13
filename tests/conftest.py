"""
pytest configuration and shared fixtures for CUDA IPC tests.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import uuid
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

# Ensure CuPy disk kernel cache uses system temp dir to prevent sandbox permission errors on ~/.cupy
os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "cupy_cache"))

if TYPE_CHECKING:
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI


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


@pytest.fixture
def isolated_barrier_shm(monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    """Redirect the activation barrier to a unique, per-test SHM segment.

    The production name is fixed; on Windows unlink() is a no-op and segment
    lifetime is handle-bound, so a leaked handle from an earlier test (or a
    live TD session) blocks create=True on the shared name. A unique name per
    test cannot collide with anything else in or outside this process.

    Yields:
        The per-test SHM segment name now bound to
        cuda_link.activation_barrier.SHM_NAME.
    """
    from multiprocessing.shared_memory import SharedMemory

    import cuda_link.activation_barrier as ab

    name = f"cudalink_barrier_test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(ab, "SHM_NAME", name)
    yield name

    with contextlib.suppress(FileNotFoundError):
        shm = SharedMemory(name=name)
        shm.close()
        shm.unlink()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Enforce quarantine marker discipline at collection time.

    A test marked @pytest.mark.quarantine must specify reason=..., issue=...,
    and expires="YYYY-MM-DD". If expires has passed, collection fails.
    """
    import datetime as dt

    today = dt.date.today()
    bad: list[str] = []
    for item in items:
        m = item.get_closest_marker("quarantine")
        if m is None:
            continue
        kw = m.kwargs
        if not kw.get("reason") or not kw.get("issue") or "expires" not in kw:
            bad.append(f"{item.nodeid}: quarantine marker requires reason=, issue=, and expires=")
        elif dt.date.fromisoformat(str(kw["expires"])) < today:
            bad.append(f"{item.nodeid}: quarantine expired on {kw['expires']} ({kw['issue']})")
    if bad:
        raise pytest.UsageError("\n".join(bad))
