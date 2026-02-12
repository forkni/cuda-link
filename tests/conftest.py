"""
pytest configuration and shared fixtures for CUDA IPC tests.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

# Add project root and td_exporter to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "td_exporter"))


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_cuda: test requires CUDA GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow (multi-process, etc.)")


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
