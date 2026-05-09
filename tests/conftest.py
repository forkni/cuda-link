"""
pytest configuration and shared fixtures for CUDA IPC tests.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

# Prepend this repo's src/ ahead of any installed cuda_link package.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


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


# ---------------------------------------------------------------------------
# FakeTDHost / FakeTOPHandle — test doubles for the TDHost seam
# ---------------------------------------------------------------------------


class FakeCUDAMemoryRef:
    """Minimal CUDAMemoryRef-shaped object for tests that don't need the real type."""

    def __init__(
        self,
        ptr: int = 0xDEADBEEF,
        width: int = 64,
        height: int = 64,
        channels: int = 4,
        size: int = 0,
        data_type: Any = None,
    ) -> None:
        self.ptr = ptr
        self.width = width
        self.height = height
        self.channels = channels
        self.size = size or width * height * channels * 4
        self.data_type = data_type


class FakeTOPHandle:
    """In-process test double for TOPHandle.

    Stores calls so tests can assert on what was invoked.
    """

    def __init__(
        self,
        pixel_format: str = "rgba32float",
        width: int = 64,
        height: int = 64,
        channels: int = 4,
        gpu_ptr: int = 0xDEADBEEF,
        inputs: list[FakeTOPHandle] | None = None,
    ) -> None:
        self._pixel_format = pixel_format
        self._width = width
        self._height = height
        self._channels = channels
        self._gpu_ptr = gpu_ptr
        self._inputs = inputs or []
        self.format_set: list[str] = []
        self.copy_cuda_calls: list[tuple] = []
        self.copy_numpy_calls: list[Any] = []
        self.resolution_set: list[tuple[int, int]] = []

    def cuda_memory(self, stream: Any = None) -> FakeCUDAMemoryRef:
        size = self._width * self._height * self._channels * 4
        return FakeCUDAMemoryRef(
            ptr=self._gpu_ptr,
            width=self._width,
            height=self._height,
            channels=self._channels,
            size=size,
        )

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    @property
    def inputs(self) -> list[FakeTOPHandle]:
        return self._inputs

    def set_format(self, fmt: str) -> None:
        self.format_set.append(fmt)

    def copy_cuda_memory(self, ptr: int, size: int, shape: Any, *, stream: int) -> None:
        self.copy_cuda_calls.append((ptr, size, shape, stream))

    def copy_numpy_array(self, arr: Any) -> None:
        self.copy_numpy_calls.append(arr)

    def set_resolution(self, width: int, height: int) -> None:
        self.resolution_set.append((width, height))

    def is_valid(self) -> bool:
        return True


class FakeTDHost:
    """In-process test double for TDHost.

    Backed by a plain dict of parameter values; records all write calls.
    """

    def __init__(self, params: dict[str, Any] | None = None, tops: dict[str, FakeTOPHandle] | None = None) -> None:
        self._params: dict[str, Any] = dict(params or {})
        self._tops: dict[str, FakeTOPHandle] = dict(tops or {})
        self.param_writes: list[tuple[str, Any]] = []
        self.enable_writes: list[tuple[str, bool]] = []
        self.custom_only_calls: list[bool] = []

    def param_value(self, name: str) -> Any:
        return self._params.get(name)

    def set_param_value(self, name: str, value: Any) -> None:
        self._params[name] = value
        self.param_writes.append((name, value))

    def set_param_enabled(self, name: str, enabled: bool) -> None:
        self.enable_writes.append((name, enabled))

    def show_custom_only(self, value: bool) -> None:
        self.custom_only_calls.append(value)

    def is_active(self) -> bool:
        return bool(self._params.get("Active", True))

    def find_top(self, name: str) -> FakeTOPHandle | None:
        return self._tops.get(name)
