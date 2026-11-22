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

# Hypothesis profiles for the property tests (TESTING_GUIDE.md, Phase J4). "ci" is Hypothesis's own
# derandomized profile (derandomize=True, deadline=None, no database), re-registered so the name is
# ours; cosmic-ray runs pass --hypothesis-profile=ci so a mutant gets the same examples every run.
# The default follows the CI variable, so a CI job gets "ci" without setting HYPOTHESIS_PROFILE.
# Guarded: the requires_native CI legs install only pytest + pytest-randomly and still load this file.
try:
    from hypothesis import settings as _hypothesis_settings
except ImportError:  # pragma: no cover -- dev extra always installs it
    _hypothesis_settings = None
if _hypothesis_settings is not None:
    _hypothesis_settings.register_profile("ci", _hypothesis_settings.get_profile("ci"))
    _hypothesis_settings.register_profile("dev", max_examples=50)
    _hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci" if os.environ.get("CI") else "dev"))

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


# Modules whose receiver compares the segment's sender_pid with _current_pid().
_SAME_PROCESS_CHECK_MODULES = ("cuda_link.importer", "Importer", "TDReceiver")


@pytest.fixture(autouse=True)
def _allow_in_process_peer(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn off the v0.6 same-process check for the in-process sender/receiver tests.

    Most tests run an Exporter and an Importer (or TDReceiver) in one process on fake CUDA,
    where the loop-back works. Real CUDA IPC cannot loop back, so production keeps the
    check. A test marked ``same_process_check`` keeps the real check.
    """
    if request.node.get_closest_marker("same_process_check") is not None:
        return
    import sys

    for name in _SAME_PROCESS_CHECK_MODULES:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "_current_pid"):
            monkeypatch.setattr(module, "_current_pid", lambda: 0)


@pytest.fixture(autouse=True)
def _isolate_native_tops_record(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Point the .tox Info page at an empty Documents folder, never the real one.

    CUDAIPCExtension reads <Documents>/Derivative/Plugins/CudaLink/VERSION.txt on init. On a
    dev box with C++ TOPs of another version installed, that would add a standing warning to
    every extension test. A test that needs a record patches native_tops_version_file.
    tests/td and tests/integration import the module here first, because some of their tests
    import it inside the test body, after this fixture has run.
    """
    import importlib
    import sys

    module = sys.modules.get("CUDAIPCExtension")
    if module is None and {"td", "integration"} & set(request.node.path.parts):
        with contextlib.suppress(ImportError):
            module = importlib.import_module("CUDAIPCExtension")
    if module is not None and hasattr(module, "_documents_dir"):
        documents = tmp_path_factory.getbasetemp() / "no-documents"  # never created: no record
        monkeypatch.setattr(module, "_documents_dir", lambda: documents)
