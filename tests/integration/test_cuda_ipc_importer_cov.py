"""Coverage-focused tests for cuda_link.cuda_ipc_importer.CUDAIPCImporter.

test_cuda_ipc_importer.py and test_importer_deprecation.py already cover
connect() (idempotency, nonexistent-SHM waiting state — requires_cuda),
get_frame_numpy() (connected + shutdown-detection), get_stats() before
connect, and the deprecation-warning machinery. This file fills the
remaining gaps, all GPU-free:

  - connect()'s post-open shape/dtype sync from `_importer._format`
    (via a monkeypatched Importer.open — no real CUDA IPC needed).
  - get_frame() / get_frame_cupy() on both the "not connected" and
    "connected" paths (only get_frame()'s not-connected path was covered).
  - get_frame_numpy()'s "not connected" path specifically (previously only
    reachable when NUMPY_AVAILABLE is False, which skips on this machine).
  - get_stats() on the connected path.
  - attach_nvml_observer() in both states.
  - frame_count / last_latency properties in both states.
  - __enter__ / __exit__ context-manager protocol.
"""

from __future__ import annotations

from unittest.mock import patch

from cuda_link.cuda_ipc_importer import CUDAIPCImporter, ImportOutcome

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFormat:
    def __init__(self, shape, dtype_str):
        self.shape = shape
        self.dtype_str = dtype_str


class _FakeResult:
    def __init__(self, outcome, frame):
        self.outcome = outcome
        self.frame = frame


class _FakeConnectedImporter:
    """Stand-in for a connected cuda_link.importer.Importer — no CUDA required."""

    _initialized = True

    def __init__(self, fmt=None):
        self._format = fmt
        self.get_frame_numpy_calls = 0
        self.attached_observer = None
        self.closed = False
        # Record the stream kwarg each call received, so tests can verify the
        # shim actually forwards it rather than dropping it on the floor.
        self.last_get_frame_stream = "__unset__"
        self.last_get_frame_cupy_stream = "__unset__"

    def get_frame(self, stream=None):
        self.last_get_frame_stream = stream
        return _FakeResult(ImportOutcome.NEW_FRAME, "tensor_sentinel")

    def get_frame_numpy(self):
        self.get_frame_numpy_calls += 1
        return _FakeResult(ImportOutcome.NEW_FRAME, "numpy_sentinel")

    def get_frame_cupy(self, stream=None):
        self.last_get_frame_cupy_stream = stream
        return _FakeResult(ImportOutcome.NEW_FRAME, "cupy_sentinel")

    def get_stats(self):
        return {"initialized": True, "frames": 5}

    def is_ready(self):
        return True

    @property
    def frame_count(self):
        return 5

    @property
    def last_latency(self):
        return 1.5

    def attach_nvml_observer(self, observer):
        self.attached_observer = observer

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# connect() — shape/dtype sync from _format
# ---------------------------------------------------------------------------


def test_connect_syncs_shape_dtype_from_format(monkeypatch):
    import cuda_link.cuda_ipc_importer as shim_mod

    fake_importer = _FakeConnectedImporter(fmt=_FakeFormat(shape=(4, 4, 3), dtype_str="uint8"))

    def _fake_open(cls, spec, *, policy=None, cuda=None):
        return fake_importer

    monkeypatch.setattr(shim_mod.Importer, "open", classmethod(_fake_open))

    imp = CUDAIPCImporter(shm_name="whatever")
    imp.connect()

    assert imp._importer is fake_importer
    assert imp.shape == (4, 4, 3)
    assert imp.dtype == "uint8"


# ---------------------------------------------------------------------------
# get_frame() — not-connected and connected paths
# ---------------------------------------------------------------------------


def test_get_frame_connected_returns_frame_on_new_frame():
    imp = CUDAIPCImporter(shm_name="x")
    fake = _FakeConnectedImporter()
    imp._importer = fake
    assert imp.get_frame(stream="s1") == "tensor_sentinel"
    # get_frame() must forward its stream kwarg to the inner Importer
    # unchanged, not silently default it to None.
    assert fake.last_get_frame_stream == "s1"


# ---------------------------------------------------------------------------
# get_frame_numpy() — not-connected path (independent of NUMPY_AVAILABLE)
# ---------------------------------------------------------------------------


def test_get_frame_numpy_returns_none_when_not_connected():
    imp = CUDAIPCImporter(shm_name="test", shape=(64, 64, 4))
    assert imp.get_frame_numpy() is None


def test_get_frame_numpy_connected_returns_frame():
    imp = CUDAIPCImporter(shm_name="x")
    fake = _FakeConnectedImporter()
    imp._importer = fake
    assert imp.get_frame_numpy() == "numpy_sentinel"
    assert fake.get_frame_numpy_calls == 1


# ---------------------------------------------------------------------------
# get_frame_cupy() — not-connected and connected paths
# ---------------------------------------------------------------------------


def test_get_frame_cupy_returns_none_when_not_connected():
    imp = CUDAIPCImporter(shm_name="x")
    assert imp.get_frame_cupy() is None


def test_get_frame_cupy_connected_returns_frame():
    imp = CUDAIPCImporter(shm_name="x")
    fake = _FakeConnectedImporter()
    imp._importer = fake
    assert imp.get_frame_cupy(stream="s2") == "cupy_sentinel"
    # get_frame_cupy() must forward its stream kwarg to the inner Importer
    # unchanged, not silently default it to None.
    assert fake.last_get_frame_cupy_stream == "s2"


# ---------------------------------------------------------------------------
# get_stats() — connected path
# ---------------------------------------------------------------------------


def test_get_stats_connected_delegates():
    imp = CUDAIPCImporter(shm_name="x")
    imp._importer = _FakeConnectedImporter()
    assert imp.get_stats() == {"initialized": True, "frames": 5}


# ---------------------------------------------------------------------------
# attach_nvml_observer() — both states
# ---------------------------------------------------------------------------


def test_attach_nvml_observer_before_connect_stores_only():
    imp = CUDAIPCImporter(shm_name="x")
    obs = object()
    imp.attach_nvml_observer(obs)
    assert imp._nvml_observer is obs
    assert imp._importer is None


def test_attach_nvml_observer_when_connected_delegates():
    imp = CUDAIPCImporter(shm_name="x")
    fake = _FakeConnectedImporter()
    imp._importer = fake
    obs = object()
    imp.attach_nvml_observer(obs)
    assert imp._nvml_observer is obs
    assert fake.attached_observer is obs


# ---------------------------------------------------------------------------
# frame_count / last_latency properties — both states
# ---------------------------------------------------------------------------


def test_frame_count_and_last_latency_default_zero_when_not_connected():
    imp = CUDAIPCImporter(shm_name="x")
    assert imp.frame_count == 0
    assert imp.last_latency == 0.0


def test_frame_count_and_last_latency_delegate_when_connected():
    imp = CUDAIPCImporter(shm_name="x")
    imp._importer = _FakeConnectedImporter()
    assert imp.frame_count == 5
    assert imp.last_latency == 1.5


# ---------------------------------------------------------------------------
# __enter__ / __exit__ context manager protocol
# ---------------------------------------------------------------------------


def test_enter_calls_connect_when_not_connected():
    imp = CUDAIPCImporter(shm_name="x")
    with patch.object(imp, "connect") as mock_connect:
        result = imp.__enter__()
    mock_connect.assert_called_once_with()
    assert result is imp


def test_enter_skips_connect_when_already_connected():
    imp = CUDAIPCImporter(shm_name="x")
    imp._importer = _FakeConnectedImporter()
    with patch.object(imp, "connect") as mock_connect:
        result = imp.__enter__()
    mock_connect.assert_not_called()
    assert result is imp


def test_exit_calls_cleanup():
    imp = CUDAIPCImporter(shm_name="x")
    with patch.object(imp, "cleanup") as mock_cleanup:
        imp.__exit__(None, None, None)
    mock_cleanup.assert_called_once_with()


def test_context_manager_full_roundtrip():
    fake = _FakeConnectedImporter()
    imp = CUDAIPCImporter(shm_name="x")
    imp._importer = fake
    with imp as ctx:
        assert ctx is imp
    assert fake.closed is True
    assert imp._importer is None
