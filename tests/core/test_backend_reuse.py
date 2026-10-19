"""
P8 completion tests — object reuse on the import hot path.

1. AcquireResult is a NamedTuple (lighter construction, immutable, no __dict__).
2. get_frame_numpy() reuses the same _NumpyBackend instance across calls.
3. get_frame() / get_frame_cupy() reuse _TorchBackend / _CupyBackend (when available).

All tests run without GPU hardware.
"""

from __future__ import annotations

import pytest

from cuda_link.shm_protocol import AcquireResult, SlotState

# ---------------------------------------------------------------------------
# A1 — AcquireResult is a NamedTuple
# ---------------------------------------------------------------------------


def test_acquire_result_is_namedtuple() -> None:
    """AcquireResult must be a subclass of tuple (NamedTuple, not dataclass)."""
    assert issubclass(AcquireResult, tuple), "AcquireResult must be a NamedTuple for lighter per-frame construction"


def test_acquire_result_defaults() -> None:
    """Positional-only construction (state) must leave optional fields at defaults."""
    r = AcquireResult(SlotState.NO_FRAME)
    assert r.state is SlotState.NO_FRAME
    assert r.slot == -1
    assert r.timestamp == 0.0
    assert r.new_version == 0
    assert r.write_idx == 0


def test_acquire_result_keyword_construction() -> None:
    """Keyword construction must still work — all call sites use keyword args."""
    r = AcquireResult(state=SlotState.NEW_FRAME, slot=2, timestamp=1.5, write_idx=7)
    assert r.state is SlotState.NEW_FRAME
    assert r.slot == 2
    assert r.timestamp == 1.5
    assert r.new_version == 0  # defaulted
    assert r.write_idx == 7


def test_acquire_result_immutable() -> None:
    """NamedTuple fields must be read-only — we never mutate AcquireResult after creation."""
    r = AcquireResult(SlotState.SHUTDOWN)
    with pytest.raises((AttributeError, TypeError)):
        r.slot = 99  # type: ignore[misc]


def test_acquire_result_version_changed() -> None:
    """VERSION_CHANGED construction (uses new_version field) must round-trip."""
    r = AcquireResult(state=SlotState.VERSION_CHANGED, new_version=42)
    assert r.state is SlotState.VERSION_CHANGED
    assert r.new_version == 42
    assert r.slot == -1  # default


def test_acquire_result_no_dict() -> None:
    """NamedTuple must not allocate a __dict__ — that is the whole point of the change."""
    r = AcquireResult(SlotState.NO_FRAME)
    assert not hasattr(r, "__dict__"), "NamedTuple must have no instance __dict__"


# ---------------------------------------------------------------------------
# A2 — backend caching (no-GPU; uses the fake/stub machinery)
# ---------------------------------------------------------------------------


def _make_importer_stub():
    """Return a minimally-initialized Importer that is closed (no real CUDA).

    We only need the object identity of the cached backend fields — we never
    actually call _consume_frame or open an IPC connection.
    """
    from unittest.mock import MagicMock

    from cuda_link._importer_port import ImportPolicy, ImportSpec
    from cuda_link.importer import Importer

    spec = ImportSpec(shm_name="__test_stub__", device=0)
    policy = ImportPolicy()
    cuda_fake = MagicMock()
    imp = object.__new__(Importer)
    # Minimal __init__ side-effects needed for field existence only:
    imp._spec = spec
    imp._policy = policy
    imp._cuda = cuda_fake
    imp._conn = None
    imp._format = None
    imp._torch = None
    imp._cupy = None
    imp._numpy = None
    imp._initialized = False
    imp._numpy_backend = None
    imp._torch_backend = None
    imp._cupy_backend = None
    imp._retry = None
    imp._last_write_idx = 0
    return imp


def _record_consumed(monkeypatch: pytest.MonkeyPatch, imp) -> list:
    """Stub the frame-consumption tail so get_frame*() exercises only backend selection.

    Returns the list of backends _consume_frame() was handed, in call order.
    """
    from cuda_link._importer_port import ImportOutcome, ImportResult

    seen: list = []

    def _consume(backend):
        seen.append(backend)
        return ImportResult(outcome=ImportOutcome.NO_FRAME)

    monkeypatch.setattr(imp, "_consume_frame", _consume)
    return seen


def test_numpy_backend_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_frame_numpy() builds one _NumpyBackend lazily and hands the same one to every call."""
    import cuda_link.importer as importer_mod

    monkeypatch.setattr(importer_mod, "NUMPY_AVAILABLE", True)
    imp = _make_importer_stub()
    seen = _record_consumed(monkeypatch, imp)

    imp.get_frame_numpy()
    imp.get_frame_numpy()

    assert len(seen) == 2
    assert isinstance(seen[0], importer_mod._NumpyBackend)
    assert seen[1] is seen[0], "second call must reuse the same _NumpyBackend instance"
    assert imp._numpy_backend is seen[0]


@pytest.mark.parametrize(
    ("method", "flag", "backend_cls", "field"),
    [
        ("get_frame", "TORCH_AVAILABLE", "_TorchBackend", "_torch_backend"),
        ("get_frame_cupy", "CUPY_AVAILABLE", "_CupyBackend", "_cupy_backend"),
    ],
    ids=["torch", "cupy"],
)
def test_stream_backend_cached_and_stream_refreshed(
    monkeypatch: pytest.MonkeyPatch, method: str, flag: str, backend_cls: str, field: str
) -> None:
    """get_frame()/get_frame_cupy() build the backend once and refresh its _stream in place.

    The backend constructors only store references, so the availability flag is patched on
    instead of requiring torch/cupy: the caching contract is pinned on every host.
    """
    import cuda_link.importer as importer_mod

    monkeypatch.setattr(importer_mod, flag, True)
    imp = _make_importer_stub()
    seen = _record_consumed(monkeypatch, imp)
    stream_a = object()
    stream_b = object()

    getattr(imp, method)(stream_a)
    first = seen[0]
    assert isinstance(first, getattr(importer_mod, backend_cls))
    assert first._stream is stream_a

    getattr(imp, method)(stream_b)
    assert seen[1] is first, f"must reuse the same {backend_cls} instance"
    assert first._stream is stream_b, "_stream must be updated in place"
    assert getattr(imp, field) is first


@pytest.mark.parametrize(
    ("method", "flag", "message"),
    [
        ("get_frame", "TORCH_AVAILABLE", "torch is required"),
        ("get_frame_numpy", "NUMPY_AVAILABLE", "numpy is required"),
        ("get_frame_cupy", "CUPY_AVAILABLE", "cupy is required"),
    ],
    ids=["torch", "numpy", "cupy"],
)
def test_missing_backend_library_raises_before_caching(
    monkeypatch: pytest.MonkeyPatch, method: str, flag: str, message: str
) -> None:
    """Without the backend library the call raises and caches no backend."""
    import cuda_link.importer as importer_mod

    monkeypatch.setattr(importer_mod, flag, False)
    imp = _make_importer_stub()
    seen = _record_consumed(monkeypatch, imp)

    with pytest.raises(RuntimeError, match=message):
        getattr(imp, method)()
    assert seen == []
    assert (imp._numpy_backend, imp._torch_backend, imp._cupy_backend) == (None, None, None)


def test_numpy_backend_none_initially() -> None:
    """All three backend fields must start as None — no eager allocation."""
    imp = _make_importer_stub()
    assert imp._numpy_backend is None
    assert imp._torch_backend is None
    assert imp._cupy_backend is None
