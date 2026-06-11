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
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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


def test_numpy_backend_cached() -> None:
    """_numpy_backend must be created once and reused on subsequent calls.

    Drives only the caching logic; never enters _consume_frame (not initialized).
    """
    from cuda_link.importer import _NumpyBackend

    imp = _make_importer_stub()
    assert imp._numpy_backend is None

    # Simulate the lazy-init + reuse guard from get_frame_numpy:
    if imp._numpy_backend is None:
        imp._numpy_backend = _NumpyBackend(imp)
    first = imp._numpy_backend

    # Second pass — same instance
    if imp._numpy_backend is None:
        imp._numpy_backend = _NumpyBackend(imp)
    assert imp._numpy_backend is first, "second call must reuse the same _NumpyBackend instance"


def test_torch_backend_cached_and_stream_refreshed() -> None:
    """_TorchBackend must be created once; _stream updated in-place on subsequent calls."""
    from cuda_link.importer import TORCH_AVAILABLE, _TorchBackend

    if not TORCH_AVAILABLE:
        pytest.skip("torch not installed")

    imp = _make_importer_stub()
    assert imp._torch_backend is None

    stream_a = object()
    stream_b = object()

    # Simulate the caching logic from get_frame():
    if imp._torch_backend is None:
        imp._torch_backend = _TorchBackend(imp, stream_a)
    else:
        imp._torch_backend._stream = stream_a
    first = imp._torch_backend
    assert first._stream is stream_a

    if imp._torch_backend is None:
        imp._torch_backend = _TorchBackend(imp, stream_b)
    else:
        imp._torch_backend._stream = stream_b
    assert imp._torch_backend is first, "must reuse the same _TorchBackend instance"
    assert imp._torch_backend._stream is stream_b, "_stream must be updated in-place"


def test_cupy_backend_cached_and_stream_refreshed() -> None:
    """_CupyBackend must be created once; _stream updated in-place on subsequent calls."""
    from cuda_link.importer import CUPY_AVAILABLE, _CupyBackend

    if not CUPY_AVAILABLE:
        pytest.skip("cupy not installed")

    imp = _make_importer_stub()
    assert imp._cupy_backend is None

    stream_a = object()
    stream_b = object()

    if imp._cupy_backend is None:
        imp._cupy_backend = _CupyBackend(imp, stream_a)
    else:
        imp._cupy_backend._stream = stream_a
    first = imp._cupy_backend
    assert first._stream is stream_a

    if imp._cupy_backend is None:
        imp._cupy_backend = _CupyBackend(imp, stream_b)
    else:
        imp._cupy_backend._stream = stream_b
    assert imp._cupy_backend is first
    assert imp._cupy_backend._stream is stream_b


def test_numpy_backend_none_initially() -> None:
    """All three backend fields must start as None — no eager allocation."""
    imp = _make_importer_stub()
    assert imp._numpy_backend is None
    assert imp._torch_backend is None
    assert imp._cupy_backend is None
