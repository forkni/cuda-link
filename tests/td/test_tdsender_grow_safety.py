"""
Characterization tests: TDSender's grow-safety guard (TDSender.py:772-780).

After a dtype-grow reopen (e.g. uint8 -> float32), TD may not have reallocated its
source texture yet, so top_op.cuda_memory() still reports the OLD (smaller) size on
the very frame that follows the reopen. The guard compares the freshly-reopened
Exporter's data_size against that stale cm.size: if data_size > cm.size, the frame is
skipped and a force-cook is issued to nudge TD into reallocating.

On a dtype-SHRINK (e.g. float32 -> uint8), data_size < cm.size, so the guard never
fires -- the shrink path is always unblocked, and reads only the valid front region
of the still-oversized allocation via GpuFrame(size=exporter.data_size).

GPU-free harness: TDSenderEngine.initialize()/export_frame() call
`Exporter.open(spec, policy=policy, cuda=None)` unconditionally -- the engine-level
`cuda` ctor kwarg is intentionally ignored (TDSender.py:182). Exporter.open is patched
to return a minimal fake Exporter double instead of a real one: FakeCUDAAdapter (the
usual GPU-free stand-in, per tests/integration/test_cpu_roundtrip.py) creates its
stream handles as an opaque `_FakeHandle` sentinel with no `.value` attribute, but
TDSender unconditionally does `int(self._exporter.ipc_stream.value)` both in
_arm_same_stream_ordering() and every export_frame() cuda_memory() call -- so a real
Exporter+FakeCUDAAdapter pairing cannot satisfy TDSender's stream-handle contract. The
fake double below sidesteps that gap entirely and isolates exactly the TDSender-owned
branching logic under test (grow-safety, not Exporter/SHM internals, which are already
covered by tests/core/test_exporter_python.py).
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from Exporter import Exporter, FrameOutcome  # noqa: E402
from fakes import FakeTDHost, FakeTOPHandle  # noqa: E402
from SHMProtocol import DtypeCodec  # noqa: E402
from TDConfig import TDSenderConfig  # noqa: E402
from TDSender import TDSenderEngine  # noqa: E402


class _FakeExporter:
    """Minimal Exporter double: only the surface TDSender.py touches.

    data_size is derived from spec the same way the real Exporter computes it
    (width * height * channels * itemsize(dtype)) so the grow-safety guard's
    `data_size > cm.size` comparison reflects a real dtype-driven reopen.
    """

    def __init__(self, spec) -> None:
        self.data_size = spec.width * spec.height * spec.channels * DtypeCodec.itemsize(spec.dtype)
        self.write_idx = 0
        self.frame_count = 0
        self.ipc_stream = types.SimpleNamespace(value=0)
        self.shm_handle = None
        self.export_calls: list[int] = []

    def record_source_sync(self, stream_id: int) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def export(self, frame) -> FrameOutcome:
        self.export_calls.append(frame.size)
        self.frame_count += 1
        self.write_idx += 1
        return FrameOutcome.PUBLISHED

    def close(self) -> None:
        pass


@pytest.fixture
def fake_exporter_open(monkeypatch: pytest.MonkeyPatch):
    """Patch Exporter.open so every TDSender-internal open() returns a _FakeExporter."""

    def _patched_open(spec, *, policy=None, cuda=None, barrier=None):  # noqa: ANN001, ARG001
        return _FakeExporter(spec)

    monkeypatch.setattr(Exporter, "open", _patched_open)


def _make_engine(shm_name: str, host: FakeTDHost) -> TDSenderEngine:
    # activation_barrier=False: HolderBarrier is an unrelated TD-side lifecycle concern
    # (ActivationBarrier.HolderBarrier); disabling it keeps this test focused on the
    # grow-safety guard alone.
    config = TDSenderConfig(activation_barrier=False, use_graphs=False)
    return TDSenderEngine(
        host=host,
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=shm_name,
        verbose=False,
    )


def test_grow_reopen_then_skip_when_cm_size_stale(fake_exporter_open: None) -> None:
    """uint8 -> float32 reopen: the frame immediately after the reopen must be skipped
    and force-cooked when TD's reported cm.size still reflects the old (smaller) uint8
    allocation."""
    top = FakeTOPHandle(
        pixel_format="8-bit fixed (RGBA)",
        pixel_format_name="rgba8fixed",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x1000,
        cuda_size=64 * 64 * 4 * 1,  # uint8: 16384 bytes
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_grow_{uuid.uuid4().hex[:8]}", host)

    # Frame 1: lazy-init probe -- opens the (fake) Exporter at uint8, skips this cook.
    assert engine.export_frame() is False
    assert engine.is_ready()

    # Frame 2: steady state at uint8 -- exports normally, no reopen.
    assert engine.export_frame() is True

    # Frame 3: TD's pixelFormatName flips to float32, but cuda_size is left at the
    # OLD uint8 value (16384) -- simulating "TD hasn't reallocated its texture yet".
    top._pixel_format = "32-bit float (RGBA)"
    top._pixel_format_name = "rgba32float"
    assert engine.export_frame() is False, (
        "Grow-safety guard must skip the frame immediately following a dtype-grow "
        "reopen when cm.size is still the stale (smaller) value."
    )
    assert top.cook_calls == [True], "Skipped frame must force-cook the source TOP to nudge reallocation."
    assert engine._current_spec.dtype == "float32", "Reopen must still have happened (spec updated to float32)."
    assert engine._exporter.data_size == 64 * 64 * 4 * 4, "New Exporter must be sized for float32 (65536 bytes)."

    # Frame 4: TD has now reallocated -- cuda_size catches up to float32; guard clears.
    top._cuda_size = 64 * 64 * 4 * 4
    assert engine.export_frame() is True, "Once cm.size catches up, the frame must publish normally."


def test_shrink_reopen_never_skips(fake_exporter_open: None) -> None:
    """float32 -> uint8 reopen: data_size < cm.size after the reopen, so the guard must
    never fire -- the shrink path is always unblocked (per TDSender.py:776-777)."""
    top = FakeTOPHandle(
        pixel_format="32-bit float (RGBA)",
        pixel_format_name="rgba32float",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x2000,
        cuda_size=64 * 64 * 4 * 4,  # float32: 65536 bytes
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_shrink_{uuid.uuid4().hex[:8]}", host)

    assert engine.export_frame() is False  # lazy-init probe
    assert engine.export_frame() is True  # steady state at float32

    # Flip to uint8 but leave cuda_size at the OLD (larger) float32 value -- TD's
    # texture hasn't shrunk yet either; data_size(new, smaller) < cm.size(old, larger).
    top._pixel_format = "8-bit fixed (RGBA)"
    top._pixel_format_name = "rgba8fixed"
    assert engine.export_frame() is True, (
        "Shrink reopen must never skip via the grow-safety guard -- "
        "data_size < cm.size is expected and safe (reads only the valid front region)."
    )
    assert top.cook_calls == [], "Shrink path must not force-cook."
    assert engine._current_spec.dtype == "uint8"
    assert engine._exporter.data_size == 64 * 64 * 4 * 1
