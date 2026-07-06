"""
Characterization tests: TDSender's host status/tint side-effects, verified through
FakeTDHost.status_calls (td_exporter/_td_fakes.py) rather than only against RealTDHost.

FakeTDHost's set_warning_status/set_error_status/set_info_status/clear_status now record
every call as ("kind", msg) into self.status_calls (Part A of this PR-review pass) --
previously they were no-ops, so status emission was only ever exercised end-to-end inside
real TouchDesigner. This file asserts the status/tint contract directly:

  - _is_unsupported_format(top_op) True  -> set_warning_status(...) every skipped frame
    (TDSender.py:536); once the format becomes supported again -> one clear_status() call
    (TDSender.py:546).
  - initialize() failing (Exporter.open raising) -> set_error_status("Initialization
    failed: ...") (TDSender.py:506).
  - A geometry/dtype-change reopen -> set_info_status(<reopen status>) immediately, before
    the first frame at the new geometry is even published (TDSender.py:763).
  - Every successfully PUBLISHED frame -> set_info_status(<pub status>) (TDSender.py:806).

See test_tdsender_grow_safety.py for the shared GPU-free harness rationale (a purpose-
built _FakeExporter double stands in for Exporter.open()).
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
    """Minimal Exporter double -- see test_tdsender_grow_safety.py for rationale."""

    def __init__(self, spec) -> None:
        self.data_size = spec.width * spec.height * spec.channels * DtypeCodec.itemsize(spec.dtype)
        self.write_idx = 0
        self.frame_count = 0
        self.ipc_stream = types.SimpleNamespace(value=0)
        self.shm_handle = None

    def record_source_sync(self, stream_id: int) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def export(self, frame) -> FrameOutcome:
        self.frame_count += 1
        self.write_idx += 1
        return FrameOutcome.PUBLISHED

    def close(self) -> None:
        pass


@pytest.fixture
def fake_exporter_open(monkeypatch: pytest.MonkeyPatch):
    def _patched_open(spec, *, policy=None, cuda=None, barrier=None):  # noqa: ANN001, ARG001
        return _FakeExporter(spec)

    monkeypatch.setattr(Exporter, "open", _patched_open)


def _make_engine(shm_name: str, host: FakeTDHost) -> TDSenderEngine:
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


def test_unsupported_format_emits_warning_then_clear(fake_exporter_open: None) -> None:
    top = FakeTOPHandle(
        pixel_format="16-bit float (RGBA)",  # matches _CUDA_UNSUPPORTED_PIXEL_FORMATS ("16float")
        pixel_format_name="rgba16float",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x8000,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_status_unsupported_{uuid.uuid4().hex[:8]}", host)

    assert engine.export_frame() is False, "Unsupported format must skip the frame."
    warning_calls = [c for c in host.status_calls if c[0] == "warning"]
    assert len(warning_calls) == 1
    assert "unsupported pixel format" in warning_calls[0][1]
    assert "16-bit float (RGBA)" in warning_calls[0][1]

    # A second frame at the SAME unsupported format must keep warning (not re-clear).
    assert engine.export_frame() is False
    assert len([c for c in host.status_calls if c[0] == "warning"]) == 2
    assert not any(c[0] == "clear" for c in host.status_calls)

    # Format becomes supported -> exactly one clear_status() call, then normal lazy-init proceeds.
    top._pixel_format = "32-bit float (RGBA)"
    top._pixel_format_name = "rgba32float"
    assert engine.export_frame() is False  # lazy-init probe frame (still returns False, but via a different path)
    clear_calls = [c for c in host.status_calls if c[0] == "clear"]
    assert len(clear_calls) == 1


def test_initialization_failure_emits_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_open(spec, *, policy=None, cuda=None, barrier=None):  # noqa: ANN001, ARG001
        raise OSError("simulated CUDA IPC failure")

    monkeypatch.setattr(Exporter, "open", _raise_open)

    top = FakeTOPHandle(
        pixel_format="32-bit float (RGBA)",
        pixel_format_name="rgba32float",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x9000,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_status_initfail_{uuid.uuid4().hex[:8]}", host)

    assert engine.export_frame() is False, "A failed initialize() must not raise out of export_frame."
    error_calls = [c for c in host.status_calls if c[0] == "error"]
    assert len(error_calls) == 1
    assert "Initialization failed" in error_calls[0][1]
    assert "simulated CUDA IPC failure" in error_calls[0][1]


def test_geometry_reopen_emits_info_status_before_publish(fake_exporter_open: None) -> None:
    top = FakeTOPHandle(
        pixel_format="8-bit fixed (RGBA)",
        pixel_format_name="rgba8fixed",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0xA000,
        cuda_size=64 * 64 * 4 * 1,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_status_reopen_{uuid.uuid4().hex[:8]}", host)

    assert engine.export_frame() is False  # lazy-init
    assert engine.export_frame() is True  # steady state at rgba8fixed
    host.status_calls.clear()

    # Geometry/dtype change (uint8 -> float32) with cuda_size deliberately left stale --
    # the reopen's set_info_status must fire even though the grow-safety guard then skips
    # this same frame's publish (info status is emitted unconditionally on reopen, before
    # the guard is even checked -- TDSender.py:763 precedes :778).
    top._pixel_format = "32-bit float (RGBA)"
    top._pixel_format_name = "rgba32float"
    assert engine.export_frame() is False, "Grow-safety guard must still skip this frame."

    info_calls = [c for c in host.status_calls if c[0] == "info"]
    assert len(info_calls) == 1, "Reopen must emit exactly one info status, independent of the grow-safety outcome."
    assert info_calls[0][1] == "64x64 rgba32float"


def test_steady_state_publish_emits_info_status_every_frame(fake_exporter_open: None) -> None:
    top = FakeTOPHandle(
        pixel_format="32-bit float (RGBA)",
        pixel_format_name="rgba32float",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0xB000,
        cuda_size=64 * 64 * 4 * 4,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_status_publish_{uuid.uuid4().hex[:8]}", host)

    assert engine.export_frame() is False  # lazy-init
    host.status_calls.clear()

    for _ in range(3):
        assert engine.export_frame() is True

    info_calls = [c for c in host.status_calls if c[0] == "info"]
    assert len(info_calls) == 3, "Every published (steady-state) frame must emit set_info_status."
    assert all(msg == "64x64 rgba32float" for _, msg in info_calls)
