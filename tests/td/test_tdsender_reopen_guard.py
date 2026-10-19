"""
TDSender geometry-change reopen: failure handling and flags-only changes
(2026-10-19 block review, items 3 and 12 sender side).

- A failing ``Exporter.open`` during a geometry-change reopen must not raise out of
  the cook callback. The already-closed old Exporter is dropped so the next cook
  re-initializes through the lazy-init path.
- An ``extra_flags``-only change (rg8fixed <-> monoalpha8fixed: same shape and dtype,
  different FLAGS_MONO_ALPHA) must trigger a reopen so the receiver sees the new flag.

GPU-free: Exporter.open is patched via the shared ``fake_exporter_open`` fixture,
which builds this module's ``_FakeExporter``.
"""

from __future__ import annotations

import types
import uuid

import pytest
from Exporter import Exporter, FrameOutcome
from fakes import FakeTDHost, FakeTOPHandle, fake_exporter_open  # noqa: F401
from SHMProtocol import FLAGS_BGRA, FLAGS_MONO_ALPHA
from TDConfig import TDSenderConfig
from TDSender import TDSenderEngine, _channel_flags


class _FakeExporter:
    """Minimal Exporter double: only the surface TDSender.py touches."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.data_size = spec.width * spec.height * spec.channels
        self.write_idx = 0
        self.frame_count = 0
        self.ipc_stream = types.SimpleNamespace(value=0)
        self.shm_handle = None
        self.closed = False

    def record_source_sync(self, stream_id: int) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def export(self, frame) -> FrameOutcome:
        self.frame_count += 1
        self.write_idx += 1
        return FrameOutcome.PUBLISHED

    def close(self) -> None:
        self.closed = True


def _make_engine(host: FakeTDHost) -> TDSenderEngine:
    config = TDSenderConfig(activation_barrier=False, use_graphs=False)
    return TDSenderEngine(
        host=host,
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=f"test_reopen_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )


def _top(pixel_format_name: str, channels: int, *, width: int = 64, height: int = 64) -> FakeTOPHandle:
    return FakeTOPHandle(
        pixel_format="8-bit fixed",
        pixel_format_name=pixel_format_name,
        width=width,
        height=height,
        channels=channels,
        gpu_ptr=0x1000,
        cuda_size=width * height * channels,
    )


def test_reopen_failure_drops_exporter_and_reinitializes(
    fake_exporter_open: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    top = _top("rgba8fixed", 4)
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(host)

    assert engine.export_frame() is False  # lazy-init probe
    assert engine.export_frame() is True  # steady state
    assert engine._tx_window.started is True
    old_exporter = engine._exporter

    # Resolution change; the reopen's Exporter.open fails (e.g. SHM create refused).
    top._width, top._height, top._cuda_size = 128, 128, 128 * 128 * 4
    working_open = Exporter.open

    def _failing_open(spec, *, policy=None, cuda=None, barrier=None):  # noqa: ANN001, ARG001
        raise OSError("simulated SHM create failure")

    monkeypatch.setattr(Exporter, "open", _failing_open)
    assert engine.export_frame() is False  # must not raise out of the cook
    assert old_exporter.closed is True
    assert engine._exporter is None
    assert engine._initialized is False
    assert engine._tx_window.started is False  # stale window would yield negative FPS on the new Exporter
    assert any(kind == "error" and "Reopen failed" in msg for kind, msg in host.status_calls)

    # Open works again: the next cook re-initializes via lazy init, then exports.
    monkeypatch.setattr(Exporter, "open", working_open)
    assert engine.export_frame() is False  # lazy-init probe at the new size
    assert engine._exporter is not None
    assert (engine._current_spec.width, engine._current_spec.height) == (128, 128)
    assert engine.export_frame() is True


def test_flags_only_change_triggers_reopen(fake_exporter_open: None) -> None:
    top = _top("rg8fixed", 2)
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(host)

    assert engine.export_frame() is False  # lazy-init probe
    assert engine.export_frame() is True
    first = engine._exporter
    assert engine._current_spec.extra_flags == 0

    # Same 64x64x2 uint8 shape, but now monoalpha: only FLAGS_MONO_ALPHA differs.
    top._pixel_format_name = "monoalpha8fixed"
    engine.export_frame()
    assert engine._exporter is not first, "flags-only change must reopen the Exporter"
    assert first.closed is True
    assert engine._current_spec.extra_flags == FLAGS_MONO_ALPHA
    assert engine._current_spec.channels == 2


@pytest.mark.parametrize(
    ("pixel_format_name", "channels", "dtype", "expected"),
    [
        ("rgba8fixed", 4, "uint8", FLAGS_BGRA),
        ("rgba16fixed", 4, "uint16", 0),
        ("rgba32float", 4, "float32", 0),
        ("rg8fixed", 2, "uint8", 0),
        ("monoalpha8fixed", 2, "uint8", FLAGS_MONO_ALPHA),
        ("mono8fixed", 1, "uint8", 0),
    ],
)
def test_channel_flags(pixel_format_name: str, channels: int, dtype: str, expected: int) -> None:
    # TD cudaMemory() returns 8-bit RGBA as B,G,R,A; 16/32-bit RGBA stay R,G,B,A.
    assert _channel_flags(pixel_format_name, channels, dtype) == expected


def test_rgba8_source_publishes_bgra_flag(fake_exporter_open: None) -> None:
    top = _top("rgba8fixed", 4)
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(host)

    assert engine.export_frame() is False  # lazy-init probe
    assert engine.export_frame() is True
    assert engine._current_spec.extra_flags == FLAGS_BGRA
