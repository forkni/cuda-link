"""
TDSender float16 sources: shared as float32 via cudaMemory(pixelFormat=...).

TD's cudaMemory() rejects every float16 texture, but converts on the GPU when asked for
a 32-bit float target (pixel-format matrix probe, TD 2025.33230). The Sender requests
the matching *32float name on every cudaMemory() call and publishes a float32 spec.
A float16 source whose pixelFormatName is unmapped still takes the skip + warning path.

GPU-free: Exporter.open is patched via the shared ``fake_exporter_open`` fixture,
which builds this module's ``_FakeExporter``.
"""

from __future__ import annotations

import types
import uuid

import pytest
from Exporter import FrameOutcome
from fakes import FakeTDHost, FakeTOPHandle, fake_exporter_open  # noqa: F401  # fixture, used via usefixtures
from SHMProtocol import FLAGS_MONO_ALPHA
from TDConfig import TDSenderConfig
from TDSender import TDSenderEngine


class _FakeExporter:
    """Minimal Exporter double: only the surface TDSender.py touches."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.data_size = spec.width * spec.height * spec.channels * 4  # float32 in every test here
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
        shm_name=f"test_f16_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )


def _top(pixel_format_name: str, channels: int, *, pixel_format: str = "16-bit float (RGBA)") -> FakeTOPHandle:
    # cuda_size is the converted float32 buffer TD hands back for the *32float target.
    return FakeTOPHandle(
        pixel_format=pixel_format,
        pixel_format_name=pixel_format_name,
        width=64,
        height=64,
        channels=channels,
        gpu_ptr=0x1000,
        cuda_size=64 * 64 * channels * 4,
    )


pytestmark = pytest.mark.usefixtures("fake_exporter_open")


@pytest.mark.parametrize(
    ("source", "target", "channels", "flags"),
    [
        ("rgba16float", "rgba32float", 4, 0),
        ("rg16float", "rg32float", 2, 0),
        ("mono16float", "mono32float", 1, 0),
        ("a16float", "a32float", 1, 0),
        ("monoalpha16float", "monoalpha32float", 2, FLAGS_MONO_ALPHA),
    ],
)
def test_float16_source_exports_as_float32(source: str, target: str, channels: int, flags: int) -> None:
    top = _top(source, channels)
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(host)

    assert engine.export_frame() is False  # lazy-init probe
    assert engine.export_frame() is True
    spec = engine._current_spec
    assert (spec.dtype, spec.channels, spec.extra_flags) == ("float32", channels, flags)
    assert top.cuda_memory_calls, "cudaMemory() was never called"
    assert all(pf == target for _, pf in top.cuda_memory_calls)
    assert not any(kind == "warning" for kind, _ in host.status_calls)


def test_unmapped_float16_name_is_still_rejected() -> None:
    # "useinput" has no float32 target to request, so the display-string check rejects it.
    top = _top("useinput", 4)
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(host)

    assert engine.export_frame() is False
    assert engine.export_frame() is False
    assert top.cuda_memory_calls == []
    assert any(kind == "warning" for kind, _ in host.status_calls)


def test_supported_source_requests_no_conversion() -> None:
    top = _top("rgba32float", 4, pixel_format="32-bit float (RGBA)")
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(host)

    assert engine.export_frame() is False
    assert engine.export_frame() is True
    assert all(pf is None for _, pf in top.cuda_memory_calls)


def test_switch_to_float16_mid_stream_keeps_exporter() -> None:
    top = _top("rgba32float", 4, pixel_format="32-bit float (RGBA)")
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(host)
    assert engine.export_frame() is False
    assert engine.export_frame() is True
    first = engine._exporter

    # Upstream flips to float16: the converted frame has the same float32 spec, so no reopen.
    top._pixel_format_name = "rgba16float"
    top._pixel_format = "16-bit float (RGBA)"
    top.cuda_memory_calls.clear()
    assert engine.export_frame() is True
    assert engine._exporter is first
    assert top.cuda_memory_calls == [(0, "rgba32float")]
