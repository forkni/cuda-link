"""
Characterization tests: TDSender's pixelFormatName override (TDSender.py:631-660).

TD's pixelFormatName is the immediate, authoritative format signal -- it updates the
instant the source format changes, whereas cm.size/cm.data_type can lag behind (most
notably on a dtype-SHRINK, where the old, larger CUDA allocation persists until TD
reallocates). When pixelFormatName maps to a known dtype/channel pair, export_frame
unconditionally overrides resolved_dtype/cm_channels with that pair -- even when
cm.size disagrees -- and the resulting H2 physical-channel-inference block (below)
is deliberately skipped whenever this override is active (TDSender.py:670), since H2
would otherwise wrongly re-derive a stale dtype from the lagging cm.size.

See test_tdsender_grow_safety.py for the shared GPU-free harness rationale (a purpose-
built _FakeExporter double stands in for Exporter.open(), because FakeCUDAAdapter's
stream sentinel has no `.value` attribute that TDSender's ipc_stream handling needs).
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


def _make_engine(shm_name: str, host: FakeTDHost, log_calls: list[str] | None = None) -> TDSenderEngine:
    config = TDSenderConfig(activation_barrier=False, use_graphs=False)
    return TDSenderEngine(
        host=host,
        config=config,
        cuda=None,
        log_fn=(log_calls.append if log_calls is not None else (lambda *a, **k: None)),
        num_slots=1,
        device=0,
        shm_name=shm_name,
        verbose=False,
    )


def test_override_wins_over_stale_cm_size(fake_exporter_open: None) -> None:
    """rgba16fixed (uint16/4ch) must win even when cm.size is still uint8-sized (stale),
    and must suppress H2 inference (which would otherwise infer a 2-channel reading)."""
    top = FakeTOPHandle(
        pixel_format="8-bit fixed (RGBA)",
        pixel_format_name="rgba8fixed",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x3000,
        cuda_size=64 * 64 * 4 * 1,  # uint8-sized: 16384 bytes
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_pf_override_{uuid.uuid4().hex[:8]}", host)

    assert engine.export_frame() is False  # lazy-init at uint8/4ch
    engine.export_frame()  # steady-state frame at uint8/4ch establishes the geom cache

    # pixelFormatName flips to uint16, but cm.size is left at the OLD uint8 value --
    # without the override, H2 would infer a 2-channel reading (16384 // (2*4096) == 2).
    top._pixel_format = "16-bit fixed (RGBA)"
    top._pixel_format_name = "rgba16fixed"
    engine.export_frame()

    assert engine._last_resolved_dtype == "uint16", "pixelFormatName override must win over the stale cm.size."
    assert engine._last_cm_channels == 4, (
        "Override must suppress H2 inference -- without suppression H2 would infer 2ch from the stale byte count."
    )
    assert engine._current_spec.dtype == "uint16", "Geometry/dtype-change reopen must have used the override dtype."


def test_unmapped_name_is_ignored(fake_exporter_open: None) -> None:
    """'useinput' (and any name absent from _PIXEL_FMT_NAME_TO_DTYPE) must not override --
    resolution falls through to the cm.size-derived dtype."""
    top = FakeTOPHandle(
        pixel_format="8-bit fixed (RGBA)",
        pixel_format_name="useinput",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x4000,
        cuda_size=64 * 64 * 4 * 1,  # uint8-sized -- matches cm.channels=4 exactly, so H2 doesn't fire either
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = _make_engine(f"test_pf_unmapped_{uuid.uuid4().hex[:8]}", host)

    assert engine.export_frame() is False  # lazy-init: useinput excluded, falls back to buffer-size guess
    engine.export_frame()

    assert engine._last_resolved_dtype == "uint8", "'useinput' must never override -- byte-derived dtype wins."
    assert engine._last_cm_channels == 4


def test_transition_logged_once(fake_exporter_open: None) -> None:
    """The pixelFormatName-change transition log fires once, not on every subsequent
    frame at the same (now-cached) format."""
    top = FakeTOPHandle(
        pixel_format="8-bit fixed (RGBA)",
        pixel_format_name="rgba8fixed",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x5000,
        cuda_size=64 * 64 * 4 * 1,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    log_calls: list[str] = []
    engine = _make_engine(f"test_pf_log_{uuid.uuid4().hex[:8]}", host, log_calls=log_calls)

    engine.export_frame()  # lazy-init
    engine.export_frame()  # steady state at rgba8fixed

    top._pixel_format = "16-bit fixed (RGBA)"
    top._pixel_format_name = "rgba16fixed"
    log_calls.clear()
    engine.export_frame()  # transition frame -- logs once
    transition_logs = [m for m in log_calls if "pixelFormatName changed" in m]
    assert len(transition_logs) == 1

    # Next frame: TD has caught up (cuda_size now matches uint16/4ch) -- same format,
    # fast path takes over, no repeat log.
    top._cuda_size = 64 * 64 * 4 * 2
    log_calls.clear()
    engine.export_frame()
    assert not any("pixelFormatName changed" in m for m in log_calls)
