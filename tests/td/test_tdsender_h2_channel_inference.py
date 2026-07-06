"""
Characterization tests: TDSender's H2 physical-channel inference (TDSender.py:662-706).

TD may report cm.channels=1 (logical mono) for a source TOP while the underlying CUDA
allocation is actually RGBA-padded (4-channel) -- e.g. a mono float32 TOP: cm.channels=1
but cm.size == W*H*4 comps*4 bytes. When the resolved dtype/channel byte count doesn't
match cm.size AND no pixelFormatName override is active (see
test_tdsender_pixelformat_override.py), export_frame tries reinterpreting the buffer as
4-channel then 2-channel before giving up. This block only fires as a *defensive*
fallback (H2) after the primary resolution path and the override both fail to explain
the byte count.

See test_tdsender_grow_safety.py for the shared GPU-free harness rationale.
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


def test_infers_rgba_padded_mono(fake_exporter_open: None) -> None:
    """cm.channels=1 but cm.size implies a 4-channel float32 allocation -- H2 must infer
    phys_ch=4/float32 and reopen accordingly."""
    w = h = 64
    top = FakeTOPHandle(
        pixel_format="32-bit float (R)",
        pixel_format_name="useinput",  # excluded from the override so H2 can run
        width=w,
        height=h,
        channels=1,  # cm.channels reported as mono (logical)
        gpu_ptr=0x6000,
        cuda_size=w * h * 4 * 4,  # actual allocation: RGBA-padded float32
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    log_calls: list[str] = []
    engine = _make_engine(f"test_h2_infer_{uuid.uuid4().hex[:8]}", host, log_calls=log_calls)

    assert engine.export_frame() is False  # lazy-init: guesses uint8 (unsupported bpp fallback)
    result = engine.export_frame()

    assert engine._last_resolved_dtype == "float32", "H2 must infer float32 from the RGBA-padded byte count."
    assert engine._last_cm_channels == 4, "H2 must infer 4 physical channels, not the logical cm.channels=1."
    assert engine._current_spec.channels == 4
    assert engine._current_spec.dtype == "float32"
    assert result is True, "Once inferred and reopened, data_size == cm.size (no grow-safety skip)."
    assert any("inferring" in m for m in log_calls), "Inference must be logged."

    # Same geometry again -- fast path caches the result, no repeat inference log.
    log_calls.clear()
    engine.export_frame()
    assert not any("inferring" in m for m in log_calls)


def test_unsupported_bytes_per_pixel_skips_and_warns_once(fake_exporter_open: None) -> None:
    """A byte count that matches none of uint8/uint16/float32 at any of the tried channel
    counts must skip the frame, log once, and set _warned_dtype_size."""
    w = h = 64
    top = FakeTOPHandle(
        pixel_format="32-bit float (R)",
        pixel_format_name="useinput",
        width=w,
        height=h,
        channels=1,
        gpu_ptr=0x7000,
        cuda_size=w * h * 3,  # 3 bytes/pixel at channels=1 -- not 1/2/4 at any tried phys_ch
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    log_calls: list[str] = []
    engine = _make_engine(f"test_h2_unsupported_{uuid.uuid4().hex[:8]}", host, log_calls=log_calls)

    assert engine.export_frame() is False  # lazy-init
    log_calls.clear()
    assert engine.export_frame() is False, "Unsupported bytes-per-pixel must skip the frame."
    assert engine._warned_dtype_size is True
    unsupported_logs = [m for m in log_calls if "Unsupported bytes-per-pixel" in m]
    assert len(unsupported_logs) == 1

    # Second consecutive call with the identical (still-unsupported) geometry must not
    # re-log -- _warned_dtype_size gates the message.
    log_calls.clear()
    assert engine.export_frame() is False
    assert not any("Unsupported bytes-per-pixel" in m for m in log_calls)
