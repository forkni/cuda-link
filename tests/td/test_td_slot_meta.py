"""TD layer slot meta (Phase 4): sender stamping, receiver channels, slot_meta Script CHOP.

- TDSender stamps GpuFrame.producer_frame from host.abs_frame() and stream_id from Streamid.
- TDReceiver counts frame gaps / producer skips with the In TOP rules and exposes the
  slot_meta channels (producer_frame mod 2^24).
- script_chop_callbacks writes one sample per channel and never raises.
- script_top_callbacks force-cooks slot_meta after an import, when the CHOP exists.

GPU-free: Exporter.open is patched via the shared ``fake_exporter_open`` fixture.
"""

from __future__ import annotations

import types
import uuid

import pytest
from Exporter import FrameOutcome
from fakes import FakeTDHost, FakeTOPHandle, fake_exporter_open  # noqa: F401  # fixture, used via usefixtures
from SHMProtocol import EMPTY_SLOT_META, INFO_CHOP_FRAME_MOD, SLOT_META_CHANNELS, SlotMeta
from TDConfig import TDSenderConfig
from TDReceiver import TDReceiverEngine
from TDSender import TDSenderEngine


class _FakeExporter:
    """Exporter double that records the GpuFrames it gets (fake_exporter_open builds it)."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.data_size = spec.width * spec.height * spec.channels * 4
        self.write_idx = 0
        self.frame_count = 0
        self.ipc_stream = types.SimpleNamespace(value=0)
        self.shm_handle = None
        self.frames: list = []
        self.last_slot_meta = EMPTY_SLOT_META

    def record_source_sync(self, stream_id: int) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def export(self, frame) -> FrameOutcome:
        self.frames.append(frame)
        self.frame_count += 1
        self.write_idx += 1
        pf = self.write_idx if frame.producer_frame is None else frame.producer_frame
        self.last_slot_meta = SlotMeta(1.0, pf, frame.source_timestamp_us, frame.stream_id, 0)
        return FrameOutcome.PUBLISHED

    def close(self) -> None:
        pass


def _sender(host: FakeTDHost) -> TDSenderEngine:
    return TDSenderEngine(
        host=host,
        config=TDSenderConfig(activation_barrier=False),
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=f"test_slot_meta_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )


def _top() -> FakeTOPHandle:
    return FakeTOPHandle(
        pixel_format="32-bit float (RGBA)",
        pixel_format_name="rgba32float",
        width=8,
        height=8,
        channels=4,
        gpu_ptr=0x1000,
        cuda_size=8 * 8 * 4 * 4,
    )


@pytest.mark.usefixtures("fake_exporter_open")
def test_sender_stamps_abs_frame_and_streamid() -> None:
    host = FakeTDHost(params={"Active": True, "Debug": False, "Streamid": 4}, tops={"ExportBuffer": _top()})
    engine = _sender(host)
    assert engine.export_frame() is False  # lazy-init probe
    host.abs_frame_value = INFO_CHOP_FRAME_MOD + 17  # one wrap of the Info CHOP channel
    assert engine.export_frame() is True

    frame = engine._exporter.frames[-1]
    assert frame.producer_frame == INFO_CHOP_FRAME_MOD + 17
    assert frame.stream_id == 4
    channels = engine.slot_meta_channels()
    assert tuple(channels) == SLOT_META_CHANNELS
    assert channels["producer_frame"] == 17.0  # mod 2^24
    assert channels["stream_id"] == 4.0
    assert channels["latency_us"] == 0.0


@pytest.mark.usefixtures("fake_exporter_open")
@pytest.mark.parametrize("streamid", [None, "x", -3, 1 << 32], ids=["unset", "non_numeric", "negative", "over_u32"])
def test_sender_invalid_streamid_is_zero(streamid: object) -> None:
    host = FakeTDHost(params={"Active": True, "Debug": False, "Streamid": streamid}, tops={"ExportBuffer": _top()})
    engine = _sender(host)
    engine.export_frame()
    assert engine.export_frame() is True
    assert engine._exporter.frames[-1].stream_id == 0


@pytest.mark.usefixtures("fake_exporter_open")
def test_sender_without_abs_frame_lets_exporter_use_write_idx() -> None:
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": _top()})
    engine = _sender(host)
    engine.export_frame()
    assert engine.export_frame() is True
    assert engine._exporter.frames[-1].producer_frame is None


def test_sender_channels_before_init_are_zero() -> None:
    engine = _sender(FakeTDHost(params={"Active": True}))
    assert engine.slot_meta_channels() == dict.fromkeys(SLOT_META_CHANNELS, 0.0)


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


def _receiver() -> TDReceiverEngine:
    return TDReceiverEngine(
        host=FakeTDHost(params={"Active": True, "Debug": False}),
        config=TDSenderConfig(),
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=f"test_slot_meta_rx_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )


def _meta(pf: int, sid: int = 2, ts: float = 0.0) -> SlotMeta:
    return SlotMeta(ts, pf, 0, sid, 0)


def test_receiver_counts_gaps_and_skips() -> None:
    rx = _receiver()
    rx._record_slot_meta(0, 1, _meta(100))  # first frame
    rx._record_slot_meta(1, 2, _meta(101))  # next
    rx._record_slot_meta(2, 4, _meta(103))  # ring gap
    rx._record_slot_meta(4, 5, _meta(106))  # sender skipped frames
    rx._record_slot_meta(0, 1, _meta(5))  # reconnect: first frame again
    assert rx.frame_gap_count == 1
    assert rx.producer_skip_count == 1
    assert rx.last_slot_meta.producer_frame == 5


def test_receiver_channels_match_in_top_names() -> None:
    rx = _receiver()
    wrapped_frame = 3 * INFO_CHOP_FRAME_MOD + 9  # three wraps of the Info CHOP channel
    rx._record_slot_meta(0, 1, _meta(wrapped_frame, sid=7))
    channels = rx.slot_meta_channels()
    assert tuple(channels) == SLOT_META_CHANNELS
    assert channels["producer_frame"] == 9.0
    assert channels["stream_id"] == 7.0
    assert channels["latency_us"] == 0.0  # sender wrote no host_timestamp
    assert rx.last_slot_meta.producer_frame == wrapped_frame


def test_receiver_latency_from_host_timestamp() -> None:
    import time

    rx = _receiver()
    rx._record_slot_meta(0, 1, _meta(1, ts=time.perf_counter() - 0.002))
    assert 1_500.0 < rx.last_latency_us < 1_000_000.0


def test_receiver_producer_frame_channel_is_exact_in_float32() -> None:
    import numpy as np

    rx = _receiver()
    rx._record_slot_meta(0, 1, _meta((1 << 40) + INFO_CHOP_FRAME_MOD - 1))
    value = rx.slot_meta_channels()["producer_frame"]
    assert float(np.float32(value)) == value == INFO_CHOP_FRAME_MOD - 1


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class _Chan(list):
    def __init__(self) -> None:
        super().__init__([0.0])


class _ScriptChop:
    def __init__(self) -> None:
        self.chans: dict[str, _Chan] = {}
        self.numSamples = 0
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1
        self.chans.clear()

    def appendChan(self, name: str) -> _Chan:
        chan = _Chan()
        self.chans[name] = chan
        return chan


def test_script_chop_writes_one_sample_per_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    import script_chop_callbacks as cb

    ext = types.SimpleNamespace(slot_meta_channels=lambda: {"producer_frame": 12.0, "stream_id": 3.0})
    monkeypatch.setattr(cb, "_extension", lambda: ext)
    chop = _ScriptChop()
    cb.onCook(chop)
    assert chop.numSamples == 1
    assert {k: v[0] for k, v in chop.chans.items()} == {"producer_frame": 12.0, "stream_id": 3.0}


def test_script_chop_without_extension_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import script_chop_callbacks as cb

    monkeypatch.setattr(cb, "_extension", lambda: None)
    chop = _ScriptChop()
    cb.onCook(chop)
    assert chop.cleared == 1
    assert chop.chans == {}


def test_script_top_force_cooks_slot_meta_chop() -> None:
    import script_top_callbacks as cb

    cooked: list[bool] = []
    chop = types.SimpleNamespace(cook=lambda force=False: cooked.append(force))
    parent = types.SimpleNamespace(op=lambda name: chop if name == "slot_meta" else None)
    cb._cook_slot_meta(types.SimpleNamespace(parent=lambda: parent))
    assert cooked == [True]


def test_execute_dat_cooks_slot_meta_after_export(monkeypatch: pytest.MonkeyPatch) -> None:
    import callbacks_template as cb

    calls: list[str] = []
    published = [True]

    def _export() -> bool:
        calls.append("export")
        return published[0]

    ext = types.SimpleNamespace(mode="Sender", export_frame=_export)
    chop = types.SimpleNamespace(cook=lambda force=False: calls.append(f"cook:{force}"))
    owner = types.SimpleNamespace(op=lambda name: chop if name == "slot_meta" else None)
    monkeypatch.setattr(cb, "_extension", lambda: ext)
    monkeypatch.setattr(cb, "parent", lambda: owner, raising=False)
    cb.onFrameEnd(1)
    assert calls == ["export", "cook:True"]

    # Nothing published (inactive, backpressure): the CHOP values did not change, no cook.
    calls.clear()
    published[0] = False
    cb.onFrameEnd(2)
    assert calls == ["export"]


def test_receiver_dropped_frame_counts_as_gap_not_producer_skip() -> None:
    """A frame the copy drops never reaches the output: the next step is from the last shown frame."""
    rx = _receiver()
    rx._record_slot_meta(rx._shown_write_idx, 1, _meta(100))
    rx._shown_write_idx = 1
    # write_idx 2 (producer_frame 101) was acquired but its copy failed: import_frame
    # returned before _record_slot_meta, so _shown_write_idx is still 1.
    rx._record_slot_meta(rx._shown_write_idx, 3, _meta(102))
    assert rx.frame_gap_count == 1
    assert rx.producer_skip_count == 0


def _raise_lookup(name: str) -> None:
    raise RuntimeError("no such op")


@pytest.mark.parametrize("op", [_raise_lookup, lambda name: None], ids=["op_lookup_raises", "no_slot_meta_chop"])
def test_script_top_slot_meta_cook_without_chop_never_raises(op) -> None:  # smell: ignore[no-assert]
    """The contract is "returns without raising": a cook callback must never raise, and with
    no CHOP there is nothing else to observe."""
    import script_top_callbacks as cb

    cb._cook_slot_meta(types.SimpleNamespace(parent=lambda: types.SimpleNamespace(op=op)))


def test_script_top_slot_meta_cook_swallows_a_failing_cook() -> None:
    import script_top_callbacks as cb

    cook_calls: list[bool] = []

    def _cook(force: bool = False) -> None:
        cook_calls.append(force)
        raise RuntimeError("cook failed")

    chop = types.SimpleNamespace(cook=_cook)
    cb._cook_slot_meta(types.SimpleNamespace(parent=lambda: types.SimpleNamespace(op=lambda name: chop)))
    assert cook_calls == [True]
