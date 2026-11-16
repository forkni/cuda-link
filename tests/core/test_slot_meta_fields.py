"""Protocol v0.6 slot meta fields end to end (Phase 4).

- GpuFrame carries producer_frame / source_timestamp_us / stream_id and checks their range.
- Exporter writes them into the slot meta (producer_frame None -> write_idx) and keeps
  last_slot_meta.
- Importer exposes the acquired frame's slot meta as last_slot_meta.
- classify_frame_step is the twin of cpp_top core/frame_step.h (same case table as
  topcore_tests test_frame_step_classification), and the In TOP Info CHOP channel names
  match SLOT_META_CHANNELS.

GPU-free: FakeCUDAAdapter and fakes.make_connected_importer.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import pytest
from fakes import make_connected_importer

from cuda_link import ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link._importer_port import ImportOutcome
from cuda_link.exporter import Exporter
from cuda_link.shm_protocol import (
    EMPTY_SLOT_META,
    INFO_CHOP_FRAME_MOD,
    SLOT_META_CHANNELS,
    FrameStep,
    SHMLayout,
    classify_frame_step,
    publish_frame,
    read_slot_meta,
)

_REPO = Path(__file__).resolve().parents[2]


def _open_exporter(num_slots: int = 2) -> Exporter:
    return Exporter.open(
        FrameSpec(
            shm_name=f"test_slot_meta_{uuid.uuid4().hex[:8]}",
            height=8,
            width=8,
            channels=4,
            dtype="uint8",
            num_slots=num_slots,
            device=0,
        ),
        policy=ExportPolicy.for_testing(),
        cuda=FakeCUDAAdapter(device=0),
    )


# ---------------------------------------------------------------------------
# GpuFrame
# ---------------------------------------------------------------------------


def test_gpuframe_slot_meta_defaults() -> None:
    frame = GpuFrame(ptr=0, size=4)
    assert frame.producer_frame is None
    assert frame.source_timestamp_us == 0
    assert frame.stream_id == 0


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"producer_frame": -1}, "producer_frame"),
        ({"producer_frame": 1 << 64}, "producer_frame"),
        ({"source_timestamp_us": -1}, "source_timestamp_us"),
        ({"source_timestamp_us": 1 << 64}, "source_timestamp_us"),
        ({"stream_id": -1}, "stream_id"),
        ({"stream_id": 1 << 32}, "stream_id"),
    ],
)
def test_gpuframe_rejects_out_of_range_slot_meta(kwargs: dict, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GpuFrame(ptr=0, size=4, **kwargs)


def test_gpuframe_accepts_range_limits() -> None:
    frame = GpuFrame(
        ptr=0, size=4, producer_frame=(1 << 64) - 1, source_timestamp_us=(1 << 64) - 1, stream_id=(1 << 32) - 1
    )
    assert frame.stream_id == (1 << 32) - 1


# ---------------------------------------------------------------------------
# Exporter -> slot meta on the wire
# ---------------------------------------------------------------------------


def test_exporter_writes_slot_meta_fields() -> None:
    exp = _open_exporter()
    try:
        t_before = time.perf_counter()
        frame = GpuFrame(
            ptr=0, size=exp.data_size, producer_frame=123_456, source_timestamp_us=987_654_321, stream_id=5
        )
        assert exp.export(frame) is FrameOutcome.PUBLISHED
        meta = read_slot_meta(exp.shm_handle.buf, SHMLayout(2), 0)  # write_idx 1 -> slot 0
        assert meta.producer_frame == 123_456
        assert meta.source_timestamp_us == 987_654_321
        assert meta.stream_id == 5
        assert meta.meta_flags == 0
        assert t_before <= meta.host_timestamp <= time.perf_counter()
        assert exp.last_slot_meta == meta
    finally:
        exp.close()


def test_exporter_producer_frame_none_publishes_write_idx() -> None:
    exp = _open_exporter()
    try:
        assert exp.last_slot_meta == EMPTY_SLOT_META
        layout = SHMLayout(2)
        for expected_write_idx in (1, 2, 3):
            assert exp.export(GpuFrame(ptr=0, size=exp.data_size)) is FrameOutcome.PUBLISHED
            slot = (expected_write_idx - 1) % 2
            meta = read_slot_meta(exp.shm_handle.buf, layout, slot)
            assert meta.producer_frame == expected_write_idx
            assert meta.stream_id == 0
    finally:
        exp.close()


def test_exporter_producer_frame_zero_means_not_set() -> None:
    """An explicit 0 is written as 0 ("not set"), not replaced by write_idx."""
    exp = _open_exporter()
    try:
        exp.export(GpuFrame(ptr=0, size=exp.data_size, producer_frame=0))
        assert read_slot_meta(exp.shm_handle.buf, SHMLayout(2), 0).producer_frame == 0
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# Importer.last_slot_meta
# ---------------------------------------------------------------------------


def test_importer_exposes_last_slot_meta() -> None:
    imp = make_connected_importer(shape=(8, 8, 4), dtype="uint8", num_slots=1, write_idx=0)
    assert imp.last_slot_meta == EMPTY_SLOT_META
    buf = imp._conn.shm_handle.buf
    layout = SHMLayout(1)
    publish_frame(
        buf, layout, 1, host_timestamp=time.perf_counter(), producer_frame=777, source_timestamp_us=42, stream_id=3
    )

    result = imp.get_frame_numpy()

    assert result.outcome is ImportOutcome.NEW_FRAME
    assert imp.last_slot_meta.producer_frame == 777
    assert imp.last_slot_meta.source_timestamp_us == 42
    assert imp.last_slot_meta.stream_id == 3


def test_importer_keeps_last_slot_meta_on_no_frame() -> None:
    imp = make_connected_importer(shape=(8, 8, 4), dtype="uint8", num_slots=1, write_idx=0)
    publish_frame(imp._conn.shm_handle.buf, SHMLayout(1), 1, host_timestamp=time.perf_counter(), producer_frame=9)
    assert imp.get_frame_numpy().outcome is ImportOutcome.NEW_FRAME
    assert imp.get_frame_numpy().outcome is ImportOutcome.NO_FRAME
    assert imp.last_slot_meta.producer_frame == 9


# ---------------------------------------------------------------------------
# classify_frame_step: same case table as topcore_tests test_frame_step_classification
# ---------------------------------------------------------------------------

_U32_MAX = 0xFFFFFFFF


@pytest.mark.parametrize(
    ("prev_widx", "prev_pf", "widx", "pf", "expected"),
    [
        (0, 0, 1, 100, FrameStep.FIRST),
        (0, 999, 7, 100, FrameStep.FIRST),
        (5, 100, 6, 101, FrameStep.NEXT),
        (5, 100, 6, 100, FrameStep.NEXT),
        (5, 100, 7, 102, FrameStep.FRAME_GAP),
        (5, 100, 9, 101, FrameStep.FRAME_GAP),
        (5, 100, 6, 102, FrameStep.PRODUCER_SKIP),
        (5, 0, 6, 50, FrameStep.NEXT),
        (5, 100, 6, 0, FrameStep.NEXT),
        (5, 100, 6, 3, FrameStep.NEXT),
        (_U32_MAX, 100, 1, 101, FrameStep.FRAME_GAP),
        (_U32_MAX - 1, 100, _U32_MAX, 101, FrameStep.NEXT),
        (5, 100, 5, 100, FrameStep.NEXT),
    ],
)
def test_classify_frame_step(prev_widx: int, prev_pf: int, widx: int, pf: int, expected: FrameStep) -> None:
    assert classify_frame_step(prev_widx, prev_pf, widx, pf) is expected


def test_cpp_frame_step_cases_match_python_table() -> None:
    """Every classify_frame_step CHECK in topcore_tests is in the Python table above, and back."""
    src = (_REPO / "cpp_top" / "tests" / "topcore_tests.cpp").read_text(encoding="utf-8")
    names = {"First": "FIRST", "Next": "NEXT", "FrameGap": "FRAME_GAP", "ProducerSkip": "PRODUCER_SKIP"}
    cpp_cases = set()
    for m in re.finditer(r"classify_frame_step\(([^)]*)\) == FrameStep::(\w+)", src):
        args = [int(a.strip().rstrip("U"), 0) for a in m.group(1).split(",")]
        cpp_cases.add((*args, names[m.group(2)]))
    params = test_classify_frame_step.pytestmark[0].args[1]
    py_cases = {(a, b, c, d, e.name) for a, b, c, d, e in params}
    assert cpp_cases
    assert cpp_cases == py_cases


def test_in_top_info_chop_names_match_slot_meta_channels() -> None:
    """The In TOP's last five Info CHOP channels are SLOT_META_CHANNELS, in order."""
    src = (_REPO / "cpp_top" / "src" / "in_top" / "CudaLinkInTOP.cpp").read_text(encoding="utf-8")
    body = src[src.index("void CudaLinkInTOP::getInfoCHOPChan") : src.index("bool CudaLinkInTOP::getInfoDATSize")]
    names = re.findall(r'chan->name->setString\("(\w+)"\)', body)
    count = int(re.search(r"constexpr int32_t kNumInfoCHOPChans = (\d+);", src).group(1))
    assert len(names) == count
    assert tuple(names[-len(SLOT_META_CHANNELS) :]) == SLOT_META_CHANNELS


def test_info_chop_frame_mod_matches_cpp() -> None:
    for rel in ("cpp_top/src/in_top/CudaLinkInTOP.cpp", "cpp_top/src/out_top/CudaLinkOutTOP.cpp"):
        src = (_REPO / rel).read_text(encoding="utf-8")
        assert "constexpr uint64_t kInfoChopFrameMod = uint64_t{1} << 24;" in src, rel
    assert INFO_CHOP_FRAME_MOD == 1 << 24
