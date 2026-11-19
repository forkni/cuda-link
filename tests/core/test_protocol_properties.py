"""Property tests for the protocol v0.6 pure cores (TESTING_GUIDE.md, Phase J4).

Generated inputs for the code the hand-picked tables cover by example:

- slot meta round trip: what publish_frame writes is what acquire_slot reads back, over the full
  field ranges (u64, u64, u32), any finite timestamp, any ring size and write_idx;
- classify_frame_step: the rules in the FrameStep docstring, each as an invariant, plus invariance
  under shifting both write_idx values by the same amount (the counter wraps at 2^32);
- GpuFrame: every value in a slot meta field's range is accepted and kept, every value outside it
  raises ValueError.

Expected values come from the documented contracts, not from the code under test. Run under
``--hypothesis-profile=ci`` for a derandomized, repeatable verdict (tests/conftest.py).
"""

from __future__ import annotations

import re

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from cuda_link import GpuFrame
from cuda_link.shm_protocol import (
    FrameStep,
    SHMLayout,
    SlotMeta,
    SlotState,
    acquire_slot,
    classify_frame_step,
    publish_frame,
)

U32_LIMIT = 1 << 32
U64_LIMIT = 1 << 64
MAX_NUM_SLOTS = 10

u32 = st.integers(min_value=0, max_value=U32_LIMIT - 1)
u64 = st.integers(min_value=0, max_value=U64_LIMIT - 1)
nonzero_write_idx = st.integers(min_value=1, max_value=U32_LIMIT - 1)

# ---------------------------------------------------------------------------
# publish_frame -> acquire_slot
# ---------------------------------------------------------------------------


@given(
    num_slots=st.integers(min_value=1, max_value=MAX_NUM_SLOTS),
    write_idx=nonzero_write_idx,
    host_timestamp=st.floats(allow_nan=False),
    producer_frame=u64,
    source_timestamp_us=u64,
    stream_id=u32,
)
def test_acquire_slot_reads_back_the_published_slot_meta(
    num_slots: int,
    write_idx: int,
    host_timestamp: float,
    producer_frame: int,
    source_timestamp_us: int,
    stream_id: int,
) -> None:
    layout = SHMLayout(num_slots)
    buf = memoryview(layout.build_buffer())
    publish_frame(
        buf,
        layout,
        write_idx,
        host_timestamp=host_timestamp,
        producer_frame=producer_frame,
        source_timestamp_us=source_timestamp_us,
        stream_id=stream_id,
    )

    result = acquire_slot(buf, layout, last_write_idx=0, last_version=0)

    assert (result.state, result.write_idx) == (SlotState.NEW_FRAME, write_idx)
    assert result.slot_meta == SlotMeta(host_timestamp, producer_frame, source_timestamp_us, stream_id, 0)


# ---------------------------------------------------------------------------
# classify_frame_step: the FrameStep docstring rules
# ---------------------------------------------------------------------------


@given(prev_producer_frame=u64, write_idx=u32, producer_frame=u64)
def test_classify_frame_step_without_previous_frame_is_first(
    prev_producer_frame: int, write_idx: int, producer_frame: int
) -> None:
    assert classify_frame_step(0, prev_producer_frame, write_idx, producer_frame) is FrameStep.FIRST


@given(
    prev_write_idx=nonzero_write_idx,
    step=st.integers(min_value=2, max_value=U32_LIMIT - 1),
    prev_producer_frame=u64,
    producer_frame=u64,
)
def test_classify_frame_step_write_idx_step_over_one_is_frame_gap(
    prev_write_idx: int, step: int, prev_producer_frame: int, producer_frame: int
) -> None:
    write_idx = (prev_write_idx + step) % U32_LIMIT
    assert classify_frame_step(prev_write_idx, prev_producer_frame, write_idx, producer_frame) is FrameStep.FRAME_GAP


@given(prev_write_idx=nonzero_write_idx, prev_producer_frame=u64, producer_frame=u64)
def test_classify_frame_step_repeated_write_idx_is_next(
    prev_write_idx: int, prev_producer_frame: int, producer_frame: int
) -> None:
    """Step 0 is neither a gap nor a skip, whatever producer_frame says."""
    assert classify_frame_step(prev_write_idx, prev_producer_frame, prev_write_idx, producer_frame) is FrameStep.NEXT


@st.composite
def _skipped_producer_frames(draw: st.DrawFn) -> tuple[int, int]:
    """(prev, current) producer frames, both set, with current >= prev + 2, inside u64."""
    prev = draw(st.integers(min_value=1, max_value=U64_LIMIT - 3))
    return prev, draw(st.integers(min_value=prev + 2, max_value=U64_LIMIT - 1))


@st.composite
def _repeated_or_restarted_producer_frames(draw: st.DrawFn) -> tuple[int, int]:
    """(prev, current) with prev set and current <= prev (repeat, restart, or 0 = not set)."""
    prev = draw(st.integers(min_value=1, max_value=U64_LIMIT - 1))
    return prev, draw(st.integers(min_value=0, max_value=prev))


@given(prev_write_idx=nonzero_write_idx, frames=_skipped_producer_frames())
def test_classify_frame_step_step_one_with_producer_jump_is_producer_skip(
    prev_write_idx: int, frames: tuple[int, int]
) -> None:
    prev_producer_frame, producer_frame = frames
    write_idx = (prev_write_idx + 1) % U32_LIMIT
    assert (
        classify_frame_step(prev_write_idx, prev_producer_frame, write_idx, producer_frame) is FrameStep.PRODUCER_SKIP
    )


@given(prev_write_idx=nonzero_write_idx, prev_producer_frame=st.integers(min_value=1, max_value=U64_LIMIT - 2))
def test_classify_frame_step_steady_state_is_next(prev_write_idx: int, prev_producer_frame: int) -> None:
    """write_idx +1 and producer_frame +1, from any previous frame number (odd or even)."""
    write_idx = (prev_write_idx + 1) % U32_LIMIT
    producer_frame = prev_producer_frame + 1
    assert classify_frame_step(prev_write_idx, prev_producer_frame, write_idx, producer_frame) is FrameStep.NEXT


@given(prev_write_idx=nonzero_write_idx, frames=_repeated_or_restarted_producer_frames())
def test_classify_frame_step_step_one_with_repeated_or_restarted_producer_frame_is_next(
    prev_write_idx: int, frames: tuple[int, int]
) -> None:
    prev_producer_frame, producer_frame = frames
    write_idx = (prev_write_idx + 1) % U32_LIMIT
    assert classify_frame_step(prev_write_idx, prev_producer_frame, write_idx, producer_frame) is FrameStep.NEXT


@given(prev_write_idx=nonzero_write_idx, producer_frame=u64)
def test_classify_frame_step_unset_previous_producer_frame_is_never_a_skip(
    prev_write_idx: int, producer_frame: int
) -> None:
    write_idx = (prev_write_idx + 1) % U32_LIMIT
    assert classify_frame_step(prev_write_idx, 0, write_idx, producer_frame) is FrameStep.NEXT


@given(
    prev_write_idx=nonzero_write_idx,
    write_idx=u32,
    shift=u32,
    prev_producer_frame=u64,
    producer_frame=u64,
)
def test_classify_frame_step_is_invariant_under_a_write_idx_shift(
    prev_write_idx: int, write_idx: int, shift: int, prev_producer_frame: int, producer_frame: int
) -> None:
    """Only the step between the two write_idx values matters, across the 2^32 wrap."""
    shifted_prev = (prev_write_idx + shift) % U32_LIMIT
    assume(shifted_prev != 0)  # 0 means "no previous frame"; one draw in 2^32
    shifted = (write_idx + shift) % U32_LIMIT

    original = classify_frame_step(prev_write_idx, prev_producer_frame, write_idx, producer_frame)
    moved = classify_frame_step(shifted_prev, prev_producer_frame, shifted, producer_frame)

    assert moved is original


# ---------------------------------------------------------------------------
# GpuFrame slot meta range checks
# ---------------------------------------------------------------------------

# (field, exclusive limit, the range as the GpuFrame docstring states it)
_SLOT_META_FIELDS = [
    pytest.param("producer_frame", U64_LIMIT, "0..2^64-1", id="producer_frame"),
    pytest.param("source_timestamp_us", U64_LIMIT, "0..2^64-1", id="source_timestamp_us"),
    pytest.param("stream_id", U32_LIMIT, "0..2^32-1", id="stream_id"),
]


@pytest.mark.parametrize(("field", "limit", "documented_range"), _SLOT_META_FIELDS)
@given(data=st.data())
def test_gpuframe_keeps_every_in_range_slot_meta_value(
    field: str, limit: int, documented_range: str, data: st.DataObject
) -> None:
    value = data.draw(st.integers(min_value=0, max_value=limit - 1), label=field)
    frame = GpuFrame(ptr=0, size=4, **{field: value})
    assert getattr(frame, field) == value


@pytest.mark.parametrize(("field", "limit", "documented_range"), _SLOT_META_FIELDS)
@given(data=st.data())
def test_gpuframe_rejects_every_out_of_range_slot_meta_value(
    field: str, limit: int, documented_range: str, data: st.DataObject
) -> None:
    """The error names the field and states its documented range."""
    value = data.draw(st.one_of(st.integers(max_value=-1), st.integers(min_value=limit)), label=field)
    with pytest.raises(ValueError, match=re.escape(f"GpuFrame.{field} must be in {documented_range}")):
        GpuFrame(ptr=0, size=4, **{field: value})
