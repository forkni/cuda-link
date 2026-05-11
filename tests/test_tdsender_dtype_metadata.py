"""
Regression test: _write_metadata_to_shm must use _detected_numpy_dtype as the
primary source of format_kind and bits_per_comp, not the allocation-size ratio.

Bug: When TD reports the same cuda_mem.size for float32 and uint8 (padded
allocation), _write_metadata_to_shm derived bits from data_size/pixel_count
(always 4 bytes/pixel for a float32 allocation), writing kind=2 bits=32 even
when the actual dtype is uint8. Receiver then decoded uint8 bytes as float32,
producing -3.4028e+38 (FLT_MIN) garbage in the normalized split view.
"""

from __future__ import annotations

import contextlib
import sys
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _make_engine(shm_name: str):
    from TDConfig import TDSenderConfig
    from TDHost import TDHost
    from TDSender import TDSenderEngine

    class _NullHost(TDHost):
        def param_value(self, name):
            return {"Active": True, "Debug": False}.get(name)

        def set_param_value(self, name, value):
            pass

        def set_param_enabled(self, name, enabled):
            pass

        def show_custom_only(self, value):
            pass

        def is_active(self):
            return True

        def find_top(self, name):
            return None

        def set_warning_status(self, msg):
            pass

        def set_error_status(self, msg):
            pass

        def clear_status(self):
            pass

        def set_info_status(self, msg):
            pass

    config = TDSenderConfig()
    return TDSenderEngine(
        host=_NullHost(),
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=shm_name,
        verbose=False,
    )


def _inject_shm_state(engine, shm_handle, width, height, channels, data_size, detected_dtype=None):
    from cuda_link.shm_protocol import SHMLayout

    engine.shm_handle = shm_handle
    engine._layout = SHMLayout(num_slots=1)
    engine.width = width
    engine.height = height
    engine.channels = channels
    engine.data_size = data_size
    engine._detected_numpy_dtype = detected_dtype


@pytest.fixture
def shm_cleanup():
    from cuda_link.shm_protocol import SHMLayout

    layout = SHMLayout(num_slots=1)
    shm = SharedMemory(create=True, size=layout.total_size)
    yield shm
    shm.close()
    with contextlib.suppress(FileNotFoundError):
        shm.unlink()


# ---------------------------------------------------------------------------
# Regression: dtype hint must override allocation-ratio bits
# ---------------------------------------------------------------------------


def test_uint8_dtype_hint_overrides_float32_allocation(shm_cleanup: SharedMemory) -> None:
    """
    When cuda_mem.data_type=uint8 but cuda_mem.size=float32_allocation (padded),
    _write_metadata_to_shm must write kind=1 bits=8, not kind=2 bits=32.

    Reproduces the mid-stream dtype-change scenario: TD keeps the float32-sized
    buffer alive when the source format switches to RGBA8, so cuda_mem.size stays
    at W*H*C*4. The old ratio-based derivation computes 4 bytes/pixel → bits=32,
    writing float32 metadata for uint8 data. Receiver then decodes uint8 bytes as
    float32, producing -3.4028e+38 (FLT_MIN) in the normalized split view.
    """
    import numpy as np

    from cuda_link.shm_protocol import FORMAT_KIND_UNSIGNED, Metadata, SHMLayout

    shm = shm_cleanup
    W, H, C = 1920, 800, 4
    float32_allocation = W * H * C * 4  # 24,576,000 — TD's padded float32-sized buffer

    engine = _make_engine(f"test_dtm_u8_{uuid.uuid4().hex[:8]}")
    _inject_shm_state(engine, shm, W, H, C, data_size=float32_allocation, detected_dtype=np.uint8)

    engine._write_metadata_to_shm()

    layout = SHMLayout(num_slots=1)
    meta = Metadata.read_from(memoryview(shm.buf), layout)

    assert meta.format_kind == FORMAT_KIND_UNSIGNED, (
        f"Expected format_kind=1 (UNSIGNED/uint8) but got {meta.format_kind}. "
        "Sender wrote float32 metadata for uint8 data — receiver would decode garbage."
    )
    assert meta.bits_per_comp == 8, (
        f"Expected bits_per_comp=8 (uint8) but got {meta.bits_per_comp}. "
        "Sender used allocation-size ratio instead of cuda_mem.data_type hint."
    )
    expected_size = W * H * C * 1
    assert meta.data_size == expected_size, (
        f"data_size={meta.data_size} != W*H*C*1={expected_size}. Invariant violated."
    )


def test_float32_dtype_hint_writes_correct_metadata(shm_cleanup: SharedMemory) -> None:
    """Normal case: float32 hint + matching allocation → kind=2 bits=32."""
    import numpy as np

    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT, Metadata, SHMLayout

    shm = shm_cleanup
    W, H, C = 1920, 800, 4
    data_size = W * H * C * 4

    engine = _make_engine(f"test_dtm_f32_{uuid.uuid4().hex[:8]}")
    _inject_shm_state(engine, shm, W, H, C, data_size=data_size, detected_dtype=np.float32)

    engine._write_metadata_to_shm()

    layout = SHMLayout(num_slots=1)
    meta = Metadata.read_from(memoryview(shm.buf), layout)

    assert meta.format_kind == FORMAT_KIND_FLOAT
    assert meta.bits_per_comp == 32
    assert meta.data_size == data_size


def test_uint16_dtype_hint_writes_correct_metadata(shm_cleanup: SharedMemory) -> None:
    """uint16 hint → kind=1 bits=16."""
    import numpy as np

    from cuda_link.shm_protocol import FORMAT_KIND_UNSIGNED, Metadata, SHMLayout

    shm = shm_cleanup
    W, H, C = 1920, 800, 4
    data_size = W * H * C * 2

    engine = _make_engine(f"test_dtm_u16_{uuid.uuid4().hex[:8]}")
    _inject_shm_state(engine, shm, W, H, C, data_size=data_size, detected_dtype=np.uint16)

    engine._write_metadata_to_shm()

    layout = SHMLayout(num_slots=1)
    meta = Metadata.read_from(memoryview(shm.buf), layout)

    assert meta.format_kind == FORMAT_KIND_UNSIGNED
    assert meta.bits_per_comp == 16
    assert meta.data_size == data_size


def test_none_dtype_hint_falls_back_to_ratio(shm_cleanup: SharedMemory) -> None:
    """When detected_dtype is None, ratio-based derivation is used as fallback."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT, Metadata, SHMLayout

    shm = shm_cleanup
    W, H, C = 1920, 800, 4
    data_size = W * H * C * 4

    engine = _make_engine(f"test_dtm_none_{uuid.uuid4().hex[:8]}")
    _inject_shm_state(engine, shm, W, H, C, data_size=data_size, detected_dtype=None)

    engine._write_metadata_to_shm()

    layout = SHMLayout(num_slots=1)
    meta = Metadata.read_from(memoryview(shm.buf), layout)

    assert meta.format_kind == FORMAT_KIND_FLOAT
    assert meta.bits_per_comp == 32
    assert meta.data_size == data_size
