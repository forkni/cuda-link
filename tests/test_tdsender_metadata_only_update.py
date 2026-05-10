"""
Regression test: metadata-only update branch must produce receiver-invariant-compatible metadata.

Bug: When TD reports the same cuda_mem.size for two frames with different height
(padded/atlas allocation), TDSenderEngine._write_metadata_to_shm() wrote stale
data_size=<allocation> while height reflected the active region, breaking the
receiver's invariant: W*H*C*(bits/8) == data_size.

Root cause: _write_metadata_to_shm() wrote self.data_size (allocation size) to the
metadata data_size field. When pixel_count = new_W*new_H*new_C doesn't divide evenly
into self.data_size, bits falls back to 32, but data_size in metadata still holds the
old allocation size. Receiver invariant: expected = W*H*C*4 != allocation. FAILS.

Fix: write data_size = pixel_count * (bits // 8) so metadata is self-consistent,
matching the Python-side exporter and the v1.0.0 invariant spec.
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
    """Construct a TDSenderEngine with fake/minimal deps (no CUDA, no SHM)."""
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


def _inject_shm_state(engine, shm_handle: SharedMemory, width: int, height: int, channels: int, data_size: int) -> None:
    """Set engine metadata-write state (simulates post-initialize condition)."""
    from cuda_link.shm_protocol import SHMLayout

    engine.shm_handle = shm_handle
    engine._layout = SHMLayout(num_slots=1)
    engine.width = width
    engine.height = height
    engine.channels = channels
    engine.data_size = data_size


@pytest.fixture
def shm_cleanup():
    """Create, yield a fresh SharedMemory, and clean up afterwards."""
    from cuda_link.shm_protocol import (
        SHMLayout,
    )

    layout = SHMLayout(num_slots=1)
    shm = SharedMemory(create=True, size=layout.total_size)
    yield shm
    shm.close()
    with contextlib.suppress(FileNotFoundError):
        shm.unlink()


# ---------------------------------------------------------------------------
# Regression test — must be RED before fix, GREEN after
# ---------------------------------------------------------------------------


def test_metadata_only_update_satisfies_receiver_invariant(shm_cleanup: SharedMemory) -> None:
    """
    After _write_metadata_to_shm(), metadata must satisfy W*H*C*(bits/8) == data_size.

    This is the exact invariant check that TDReceiver.initialize_receiver() performs.
    Failing this test means the receiver would log
    "Metadata size invariant failed … Sender/receiver protocol mismatch."
    and loop indefinitely waiting for the sender.

    Reproduces the padded-allocation scenario: sender initialised at 1920×1080 float32
    (data_size=33,177,600), then height reported by TD dropped to 571 (active region)
    while cuda_mem.size stayed at 33,177,600 (padded allocation).
    """
    from cuda_link.shm_protocol import Metadata, SHMLayout

    shm = shm_cleanup
    engine_name = f"test_mu_{uuid.uuid4().hex[:8]}"
    engine = _make_engine(engine_name)

    # Simulate the stale state that the metadata-only-update elif produces:
    # self.data_size stayed at old allocation size, but dims were updated to new values.
    W, H, C = 1920, 571, 4
    old_allocation = 1920 * 1080 * 4 * 4  # 33,177,600 — padded allocation from previous init

    _inject_shm_state(engine, shm, width=W, height=H, channels=C, data_size=old_allocation)

    engine._write_metadata_to_shm()

    layout = SHMLayout(num_slots=1)
    meta = Metadata.read_from(memoryview(shm.buf), layout)

    # Receiver invariant (exact check from TDReceiver.py:611-616):
    expected_size = meta.width * meta.height * meta.num_comps * (meta.bits_per_comp // 8)
    assert meta.bits_per_comp > 0, "bits_per_comp must be non-zero (sender bug: format undetectable)"
    assert expected_size == meta.data_size, (
        f"Receiver invariant FAILED: W*H*C*(bits/8)={expected_size} "
        f"but metadata.data_size={meta.data_size}. "
        f"Receiver would log 'Metadata size invariant failed' and refuse connection."
    )


def test_metadata_normal_case_unaffected(shm_cleanup: SharedMemory) -> None:
    """Normal case: W*H*C*(bits/8) == self.data_size — result must be unchanged."""
    from cuda_link.shm_protocol import Metadata, SHMLayout

    shm = shm_cleanup
    engine = _make_engine(f"test_mu_norm_{uuid.uuid4().hex[:8]}")

    W, H, C = 1920, 1080, 4
    data_size = W * H * C * 4  # 33,177,600 — exactly divisible, normal case

    _inject_shm_state(engine, shm, width=W, height=H, channels=C, data_size=data_size)
    engine._write_metadata_to_shm()

    layout = SHMLayout(num_slots=1)
    meta = Metadata.read_from(memoryview(shm.buf), layout)

    assert meta.width == W
    assert meta.height == H
    assert meta.num_comps == C
    assert meta.bits_per_comp == 32
    assert meta.data_size == data_size  # must equal exactly in normal case
    expected_size = meta.width * meta.height * meta.num_comps * (meta.bits_per_comp // 8)
    assert expected_size == meta.data_size
