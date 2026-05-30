"""
Regression test: TDReceiverEngine._refresh_on_version_change must update
self._format (format_kind, bits_per_comp) and self._connection.ipc_version
in-place when the sender bumps the SHM version, without closing the SHM handle.

Bug: The VERSION_CHANGED handler in import_frame called cleanup() + return False,
relying on the next initialize_receiver() to re-read metadata. This path is correct
only if the sender wrote the right metadata — but the sender itself was writing wrong
kind/bits (bits derived from padded allocation, not from cuda_mem.data_type), so the
receiver would re-init with the wrong format anyway. This test validates the receiver's
in-place refresh seam independently of the sender bug.
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


def _make_receiver(shm_name: str):
    from TDConfig import TDSenderConfig
    from TDHost import TDHost
    from TDReceiver import TDReceiverEngine

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
    return TDReceiverEngine(
        host=_NullHost(),
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=shm_name,
        verbose=False,
    )


def _write_shm_frame(
    shm: SharedMemory,
    version: int,
    width: int,
    height: int,
    channels: int,
    format_kind: int,
    bits: int,
    flags: int,
    data_size: int,
) -> None:
    """Write a complete SHM frame (header + slot placeholder + metadata)."""
    from cuda_link.shm_protocol import Metadata, SHMLayout

    layout = SHMLayout(num_slots=1)
    buf = shm.buf
    buf[: layout.total_size] = layout.build_buffer(version=version, write_idx=0)
    Metadata(
        width=width,
        height=height,
        num_comps=channels,
        format_kind=format_kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=data_size,
    ).pack_into(memoryview(buf), layout)


def _inject_receiver_connection(
    engine,
    shm: SharedMemory,
    ipc_version: int,
    format_kind: int,
    bits_per_comp: int,
    width: int,
    height: int,
    channels: int,
    buffer_size: int,
) -> None:
    """Simulate post-initialize_receiver state (no real CUDA handles)."""
    from TDReceiver import FormatDescriptor, ReceiverConnection

    from cuda_link.shm_protocol import SHMLayout

    layout = SHMLayout(num_slots=1)
    engine._connection = ReceiverConnection(
        shm_handle=shm,
        dev_ptrs=[None],
        ipc_handles=[None],
        ipc_events=[None],
        stream=None,
        layout=layout,
        num_slots=1,
        ipc_version=ipc_version,
        shutdown_offset=layout.shutdown_offset,
        last_write_idx=0,
    )
    engine._format = FormatDescriptor(
        width=width,
        height=height,
        num_comps=channels,
        format_kind=format_kind,
        bits_per_comp=bits_per_comp,
        flags=0,
        buffer_size=buffer_size,
    )
    engine._initialized = True


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
# Regression tests — RED before fix, GREEN after
# ---------------------------------------------------------------------------


def test_receiver_refreshes_format_on_version_change(shm_cleanup: SharedMemory) -> None:
    """
    After _refresh_on_version_change(new_version), self._format must reflect the
    new dtype (kind=1 bits=8 for uint8) and ipc_version must advance to new_version.
    SHM must remain open (no cleanup called).
    """
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT, FORMAT_KIND_UNSIGNED

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_{uuid.uuid4().hex[:8]}")

    W, H, C = 1920, 800, 4
    float32_size = W * H * C * 4

    # Step 1: initial SHM state — float32 at version 1
    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=float32_size,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        width=W,
        height=H,
        channels=C,
        buffer_size=float32_size,
    )

    assert engine._format.bits_per_comp == 32

    # Step 2: sender writes new uint8 metadata and bumps version to 2
    uint8_size = W * H * C * 1
    _write_shm_frame(
        shm,
        version=2,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits=8,
        flags=0,
        data_size=uint8_size,
    )

    # Step 3: receiver calls in-place refresh (new method — AttributeError if not implemented)
    result = engine._refresh_on_version_change(2)

    assert result is True, "_refresh_on_version_change must return True for valid metadata"
    assert engine._format.bits_per_comp == 8, (
        f"Expected bits_per_comp=8 after refresh, got {engine._format.bits_per_comp}. "
        "Receiver did not pick up the new dtype."
    )
    assert engine._format.format_kind == FORMAT_KIND_UNSIGNED, (
        f"Expected format_kind=1 (UNSIGNED) after refresh, got {engine._format.format_kind}."
    )
    assert engine._connection.ipc_version == 2, (
        f"ipc_version must be updated to 2, got {engine._connection.ipc_version}. "
        "Receiver would re-detect VERSION_CHANGED on every subsequent frame."
    )
    assert engine._connection.shm_handle is not None, "SHM must NOT be closed by _refresh_on_version_change."


def test_receiver_refresh_normal_case_preserves_format(shm_cleanup: SharedMemory) -> None:
    """Format-identical version bump (e.g. dim-only change) must still advance ipc_version."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_nc_{uuid.uuid4().hex[:8]}")

    W, H, C = 1920, 800, 4
    data_size = W * H * C * 4

    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=data_size,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        width=W,
        height=H,
        channels=C,
        buffer_size=data_size,
    )

    result = engine._refresh_on_version_change(2)

    assert result is True
    assert engine._format.bits_per_comp == 32
    assert engine._format.format_kind == FORMAT_KIND_FLOAT
    assert engine._connection.ipc_version == 2
    assert engine._connection.shm_handle is not None


def test_receiver_refresh_invalid_invariant_returns_false(shm_cleanup: SharedMemory) -> None:
    """Corrupt metadata (invariant fails) must cause refresh to return False."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_inv_{uuid.uuid4().hex[:8]}")

    W, H, C = 1920, 800, 4
    data_size = W * H * C * 4

    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=data_size,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        width=W,
        height=H,
        channels=C,
        buffer_size=data_size,
    )

    # Write metadata with wrong data_size (invariant fails)
    _write_shm_frame(
        shm,
        version=2,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=data_size + 1,
    )  # off by one — invariant fails

    result = engine._refresh_on_version_change(2)

    assert result is False, "Corrupt metadata must cause refresh to return False (fallback to cleanup)"


# ---------------------------------------------------------------------------
# _to_td_pixel_format — par.format string mapping (Bug A fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "format_kind,bits,num_comps,expected",
    [
        (2, 32, 4, "rgba32float"),  # float32 RGBA — canonical streaming format
        (2, 32, 1, "r32float"),  # float32 mono
        (1, 16, 4, "rgba16fixed"),  # uint16 RGBA — "16-bit fixed (RGBA)" in TD
        (1, 16, 1, "r16fixed"),  # uint16 mono — "16-bit fixed (R)" in TD
        (1, 8, 4, "rgba8fixed"),  # uint8 RGBA
        (1, 8, 1, "r8fixed"),  # uint8 mono
        (1, 16, 2, "rg16fixed"),  # uint16 RG
        (2, 32, 2, "rg32float"),  # float32 RG
    ],
)
def test_to_td_pixel_format(format_kind, bits, num_comps, expected):
    """_to_td_pixel_format must produce the correct par.format string for every dtype/channel combo."""
    from TDReceiver import _to_td_pixel_format

    result = _to_td_pixel_format(format_kind, bits, num_comps)
    assert result == expected, (
        f"_to_td_pixel_format({format_kind},{bits},{num_comps}) = {result!r}, expected {expected!r}"
    )


# ---------------------------------------------------------------------------
# needs_format_update flag (Bug A) — set on dtype change, clear on same dtype
# ---------------------------------------------------------------------------


def test_needs_format_update_set_on_bits_change(shm_cleanup: SharedMemory) -> None:
    """_refresh_on_version_change must set needs_format_update when bits_per_comp changes."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT, FORMAT_KIND_UNSIGNED

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_fmtflag_{uuid.uuid4().hex[:8]}")

    W, H, C = 1920, 1080, 4
    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=W * H * C * 4,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        width=W,
        height=H,
        channels=C,
        buffer_size=W * H * C * 4,
    )

    # Write new metadata: float32 → uint16
    _write_shm_frame(
        shm,
        version=2,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits=16,
        flags=0,
        data_size=W * H * C * 2,
    )

    result = engine._refresh_on_version_change(2)

    assert result is True
    assert engine._retry.needs_format_update is True, (
        "needs_format_update must be True when bits_per_comp changes (float32→uint16)"
    )


def test_needs_format_update_set_on_num_comps_change(shm_cleanup: SharedMemory) -> None:
    """needs_format_update must be set when num_comps changes (e.g. RGBA→mono)."""
    from cuda_link.shm_protocol import FORMAT_KIND_UNSIGNED

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_numcomps_{uuid.uuid4().hex[:8]}")

    W, H = 1920, 1080
    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=4,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits=16,
        flags=0,
        data_size=W * H * 4 * 2,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits_per_comp=16,
        width=W,
        height=H,
        channels=4,
        buffer_size=W * H * 4 * 2,
    )

    # Change from 4ch → 1ch (RGBA → mono)
    _write_shm_frame(
        shm,
        version=2,
        width=W,
        height=H,
        channels=1,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits=16,
        flags=0,
        data_size=W * H * 1 * 2,
    )

    result = engine._refresh_on_version_change(2)

    assert result is True
    assert engine._retry.needs_format_update is True, (
        "needs_format_update must be True when num_comps changes (4ch RGBA → 1ch mono)"
    )


def test_needs_format_update_not_set_on_identical_format(shm_cleanup: SharedMemory) -> None:
    """needs_format_update must NOT be set when the format is unchanged (e.g. resolution-only bump)."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_nofmt_{uuid.uuid4().hex[:8]}")

    W, H, C = 1920, 1080, 4
    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=W * H * C * 4,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        width=W,
        height=H,
        channels=C,
        buffer_size=W * H * C * 4,
    )

    # Same format (float32 RGBA) — only version bumped
    _write_shm_frame(
        shm,
        version=2,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=W * H * C * 4,
    )

    engine._retry.needs_format_update = False  # ensure it starts clear
    result = engine._refresh_on_version_change(2)

    assert result is True
    assert engine._retry.needs_format_update is False, "needs_format_update must stay False when format is unchanged"


# ---------------------------------------------------------------------------
# consume_pending_format — the fallback-path entry point (Part 5)
# ---------------------------------------------------------------------------


def test_consume_pending_format_returns_none_when_clear(shm_cleanup: SharedMemory) -> None:
    """consume_pending_format() returns None when no format update is pending."""
    engine = _make_receiver(f"test_rdr_cpf_none_{uuid.uuid4().hex[:8]}")
    # Flag starts False by default
    result = engine.consume_pending_format()
    assert result is None


def test_consume_pending_format_returns_format_and_clears_flag(shm_cleanup: SharedMemory) -> None:
    """After dtype change, consume_pending_format() returns the correct par.format string
    and clears needs_format_update (so the next call returns None — idempotent)."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT, FORMAT_KIND_UNSIGNED

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_cpf_{uuid.uuid4().hex[:8]}")

    W, H, C = 1920, 1080, 4
    # Initial state: float32 RGBA
    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=W * H * C * 4,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        width=W,
        height=H,
        channels=C,
        buffer_size=W * H * C * 4,
    )

    # Version change: float32 → uint16
    _write_shm_frame(
        shm,
        version=2,
        width=W,
        height=H,
        channels=C,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits=16,
        flags=0,
        data_size=W * H * C * 2,
    )
    engine._refresh_on_version_change(2)

    assert engine._retry.needs_format_update is True  # flag set by refresh

    fmt = engine.consume_pending_format()
    assert fmt == "rgba16fixed", (
        f"Expected 'rgba16fixed' for uint16 RGBA, got {fmt!r}. "
        "This is the par.format string the Script TOP must receive."
    )
    # Flag must be cleared — second call returns None
    assert engine.consume_pending_format() is None, (
        "consume_pending_format() must be idempotent: returns None after first consume"
    )
    assert engine._retry.needs_format_update is False


def test_consume_pending_format_mono_uint16(shm_cleanup: SharedMemory) -> None:
    """1ch uint16 (mono) → consume_pending_format() returns 'r16fixed'."""
    from cuda_link.shm_protocol import FORMAT_KIND_UNSIGNED

    shm = shm_cleanup
    engine = _make_receiver(f"test_rdr_cpf_mono_{uuid.uuid4().hex[:8]}")

    W, H = 1920, 1080
    _write_shm_frame(
        shm,
        version=1,
        width=W,
        height=H,
        channels=4,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits=16,
        flags=0,
        data_size=W * H * 4 * 2,
    )
    _inject_receiver_connection(
        engine,
        shm,
        ipc_version=1,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits_per_comp=16,
        width=W,
        height=H,
        channels=4,
        buffer_size=W * H * 4 * 2,
    )

    _write_shm_frame(
        shm,
        version=2,
        width=W,
        height=H,
        channels=1,
        format_kind=FORMAT_KIND_UNSIGNED,
        bits=16,
        flags=0,
        data_size=W * H * 1 * 2,
    )
    engine._refresh_on_version_change(2)

    fmt = engine.consume_pending_format()
    assert fmt == "r16fixed", f"Expected 'r16fixed' for 1ch uint16, got {fmt!r}"
