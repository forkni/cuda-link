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
import struct
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
    from SHMProtocol import (
        MAGIC_OFFSET,
        NUM_SLOTS_OFFSET,
        PROTOCOL_MAGIC,
        VERSION_OFFSET,
    )

    from cuda_link.shm_protocol import Metadata, SHMLayout

    buf = shm.buf
    struct.pack_into("<I", buf, MAGIC_OFFSET, PROTOCOL_MAGIC)
    struct.pack_into("<Q", buf, VERSION_OFFSET, version)
    struct.pack_into("<I", buf, NUM_SLOTS_OFFSET, 1)

    layout = SHMLayout(num_slots=1)
    buf[layout.shutdown_offset] = 0
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
