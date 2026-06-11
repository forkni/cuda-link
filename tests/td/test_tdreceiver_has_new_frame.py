"""
P11 unit tests — TDReceiverEngine.has_new_frame().

has_new_frame() is the cheap SHM probe called from onFrameStart before
import_buffer.cook(force=True).  It must return False ONLY when the sender
has not written a new frame (idle case).  Every "must act" path must return True.

Branch coverage:
  - uninitialized engine               → True  (retry/init path must run)
  - no SHM handle                      → True  (let import_frame handle it)
  - idle   (write_idx == cursor,
             version matches,
             shutdown == 0)             → False (cook skipped ← load-bearing)
  - new frame (write_idx advanced)     → True
  - version change                     → True
  - shutdown flag set                  → True

Harness: reuses _make_receiver, _inject_receiver_connection, and _write_shm_frame
from test_tdreceiver_dtype_refresh.  No real CUDA required.
"""

from __future__ import annotations

import contextlib
import struct
import sys
import uuid
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Shared harness (mirrors test_tdreceiver_dtype_refresh)
# ---------------------------------------------------------------------------


def _make_receiver(shm_name: str):
    from TDConfig import TDSenderConfig
    from TDHost import TDHost
    from TDReceiver import TDReceiverEngine

    class _NullHost(TDHost):
        def __init__(self):
            self.cuda_shapes = []

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

        def wrap_top(self, top):
            return top

        def make_cuda_shape(self, width, height, num_comps, data_type):
            shape = type(
                "_Shape",
                (),
                {"width": width, "height": height, "numComps": num_comps, "dataType": data_type},
            )()
            self.cuda_shapes.append(shape)
            return shape

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
    write_idx: int = 0,
) -> None:
    """Inject a minimal ReceiverConnection so the engine is 'initialized'."""
    from TDReceiver import ReceiverConnection

    from cuda_link.importer import Format
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT, Metadata, SHMLayout

    layout = SHMLayout(num_slots=1)
    W, H, C = 64, 64, 4
    md = Metadata(
        width=W,
        height=H,
        num_comps=C,
        format_kind=FORMAT_KIND_FLOAT,
        bits_per_comp=32,
        flags=0,
        data_size=W * H * C * 4,
    )
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
        last_write_idx=write_idx,
    )
    engine._format = Format.from_metadata(md)
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
# Helper: write a concrete version and write_idx into SHM
# ---------------------------------------------------------------------------


def _set_shm_write_idx(shm: SharedMemory, idx: int) -> None:
    """Overwrite write_idx field at WRITE_IDX_OFFSET (little-endian uint32)."""
    from cuda_link.shm_protocol import WRITE_IDX_OFFSET

    struct.pack_into("<I", shm.buf, WRITE_IDX_OFFSET, idx)


def _set_shm_version(shm: SharedMemory, version: int) -> None:
    """Overwrite version field at VERSION_OFFSET (little-endian uint64)."""
    from cuda_link.shm_protocol import VERSION_OFFSET

    struct.pack_into("<Q", shm.buf, VERSION_OFFSET, version)


# ---------------------------------------------------------------------------
# has_new_frame branch tests
# ---------------------------------------------------------------------------


def test_has_new_frame_uninitialized(shm_cleanup: SharedMemory) -> None:
    """Uninitialized engine must return True — the retry/init path must run."""

    engine = _make_receiver(f"test_hnf_uninit_{uuid.uuid4().hex[:8]}")
    assert not engine._initialized, "engine must start uninitialized"
    assert engine.has_new_frame() is True


def test_has_new_frame_no_shm_handle(shm_cleanup: SharedMemory) -> None:
    """Initialized engine with shm_handle=None must return True (let import_frame decide)."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    _write_shm_frame(
        shm,
        version=1,
        width=64,
        height=64,
        channels=4,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=64 * 64 * 4 * 4,
    )
    engine = _make_receiver(f"test_hnf_noshm_{uuid.uuid4().hex[:8]}")
    _inject_receiver_connection(engine, shm, ipc_version=1)
    engine._connection.shm_handle = None  # remove SHM reference

    assert engine.has_new_frame() is True


def test_has_new_frame_idle_returns_false(shm_cleanup: SharedMemory) -> None:
    """LOAD-BEARING: idle state (write_idx unchanged, version matches, shutdown=0) → False.

    This is the only case where the cook is skipped.  It must be False or the
    optimization does nothing.
    """
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    VERSION = 1
    WRITE_IDX = 5  # nonzero — 0 means 'no frames yet', protocol treats it as NO_FRAME

    _write_shm_frame(
        shm,
        version=VERSION,
        width=64,
        height=64,
        channels=4,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=64 * 64 * 4 * 4,
    )
    _set_shm_version(shm, VERSION)
    _set_shm_write_idx(shm, WRITE_IDX)
    # shutdown byte remains 0 (layout.build_buffer initializes it to 0)

    engine = _make_receiver(f"test_hnf_idle_{uuid.uuid4().hex[:8]}")
    _inject_receiver_connection(engine, shm, ipc_version=VERSION, write_idx=WRITE_IDX)
    # cursor matches SHM write_idx → idle

    result = engine.has_new_frame()
    assert result is False, (
        "has_new_frame() must return False when write_idx == cursor, version matches, "
        "and shutdown flag is 0 — this is what skips the cook"
    )


def test_has_new_frame_new_frame(shm_cleanup: SharedMemory) -> None:
    """New frame (write_idx advanced beyond cursor) must return True."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    VERSION = 1
    _write_shm_frame(
        shm,
        version=VERSION,
        width=64,
        height=64,
        channels=4,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=64 * 64 * 4 * 4,
    )
    _set_shm_version(shm, VERSION)
    _set_shm_write_idx(shm, 5)  # SHM says frame 5

    engine = _make_receiver(f"test_hnf_newframe_{uuid.uuid4().hex[:8]}")
    _inject_receiver_connection(engine, shm, ipc_version=VERSION, write_idx=4)
    # cursor=4, SHM write_idx=5 → new frame available

    assert engine.has_new_frame() is True


def test_has_new_frame_version_change(shm_cleanup: SharedMemory) -> None:
    """Version change (sender restart) must return True even if write_idx matches cursor."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    OLD_VERSION = 1
    NEW_VERSION = 2
    WRITE_IDX = 3

    _write_shm_frame(
        shm,
        version=OLD_VERSION,
        width=64,
        height=64,
        channels=4,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=64 * 64 * 4 * 4,
    )
    # Engine thinks version=1; SHM now shows version=2
    _set_shm_version(shm, NEW_VERSION)
    _set_shm_write_idx(shm, WRITE_IDX)

    engine = _make_receiver(f"test_hnf_version_{uuid.uuid4().hex[:8]}")
    _inject_receiver_connection(engine, shm, ipc_version=OLD_VERSION, write_idx=WRITE_IDX)
    # write_idx matches cursor (would be idle), but version differs → must cook

    assert engine.has_new_frame() is True


def test_has_new_frame_shutdown(shm_cleanup: SharedMemory) -> None:
    """Shutdown flag set must return True so import_frame can handle cleanup."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    VERSION = 1
    WRITE_IDX = 7

    _write_shm_frame(
        shm,
        version=VERSION,
        width=64,
        height=64,
        channels=4,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=64 * 64 * 4 * 4,
    )
    _set_shm_version(shm, VERSION)
    _set_shm_write_idx(shm, WRITE_IDX)

    engine = _make_receiver(f"test_hnf_shutdown_{uuid.uuid4().hex[:8]}")
    _inject_receiver_connection(engine, shm, ipc_version=VERSION, write_idx=WRITE_IDX)

    # Set the shutdown byte
    shm.buf[engine._connection.shutdown_offset] = 1

    assert engine.has_new_frame() is True


def test_has_new_frame_idle_then_new_frame(shm_cleanup: SharedMemory) -> None:
    """Transition: idle → new frame.  First call False, second True after write_idx advances."""
    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    shm = shm_cleanup
    VERSION = 1
    WRITE_IDX = 10

    _write_shm_frame(
        shm,
        version=VERSION,
        width=64,
        height=64,
        channels=4,
        format_kind=FORMAT_KIND_FLOAT,
        bits=32,
        flags=0,
        data_size=64 * 64 * 4 * 4,
    )
    _set_shm_version(shm, VERSION)
    _set_shm_write_idx(shm, WRITE_IDX)

    engine = _make_receiver(f"test_hnf_trans_{uuid.uuid4().hex[:8]}")
    _inject_receiver_connection(engine, shm, ipc_version=VERSION, write_idx=WRITE_IDX)

    # Idle
    assert engine.has_new_frame() is False

    # Producer advances write_idx
    _set_shm_write_idx(shm, WRITE_IDX + 1)
    assert engine.has_new_frame() is True
