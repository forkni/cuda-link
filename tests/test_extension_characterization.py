"""
Characterization tests for CUDAIPCExtension — lock observable behaviour before refactoring.

These tests exercise the current CUDAIPCExtension end-to-end through its public surface
so any future refactor that breaks them is caught immediately.

All GPU-touching tests require a physical CUDA device and are marked requires_cuda.
Non-GPU tests (mode defaults, graceful-failure paths) run in any environment.
"""

from __future__ import annotations

import struct

import pytest
from conftest import FakeTDHost, FakeTOPHandle

from cuda_link.shm_protocol import (
    PROTOCOL_MAGIC,
    SHMLayout,
    read_magic,
    read_num_slots,
    read_version,
    read_write_idx,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sender_host(shm_name: str, num_slots: int = 3) -> FakeTDHost:
    return FakeTDHost(
        params={
            "Mode": "Sender",
            "Ipcmemname": shm_name,
            "Numslots": num_slots,
            "Active": True,
            "Debug": False,
            "Hidebuiltin": False,
        }
    )


def _make_receiver_host(shm_name: str, num_slots: int = 3) -> FakeTDHost:
    return FakeTDHost(
        params={
            "Mode": "Receiver",
            "Ipcmemname": shm_name,
            "Numslots": num_slots,
            "Active": True,
            "Debug": False,
        }
    )


def _shm_header(buf: memoryview) -> tuple[int, int, int, int]:
    """Parse (magic, version, num_slots, write_idx) from raw SHM buffer."""
    return read_magic(buf), read_version(buf), read_num_slots(buf), read_write_idx(buf)


def _shutdown_byte(buf: memoryview, num_slots: int) -> int:
    """Return the shutdown flag byte (1 = shutdown signalled)."""
    return buf[SHMLayout(num_slots).shutdown_offset]


# ---------------------------------------------------------------------------
# Sender — initialization and binary layout
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_sender_initialize_sets_correct_shm_layout(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """initialize() writes a well-formed SHM header (magic, version, num_slots, write_idx=0)."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host(temp_shm_name, num_slots=3)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    assert ext.initialize(width=16, height=16, channels=4)

    buf = ext._engine.shm_handle.buf
    magic, version, num_slots, write_idx = _shm_header(buf)

    assert magic == PROTOCOL_MAGIC, f"SHM magic mismatch: {magic:#x}"
    assert version >= 1, "SHM version must be ≥ 1 after init"
    assert num_slots == 3
    assert write_idx == 0, "write_idx must be 0 directly after init"
    assert _shutdown_byte(buf, 3) == 0, "shutdown flag must be 0 after init"

    ext.cleanup()


@pytest.mark.requires_cuda
def test_sender_shm_size_matches_protocol(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """SHM allocation is exactly header + N*slot + shutdown + metadata + timestamp bytes."""
    from CUDAIPCExtension import CUDAIPCExtension

    NUM_SLOTS = 3
    host = _make_sender_host(temp_shm_name, num_slots=NUM_SLOTS)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    ext.initialize(width=16, height=16, channels=4)

    # 20 header + 3*128 slots + 1 shutdown + 20 metadata + 8 timestamp = 433
    expected = SHMLayout(NUM_SLOTS).total_size
    assert len(ext._engine.shm_handle.buf) >= expected, f"SHM too small: {len(ext._engine.shm_handle.buf)} < {expected}"

    ext.cleanup()


# ---------------------------------------------------------------------------
# Sender — export_frame and write_idx monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_sender_export_loop_write_idx_monotone(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """export_frame() increments write_idx by 1 each call and the value is visible in SHM."""
    from CUDAIPCExtension import CUDAIPCExtension

    NUM_SLOTS = 3
    W, H = 8, 8
    host = _make_sender_host(temp_shm_name, num_slots=NUM_SLOTS)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    ext.initialize(width=W, height=H, channels=4)

    # Provide a real GPU pointer so memcpy succeeds
    gpu_ptr = cuda_runtime.malloc(W * H * 4 * 4)  # float32 RGBA
    host._tops["ExportBuffer"] = FakeTOPHandle(width=W, height=H, channels=4, gpu_ptr=gpu_ptr.value)

    NUM_FRAMES = 7
    try:
        prev_write_idx = 0
        for frame in range(NUM_FRAMES):
            ok = ext.export_frame()
            assert ok, f"export_frame() returned False on frame {frame}"

            raw_wi = struct.unpack_from("<I", ext._engine.shm_handle.buf, 16)[0]
            assert raw_wi == prev_write_idx + 1, (
                f"SHM write_idx expected {prev_write_idx + 1}, got {raw_wi} on frame {frame}"
            )
            assert ext._engine.write_idx == raw_wi, "in-memory write_idx must match SHM value"
            prev_write_idx = raw_wi
    finally:
        cuda_runtime.free(gpu_ptr)
        ext.cleanup()


@pytest.mark.requires_cuda
def test_sender_slot_wraps_at_num_slots(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """Slot index wraps via modulo: after num_slots frames the ring starts over."""
    from CUDAIPCExtension import CUDAIPCExtension

    NUM_SLOTS = 3
    W, H = 8, 8
    host = _make_sender_host(temp_shm_name, num_slots=NUM_SLOTS)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    ext.initialize(width=W, height=H, channels=4)

    gpu_ptr = cuda_runtime.malloc(W * H * 4 * 4)
    host._tops["ExportBuffer"] = FakeTOPHandle(width=W, height=H, channels=4, gpu_ptr=gpu_ptr.value)

    # Expected slot sequence for 6 frames with 3 slots: 0 1 2 0 1 2
    expected_slots = [i % NUM_SLOTS for i in range(6)]
    try:
        for i, expected_slot in enumerate(expected_slots):
            slot_before = ext._engine.write_idx % NUM_SLOTS
            assert slot_before == expected_slot, f"Frame {i}: expected slot {expected_slot}, got {slot_before}"
            ext.export_frame()
    finally:
        cuda_runtime.free(gpu_ptr)
        ext.cleanup()


# ---------------------------------------------------------------------------
# Sender — cleanup ordering invariants
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_sender_cleanup_resets_all_state(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """After cleanup(): _initialized=False, dev_ptrs=[], shm_handle=None, write_idx=0."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host(temp_shm_name, num_slots=3)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    ext.initialize(width=16, height=16, channels=4)

    assert ext._engine._initialized
    assert ext._engine.shm_handle is not None

    ext.cleanup()

    assert not ext._engine._initialized, "_initialized must be False after cleanup"
    assert ext._engine.dev_ptrs == [], "dev_ptrs must be empty list after cleanup"
    assert ext._engine.shm_handle is None, "shm_handle must be None after cleanup"
    assert ext._engine.write_idx == 0, "write_idx must reset to 0 after cleanup"
    assert ext._engine.frame_count == 0, "frame_count must reset to 0 after cleanup"
    assert not ext.is_ready(), "is_ready() must return False after cleanup"


@pytest.mark.requires_cuda
def test_sender_cleanup_sets_shutdown_flag(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """cleanup() writes shutdown_flag=1 before releasing SharedMemory."""
    from multiprocessing.shared_memory import SharedMemory

    from CUDAIPCExtension import CUDAIPCExtension

    NUM_SLOTS = 3
    host = _make_sender_host(temp_shm_name, num_slots=NUM_SLOTS)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    ext.initialize(width=16, height=16, channels=4)

    # Capture the SHM name before cleanup (needed to re-open after close)
    shm_name = ext.shm_name

    # Temporarily open a second handle so we can read after ext.cleanup() closes its handle
    shm_reader = SharedMemory(name=shm_name)
    try:
        shutdown_offset = SHMLayout(NUM_SLOTS).shutdown_offset
        assert shm_reader.buf[shutdown_offset] == 0, "shutdown flag should be 0 before cleanup"
        # cleanup() writes flag=1, then closes the sender's SHM handle, then unlinks.
        # Our reader's handle keeps the Windows kernel object alive so we can still read.
        ext.cleanup()
        # After sender cleanup the buffer should show flag=1
        # (Windows: SHM persists until our reader closes it)
        assert shm_reader.buf[shutdown_offset] == 1, "shutdown flag must be 1 after cleanup"
    finally:
        shm_reader.close()
        # Don't unlink — sender already did (or it's Windows no-op; cleanup handles it)


@pytest.mark.requires_cuda
def test_sender_double_cleanup_is_idempotent(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """cleanup() called twice does not raise and leaves state consistent."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host(temp_shm_name, num_slots=3)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    ext.initialize(width=16, height=16, channels=4)

    ext.cleanup()
    ext.cleanup()  # Must not raise

    assert not ext._engine._initialized
    assert not ext.is_ready()


# ---------------------------------------------------------------------------
# switch_mode — state leak prevention
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_switch_mode_sender_to_receiver_no_shm_leak(
    cuda_runtime: object,
    temp_shm_name: str,
    shared_memory_cleanup: list[str],
) -> None:
    """After switch_mode(Receiver): shm_handle=None, dev_ptrs=[], _graph_execs all None."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host(temp_shm_name, num_slots=3)
    shared_memory_cleanup.append(temp_shm_name)

    ext = CUDAIPCExtension(None, host=host)
    ext._engine.cuda = cuda_runtime
    ext.initialize(width=16, height=16, channels=4)

    assert ext._engine._initialized
    assert ext.mode == "Sender"

    ext.switch_mode("Receiver")

    assert ext.mode == "Receiver"
    assert not ext._engine._initialized, "_initialized must be False after mode switch"
    assert ext._engine.shm_handle is None, "shm_handle must be None after mode switch"
    assert ext._engine.dev_ptrs == [], "dev_ptrs must be empty after Sender cleanup"
    # After mode switch the old sender engine is discarded; engine is now TDReceiverEngine
    from TDReceiver import TDReceiverEngine

    assert isinstance(ext._engine, TDReceiverEngine), "engine must be receiver after mode switch"
    assert not hasattr(ext._engine, "_graph_execs"), "no dangling graph execs on receiver engine"


def test_switch_mode_receiver_to_sender_resets_rx_state() -> None:
    """switch_mode(Sender) resets receiver state and resizes sender arrays to num_slots."""
    from CUDAIPCExtension import CUDAIPCExtension

    # Receiver mode — no CUDA needed for this structural test
    host = _make_receiver_host("dummy_shm", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)

    assert ext.mode == "Receiver"

    # Poke format state directly on the value object
    ext._engine._format.width = 1920
    ext._engine._format.height = 1080

    ext.switch_mode("Sender")

    assert ext.mode == "Sender"
    # After engine replacement receiver state is gone (new sender engine has no _format attr)
    # Sender arrays sized to num_slots
    from TDSender import TDSenderEngine

    assert isinstance(ext._engine, TDSenderEngine), "engine must be sender after mode switch"
    assert len(ext._engine.dev_ptrs) == 3
    assert all(p is None for p in ext._engine.dev_ptrs)
    assert len(ext._engine.ipc_handles) == 3


def test_switch_mode_noop_same_mode() -> None:
    """switch_mode() with the same mode does nothing and does not crash."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host("dummy_shm", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)

    # Should be a silent no-op
    ext.switch_mode("Sender")
    assert ext.mode == "Sender"


# ---------------------------------------------------------------------------
# Receiver — graceful failure paths (no CUDA required)
# ---------------------------------------------------------------------------


def test_receiver_init_fails_gracefully_when_no_shm() -> None:
    """initialize_receiver() returns False when SharedMemory does not exist (no crash)."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_receiver_host("nonexistent_shm_xyz_12345", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)

    result = ext.initialize_receiver()

    assert result is False, "initialize_receiver() must return False when SHM absent"
    assert not ext._engine._initialized, "_initialized must remain False"
    assert not ext.is_ready(), "is_ready() must return False"


def test_receiver_import_frame_lazy_retry_without_shm() -> None:
    """import_frame() with no SHM triggers retry logic and returns False (no crash)."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_receiver_host("nonexistent_shm_xyz_12345", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)

    # First call: frames_since_last_retry=0 → increments to 1 ≥ 1 → attempt → fail
    result = ext.import_frame(None)  # type: ignore[arg-type]

    assert result is False, "import_frame() must return False when uninitialized and SHM absent"


def test_receiver_import_frame_shutdown_byte_calls_cleanup() -> None:
    """import_frame() must call cleanup() when the SHM shutdown byte is set.

    Drives the shutdown-detection branch (TDReceiver:191-194). Pre-fix this path
    called self.cleanup_receiver() which no longer exists on TDReceiverEngine after
    the engine-split refactor, producing an AttributeError every time the sender
    disconnected or restarted.
    """
    from multiprocessing import shared_memory

    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_receiver_host("dummy_shm_xyz", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)
    engine = ext._engine

    shm = shared_memory.SharedMemory(name="test_rx_shutdown_regression_xyz", create=True, size=1024)
    try:
        engine.shm_handle = shm  # uses shm_handle property → _connection.shm_handle
        layout = SHMLayout(0)
        engine._connection.num_slots = 0
        engine._connection.layout = layout
        engine._connection.shutdown_offset = layout.shutdown_offset
        engine._connection.ipc_version = 1
        engine._connection.dev_ptrs = []
        engine._connection.ipc_events = []
        engine._connection.stream = None
        engine.cuda = None
        engine._initialized = True

        shm.buf[layout.shutdown_offset] = 1  # mark shutdown byte

        result = engine.import_frame(None)  # type: ignore[arg-type]
        assert result is False
        assert engine._initialized is False, "cleanup() must reset _initialized"
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# Public API shape — no CUDA required
# ---------------------------------------------------------------------------


def test_mode_property_reflects_owner_par() -> None:
    """mode property returns the Mode parameter value from host."""
    from CUDAIPCExtension import CUDAIPCExtension

    for mode in ("Sender", "Receiver"):
        host = FakeTDHost(params={"Mode": mode, "Ipcmemname": "x", "Numslots": 3, "Active": True})
        ext = CUDAIPCExtension(None, host=host)
        assert ext.mode == mode


def test_get_stats_sender_contains_required_keys() -> None:
    """get_stats() in Sender mode returns all expected keys."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host("dummy_shm", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)
    stats = ext.get_stats()

    required = {
        "mode",
        "initialized",
        "frame_count",
        "shm_name",
        "num_slots",
        "buffer_size_mb",
        "resolution",
        "write_idx",
        "dev_ptrs",
    }
    missing = required - stats.keys()
    assert not missing, f"get_stats() missing keys: {missing}"


def test_get_stats_receiver_contains_required_keys() -> None:
    """get_stats() in Receiver mode returns all expected keys."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_receiver_host("dummy", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)
    stats = ext.get_stats()

    required = {
        "mode",
        "initialized",
        "frame_count",
        "shm_name",
        "num_slots",
        "rx_resolution",
        "rx_buffer_size_mb",
        "rx_last_write_idx",
        "rx_dev_ptrs",
    }
    missing = required - stats.keys()
    assert not missing, f"get_stats() missing Receiver keys: {missing}"


def test_is_ready_false_before_init() -> None:
    """is_ready() returns False before any initialization."""
    from CUDAIPCExtension import CUDAIPCExtension

    for mode in ("Sender", "Receiver"):
        host = FakeTDHost(params={"Mode": mode, "Ipcmemname": "x", "Numslots": 3, "Active": True})
        ext = CUDAIPCExtension(None, host=host)
        assert not ext.is_ready(), f"is_ready() should be False before init in {mode} mode"


# ---------------------------------------------------------------------------
# Facade mode-gating — wrong-mode calls must be safe no-ops
# ---------------------------------------------------------------------------


def test_check_deferred_cleanup_no_attr_error_in_sender_mode() -> None:
    """_check_deferred_cleanup() must not raise AttributeError when in Sender mode."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host("dummy_shm", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)
    assert ext.mode == "Sender"
    ext._check_deferred_cleanup()  # must not raise


def test_import_frame_returns_false_in_sender_mode() -> None:
    """import_frame() in Sender mode must return False, not crash."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host("dummy_shm", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)
    assert ext.mode == "Sender"
    result = ext.import_frame(None)  # type: ignore[arg-type]
    assert result is False


def test_update_receiver_resolution_no_attr_error_in_receiver_mode() -> None:
    """update_receiver_resolution() must not raise AttributeError when in Receiver mode."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_receiver_host("dummy_shm", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)
    assert ext.mode == "Receiver"
    ext.update_receiver_resolution(None)  # must not raise (needs_resolution_update=False → early return)


def test_export_frame_returns_false_in_receiver_mode() -> None:
    """export_frame() in Receiver mode must return False, not crash."""
    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_receiver_host("dummy_shm", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)
    assert ext.mode == "Receiver"
    result = ext.export_frame()
    assert result is False


# ---------------------------------------------------------------------------
# reconfigure_and_reinit — unit tests (no CUDA required)
# ---------------------------------------------------------------------------


def test_reconfigure_and_reinit_calls_cleanup_then_sets_field() -> None:
    """reconfigure_and_reinit() calls cleanup() then sets the named field via its setter."""
    from unittest.mock import patch

    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_sender_host("original_name", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)

    cleanup_calls: list[None] = []
    with patch.object(ext, "cleanup", side_effect=lambda: cleanup_calls.append(None)):
        ext.reconfigure_and_reinit("shm_name", "new_name")

    assert len(cleanup_calls) == 1, "cleanup() must be called exactly once"
    assert ext.shm_name == "new_name", "field must be updated after cleanup"


def test_reconfigure_and_reinit_triggers_immediate_reconnect_in_receiver_mode() -> None:
    """reconfigure_and_reinit() calls request_immediate_reconnect() in Receiver mode."""
    from unittest.mock import patch

    from CUDAIPCExtension import CUDAIPCExtension

    host = _make_receiver_host("original_name", num_slots=3)
    ext = CUDAIPCExtension(None, host=host)

    reconnect_calls: list[None] = []
    with (
        patch.object(ext, "cleanup"),
        patch.object(ext, "request_immediate_reconnect", side_effect=lambda: reconnect_calls.append(None)),
    ):
        ext.reconfigure_and_reinit("shm_name", "new_name")

    assert len(reconnect_calls) == 1, "request_immediate_reconnect() must be called in Receiver mode"
