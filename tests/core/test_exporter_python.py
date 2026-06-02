"""
Tests for Exporter (Python-side GPU exporter).

Tests are split into two groups:
- Pure unit tests (no CUDA required): dtype validation, frame spec, layout, etc.
- CUDA integration tests (@pytest.mark.requires_cuda): SHM protocol, ring buffer, etc.
"""

from __future__ import annotations

import struct
import time
import uuid
from multiprocessing.shared_memory import SharedMemory

import pytest


def _make_bare(shm_name: str = "x", height: int = 64, width: int = 64, **spec_kwargs):
    """Construct an Exporter without calling _initialize() — safe for pure unit tests."""
    from cuda_link import ExportPolicy, FrameSpec
    from cuda_link._cuda_adapters import FakeCUDAAdapter
    from cuda_link.exporter import Exporter

    spec = FrameSpec(shm_name=shm_name, height=height, width=width, **spec_kwargs)
    return Exporter(spec, ExportPolicy.for_testing(), FakeCUDAAdapter())


# ---------------------------------------------------------------------------
# Pure unit tests (no CUDA required)
# ---------------------------------------------------------------------------


def test_kind_bits_mapping() -> None:
    """dtype strings map to correct (format_kind, bits, flags) wire encoding.

    The authoritative registry lives in DtypeCodec (shm_protocol) — not in the exporter.
    """
    from cuda_link.shm_protocol import (
        FLAGS_BFLOAT16,
        FORMAT_KIND_FLOAT,
        FORMAT_KIND_SIGNED,
        FORMAT_KIND_UNSIGNED,
        DtypeCodec,
    )

    assert DtypeCodec.encode("float32") == (FORMAT_KIND_FLOAT, 32, 0)
    assert DtypeCodec.encode("float16") == (FORMAT_KIND_FLOAT, 16, 0)
    assert DtypeCodec.encode("bfloat16") == (FORMAT_KIND_FLOAT, 16, FLAGS_BFLOAT16)
    assert DtypeCodec.encode("uint8") == (FORMAT_KIND_UNSIGNED, 8, 0)
    assert DtypeCodec.encode("uint16") == (FORMAT_KIND_UNSIGNED, 16, 0)
    assert DtypeCodec.encode("int8") == (FORMAT_KIND_SIGNED, 8, 0)
    assert DtypeCodec.encode("int16") == (FORMAT_KIND_SIGNED, 16, 0)
    assert FLAGS_BFLOAT16 == 0x0001


def test_dtype_itemsize_mapping() -> None:
    """dtype strings map to correct byte sizes (via DtypeCodec, not exporter internals)."""
    from cuda_link.shm_protocol import DtypeCodec

    assert DtypeCodec.itemsize("float32") == 4
    assert DtypeCodec.itemsize("float16") == 2
    assert DtypeCodec.itemsize("bfloat16") == 2
    assert DtypeCodec.itemsize("uint8") == 1
    assert DtypeCodec.itemsize("uint16") == 2
    assert DtypeCodec.itemsize("int8") == 1
    assert DtypeCodec.itemsize("int16") == 2


def test_data_size_calculation_uint8() -> None:
    """data_size is correctly computed for uint8."""
    # 512 x 512 x 4 channels x 1 byte = 1,048,576 bytes
    exp = _make_bare("x", 512, 512, channels=4, dtype="uint8")
    assert exp.data_size == 512 * 512 * 4 * 1


def test_data_size_calculation_float32() -> None:
    """data_size is correctly computed for float32."""
    # 256 x 256 x 4 x 4 bytes = 1,048,576 bytes
    exp = _make_bare("x", 256, 256, channels=4, dtype="float32")
    assert exp.data_size == 256 * 256 * 4 * 4


def test_invalid_dtype_raises() -> None:
    """Unsupported dtype raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _make_bare("x", 64, 64, dtype="int32")


def test_num_slots_zero_raises() -> None:
    """num_slots=0 raises ValueError."""
    with pytest.raises(ValueError, match="num_slots"):
        _make_bare("x", 64, 64, num_slots=0)


def test_num_slots_over_max_raises() -> None:
    """num_slots=11 raises ValueError."""
    with pytest.raises(ValueError, match="num_slots"):
        _make_bare("x", 64, 64, num_slots=11)


def test_num_slots_max_valid() -> None:
    """num_slots=10 is accepted."""
    exp = _make_bare("x", 64, 64, num_slots=10)
    assert exp._spec.num_slots == 10


def test_double_close_no_crash() -> None:
    """Calling close() twice does not raise."""
    from cuda_link import ExportPolicy, FrameSpec
    from cuda_link._cuda_adapters import FakeCUDAAdapter
    from cuda_link.exporter import Exporter

    exp = Exporter.open(
        FrameSpec(shm_name=f"test_{uuid.uuid4().hex[:8]}", height=64, width=64),
        policy=ExportPolicy.for_testing(),
        cuda=FakeCUDAAdapter(),
    )
    exp.close()
    exp.close()  # Double-close must not raise


def test_export_sync_false_no_stream_sync_when_event_present() -> None:
    """export_sync=False (default): stream_synchronize NOT called when IPC event exists."""
    from unittest.mock import patch

    from cuda_link import ExportPolicy, FrameSpec, GpuFrame
    from cuda_link._cuda_adapters import FakeCUDAAdapter
    from cuda_link.exporter import Exporter

    fake_cuda = FakeCUDAAdapter()
    exp = Exporter.open(
        FrameSpec(shm_name=f"test_{uuid.uuid4().hex[:8]}", height=64, width=64),
        policy=ExportPolicy.for_testing(),
        cuda=fake_cuda,
    )
    try:
        with patch.object(fake_cuda, "stream_synchronize") as mock_sync:
            exp.export(GpuFrame(ptr=1000, size=exp.data_size))
        mock_sync.assert_not_called()
    finally:
        exp.close()


def test_export_sync_false_calls_stream_sync_when_no_event() -> None:
    """export_sync=False: stream_synchronize IS called when no IPC event (correctness backstop)."""
    from unittest.mock import patch

    from cuda_link import ExportPolicy, FrameSpec, GpuFrame
    from cuda_link._cuda_adapters import FakeCUDAAdapter
    from cuda_link.exporter import Exporter

    fake_cuda = FakeCUDAAdapter()
    with patch.object(fake_cuda, "create_ipc_event", return_value=None):
        exp = Exporter.open(
            FrameSpec(shm_name=f"test_{uuid.uuid4().hex[:8]}", height=64, width=64),
            policy=ExportPolicy.for_testing(),
            cuda=fake_cuda,
        )
    try:
        with patch.object(fake_cuda, "stream_synchronize") as mock_sync:
            exp.export(GpuFrame(ptr=1000, size=exp.data_size))
        mock_sync.assert_called_once()
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# CUDA integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_open_succeeds(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Exporter.open() returns a ready exporter."""
    from cuda_link import Exporter, FrameSpec

    shared_memory_cleanup.append(temp_shm_name)
    exp = Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64))
    try:
        assert exp.is_ready()
    finally:
        exp.close()


@pytest.mark.requires_cuda
def test_close_idempotent(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Calling close() twice does not raise (CUDA path)."""
    from cuda_link import Exporter, FrameSpec

    shared_memory_cleanup.append(temp_shm_name)
    exp = Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64))
    exp.close()
    exp.close()  # Second close must not raise


@pytest.mark.requires_cuda
def test_shm_protocol_magic(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """SharedMemory header contains correct protocol magic 0x43495044 ('CIPD')."""
    from cuda_link import Exporter, FrameSpec
    from cuda_link.shm_protocol import PROTOCOL_MAGIC

    shared_memory_cleanup.append(temp_shm_name)
    exp = Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64))
    try:
        shm = SharedMemory(name=temp_shm_name)
        try:
            magic = struct.unpack_from("<I", shm.buf, 0)[0]
            assert magic == PROTOCOL_MAGIC, f"Expected 0x{PROTOCOL_MAGIC:08x}, got 0x{magic:08x}"
        finally:
            shm.close()
    finally:
        exp.close()


@pytest.mark.requires_cuda
def test_shm_protocol_num_slots(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """SharedMemory header encodes the correct num_slots value."""
    from cuda_link import Exporter, FrameSpec

    NUM_SLOTS_OFFSET = 12

    shared_memory_cleanup.append(temp_shm_name)
    exp = Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64, num_slots=3))
    try:
        shm = SharedMemory(name=temp_shm_name)
        try:
            num_slots = struct.unpack_from("<I", shm.buf, NUM_SLOTS_OFFSET)[0]
            assert num_slots == 3
        finally:
            shm.close()
    finally:
        exp.close()


@pytest.mark.requires_cuda
def test_ring_buffer_write_idx_increments(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """write_idx in SharedMemory increments after each export() call."""
    from cuda_link import Exporter, FrameSpec, GpuFrame
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime

    WRITE_IDX_OFFSET = 16

    shared_memory_cleanup.append(temp_shm_name)
    exp = Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64, dtype="uint8"))
    cuda = get_cuda_runtime()

    try:
        shm = SharedMemory(name=temp_shm_name)
        test_size = 64 * 64 * 4
        gpu_buf = cuda.malloc(test_size)

        try:
            for expected_idx in range(1, 4):
                exp.export(GpuFrame(ptr=int(gpu_buf.value), size=test_size))
                write_idx = struct.unpack_from("<I", shm.buf, WRITE_IDX_OFFSET)[0]
                assert write_idx == expected_idx, f"Expected write_idx={expected_idx}, got {write_idx}"
        finally:
            cuda.free(gpu_buf)
            shm.close()
    finally:
        exp.close()


@pytest.mark.requires_cuda
def test_shutdown_flag_set_on_close(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """close() writes shutdown flag=1 to SharedMemory before unlinking."""
    from cuda_link import Exporter, FrameSpec
    from cuda_link.shm_protocol import SHMLayout

    shared_memory_cleanup.append(temp_shm_name)
    num_slots = 2
    exp = Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64, num_slots=num_slots))
    layout = SHMLayout(num_slots)

    shm_observer = SharedMemory(name=temp_shm_name)
    try:
        assert shm_observer.buf[layout.shutdown_offset] == 0

        exp.close()

        assert shm_observer.buf[layout.shutdown_offset] == 1
    finally:
        shm_observer.close()
        # Do NOT unlink — close() already did it; shared_memory_cleanup will suppress FileNotFoundError


@pytest.mark.requires_cuda
def test_context_manager(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Context manager cleans up correctly on exit."""
    from cuda_link import Exporter, FrameSpec

    shared_memory_cleanup.append(temp_shm_name)
    with Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64)) as exp:
        assert exp.is_ready()

    assert not exp.is_ready()


@pytest.mark.requires_cuda
def test_timestamp_uses_perf_counter(temp_shm_name: str, shared_memory_cleanup: list) -> None:
    """Producer timestamp in SharedMemory is from time.perf_counter() (monotonic clock).

    We verify this by checking the timestamp is within [t_before, t_after + epsilon]
    and that the value is consistent with perf_counter resolution (not epoch seconds).
    """
    from cuda_link import Exporter, FrameSpec, GpuFrame
    from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
    from cuda_link.shm_protocol import SHMLayout

    shared_memory_cleanup.append(temp_shm_name)
    num_slots = 2
    exp = Exporter.open(FrameSpec(shm_name=temp_shm_name, height=64, width=64, num_slots=num_slots))
    cuda = get_cuda_runtime()
    layout = SHMLayout(num_slots)

    try:
        shm = SharedMemory(name=temp_shm_name)
        test_size = 64 * 64 * 4
        gpu_buf = cuda.malloc(test_size)

        try:
            t_before = time.perf_counter()
            exp.export(GpuFrame(ptr=int(gpu_buf.value), size=test_size))
            t_after = time.perf_counter()

            ts_bytes = bytes(shm.buf[layout.timestamp_offset : layout.timestamp_offset + 8])
            ts = struct.unpack("<d", ts_bytes)[0]

            assert t_before <= ts <= t_after + 0.001, (
                f"Timestamp {ts:.6f} not in expected perf_counter range [{t_before:.6f}, {t_after + 0.001:.6f}]"
            )
        finally:
            cuda.free(gpu_buf)
            shm.close()
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# Improvement 2: SharedMemory write ordering (atomicity)
# — rewritten in v1.6 to use Exporter.open() + FakeCUDAAdapter instead of
#   object.__new__(CUDAIPCExporter) with 25 hand-wired private attributes.
# ---------------------------------------------------------------------------


def _make_write_order_exporter():
    """Open a real Exporter backed by FakeCUDAAdapter for write-ordering tests."""
    from cuda_link import ExportPolicy, FrameSpec
    from cuda_link._cuda_adapters import FakeCUDAAdapter
    from cuda_link.exporter import Exporter

    shm_name = f"test_shm_write_{uuid.uuid4().hex[:8]}"
    fake = FakeCUDAAdapter(device=0)
    policy = ExportPolicy.for_testing()
    return Exporter.open(
        FrameSpec(shm_name=shm_name, height=8, width=8, channels=4, dtype="uint8", num_slots=2, device=0),
        policy=policy,
        cuda=fake,
    )


def test_shm_write_correctness_after_export() -> None:
    """After export(), shutdown_flag=0 and write_idx is incremented by 1."""
    from cuda_link import FrameOutcome, GpuFrame

    exp = _make_write_order_exporter()
    try:
        buf = exp.shm_handle.buf
        buf[exp._shutdown_offset] = 1  # stale flag from a prior session
        initial_write_idx = exp.write_idx

        outcome = exp.export(GpuFrame(ptr=0, size=exp.data_size))

        assert outcome != FrameOutcome.FAILED
        assert buf[exp._shutdown_offset] == 0, "shutdown_flag must be 0 after export()"
        actual_write_idx = struct.unpack_from("<I", buf, 16)[0]
        assert actual_write_idx == initial_write_idx + 1, (
            f"write_idx must increment from {initial_write_idx} to {initial_write_idx + 1}, got {actual_write_idx}"
        )
    finally:
        exp.close()


def test_shm_write_ordering_shutdown_before_write_idx() -> None:
    """shutdown_flag is cleared BEFORE write_idx is published.

    Tests the invariant directly against publish_frame() in shm_protocol —
    the function that owns this ordering guarantee.
    """
    import struct as real_struct
    from unittest.mock import MagicMock, patch

    from cuda_link.shm_protocol import SHMLayout, publish_frame

    num_slots = 2
    layout = SHMLayout(num_slots)
    write_log: list[tuple] = []

    class _SpyBuf(bytearray):
        def __setitem__(self, key, val) -> None:
            if isinstance(key, int):
                write_log.append(("setitem", key, val))
            super().__setitem__(key, val)

    real_st_u32 = real_struct.Struct("<I")
    real_st_f64 = real_struct.Struct("<d")

    def spy_u32_pack_into(buf, offset: int, *args) -> None:
        write_log.append(("pack_into", offset))
        real_st_u32.pack_into(buf, offset, *args)

    def spy_f64_pack_into(buf, offset: int, *args) -> None:
        real_st_f64.pack_into(buf, offset, *args)

    mock_st_u32 = MagicMock()
    mock_st_u32.pack_into.side_effect = spy_u32_pack_into
    mock_st_f64 = MagicMock()
    mock_st_f64.pack_into.side_effect = spy_f64_pack_into

    spy_buf = _SpyBuf(layout.total_size)
    spy_buf[layout.shutdown_offset] = 1  # stale flag
    write_log.clear()

    with (
        patch("cuda_link.shm_protocol._ST_U32", mock_st_u32),
        patch("cuda_link.shm_protocol._ST_F64", mock_st_f64),
    ):
        publish_frame(spy_buf, layout=layout, write_idx=1, timestamp=0.0)

    WRITE_IDX_OFFSET = 16
    shutdown_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "setitem" and e[1] == layout.shutdown_offset),
        None,
    )
    write_idx_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "pack_into" and e[1] == WRITE_IDX_OFFSET),
        None,
    )

    assert shutdown_pos is not None, "shutdown_flag must be written by publish_frame()"
    assert write_idx_pos is not None, "write_idx must be published by publish_frame()"
    assert shutdown_pos < write_idx_pos, (
        f"shutdown_flag write (log[{shutdown_pos}]) must precede "
        f"write_idx publish (log[{write_idx_pos}]); full log: {write_log}"
    )


def test_release_fence_called_between_flag_and_write_idx() -> None:
    """C3: _release_fence() is called after shutdown_flag clear and before write_idx publish.

    Tests the invariant directly against publish_frame() in shm_protocol.
    """
    import struct as real_struct
    from unittest.mock import MagicMock, patch

    from cuda_link.shm_protocol import SHMLayout, publish_frame

    num_slots = 2
    layout = SHMLayout(num_slots)
    fence_calls: list[int] = []
    write_log: list[tuple] = []

    class _SpyBuf(bytearray):
        def __setitem__(self, key, val) -> None:
            if isinstance(key, int):
                write_log.append(("setitem", key, val))
            super().__setitem__(key, val)

    real_st_u32 = real_struct.Struct("<I")
    real_st_f64 = real_struct.Struct("<d")

    def spy_u32_pack_into(buf, offset: int, *args) -> None:
        write_log.append(("pack_into", offset))
        real_st_u32.pack_into(buf, offset, *args)

    def spy_f64_pack_into(buf, offset: int, *args) -> None:
        real_st_f64.pack_into(buf, offset, *args)

    def spy_fence() -> None:
        fence_calls.append(len(write_log))

    mock_st_u32 = MagicMock()
    mock_st_u32.pack_into.side_effect = spy_u32_pack_into
    mock_st_f64 = MagicMock()
    mock_st_f64.pack_into.side_effect = spy_f64_pack_into

    spy_buf = _SpyBuf(layout.total_size)
    spy_buf[layout.shutdown_offset] = 1
    write_log.clear()

    with (
        patch("cuda_link.shm_protocol._ST_U32", mock_st_u32),
        patch("cuda_link.shm_protocol._ST_F64", mock_st_f64),
        patch("cuda_link.shm_protocol._release_fence", spy_fence),
    ):
        publish_frame(spy_buf, layout=layout, write_idx=1, timestamp=0.0)

    assert len(fence_calls) == 1, "exactly one fence call per publish_frame()"

    WRITE_IDX_OFFSET = 16
    shutdown_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "setitem" and e[1] == layout.shutdown_offset),
        None,
    )
    write_idx_pos = next(
        (i for i, e in enumerate(write_log) if e[0] == "pack_into" and e[1] == WRITE_IDX_OFFSET),
        None,
    )
    fence_pos = fence_calls[0]

    assert shutdown_pos is not None
    assert write_idx_pos is not None
    assert shutdown_pos < fence_pos, "fence must come AFTER shutdown_flag write"
    assert fence_pos <= write_idx_pos, "fence must come BEFORE write_idx publish"
