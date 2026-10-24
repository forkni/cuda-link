"""
TDReceiverEngine.initialize_receiver(): guard, failure and commit paths.

The segment is an in-memory stand-in for the producer's SharedMemory (one bytearray
built with the real SHMLayout / Metadata packers), and CUDA is a recording fake.
Header rejection and the stream-create leak are pinned in
test_tdreceiver_header_validation.py / test_tdreceiver_shm_leak.py; this file covers
the rest: metadata guards, the shutdown flag, per-slot IPC opens with partial cleanup,
stream reuse, the commit (connection, format, shape, float16 staging) and the
post-commit release.
"""

from __future__ import annotations

import ctypes
import struct
from multiprocessing.shared_memory import SharedMemory

import pytest

from cuda_link.shm_protocol import (
    FLAGS_MONO_ALPHA,
    FORMAT_KIND_FLOAT,
    FORMAT_KIND_UNSIGNED,
    METADATA_SIZE,
    SHM_HEADER_SIZE,
    SLOT_SIZE,
    Metadata,
    SHMLayout,
)

from .test_tdreceiver_shm_leak import _make_receiver


class _FakeShm:
    """In-memory SharedMemory: just `.buf` and a recorded `.close()`."""

    def __init__(self, data: bytearray) -> None:
        self.buf = memoryview(data)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeCuda:
    """Recording CUDA adapter. `fail_mem_slot` / `fail_event_slot` raise on that open call."""

    def __init__(self, *, fail_mem_slot: int | None = None, fail_event_slot: int | None = None) -> None:
        self.fail_mem_slot = fail_mem_slot
        self.fail_event_slot = fail_event_slot
        self.streams_created: list[int] = []
        self.mem_opened: list[bytes] = []
        self.mem_closed: list[int] = []
        self.event_opened: list[bytes] = []
        self.events_destroyed: list[object] = []
        self.streams_destroyed: list[object] = []
        self.host_registered: list[tuple[int, int]] = []
        self._next_ptr = 0x10000

    def get_device(self) -> int:
        return 0

    def create_stream_with_priority(self, flags: int) -> ctypes.c_void_p:
        self.streams_created.append(flags)
        return ctypes.c_void_p(0x5000)

    def ipc_open_mem_handle(self, handle, flags: int = 1) -> ctypes.c_void_p:
        slot = len(self.mem_opened)
        self.mem_opened.append(bytes(handle))
        assert flags == 1, "cudaIpcMemLazyEnablePeerAccess"
        if slot == self.fail_mem_slot:
            raise RuntimeError("injected ipc_open_mem_handle failure")
        self._next_ptr += 0x1000
        return ctypes.c_void_p(self._next_ptr)

    def ipc_close_mem_handle(self, dev_ptr: ctypes.c_void_p) -> None:
        self.mem_closed.append(dev_ptr.value)

    def ipc_open_event_handle(self, handle) -> object:
        slot = len(self.event_opened)
        self.event_opened.append(bytes(handle))
        if slot == self.fail_event_slot:
            raise RuntimeError("injected ipc_open_event_handle failure")
        return object()

    def destroy_event(self, event: object) -> None:
        self.events_destroyed.append(event)

    def destroy_stream(self, stream: object) -> None:
        self.streams_destroyed.append(stream)

    def malloc_host(self, size: int):
        raise RuntimeError("no pinned memory in tests")  # exercise the pageable path

    def free_host(self, ptr) -> None:
        pass

    def host_register(self, ptr: int, size: int, flags: int = 0) -> None:
        self.host_registered.append((ptr, size))

    def host_unregister(self, ptr: int) -> None:
        pass


def _segment(
    num_slots: int = 1,
    *,
    width: int = 64,
    height: int = 64,
    comps: int = 4,
    kind: int = FORMAT_KIND_FLOAT,
    bits: int = 32,
    flags: int = 0,
    data_size: int | None = None,
    shutdown: bool = False,
    mem_tags: list[int] | None = None,
    event_tags: list[int] | None = None,
    version: int = 7,
) -> bytearray:
    """A well-formed producer segment; tag 0 leaves that slot's handle all zeros."""
    layout = SHMLayout(num_slots=num_slots)
    buf = layout.build_buffer(version=version, write_idx=0)
    if data_size is None:
        data_size = width * height * comps * bits // 8
    Metadata(
        width=width,
        height=height,
        num_comps=comps,
        format_kind=kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=data_size,
    ).pack_into(memoryview(buf), layout)
    if shutdown:
        buf[layout.shutdown_offset] = 1
    for slot in range(num_slots):
        base = SHM_HEADER_SIZE + slot * SLOT_SIZE
        mem_tag = mem_tags[slot] if mem_tags is not None else slot + 1
        buf[base : base + 64] = bytes([mem_tag]) * 64
        event_tag = event_tags[slot] if event_tags is not None else 0
        buf[base + 64 : base + 128] = bytes([event_tag]) * 64
    return buf


class _Rig:
    def __init__(self, engine, cuda: _FakeCuda, shm: _FakeShm, logs: list[str], params: list) -> None:
        self.engine = engine
        self.cuda = cuda
        self.shm = shm
        self.logs = logs
        self.params = params


def _rig(monkeypatch: pytest.MonkeyPatch, data: bytearray, cuda: _FakeCuda | None = None) -> _Rig:
    import TDReceiver as TDReceiverMod

    cuda = cuda or _FakeCuda()
    shm = _FakeShm(data)
    monkeypatch.setattr(TDReceiverMod, "get_cuda_runtime", lambda device: cuda)
    monkeypatch.setattr(TDReceiverMod, "SharedMemory", lambda name: shm)

    engine = _make_receiver("unused-in-memory")
    logs: list[str] = []
    params: list = []
    engine._log = lambda msg, *a, **k: logs.append(msg)
    engine._host.set_param_value = lambda name, value: params.append(("value", name, value))
    engine._host.set_param_enabled = lambda name, enabled: params.append(("enabled", name, enabled))
    return _Rig(engine, cuda, shm, logs, params)


def _assert_rejected_cleanly(rig: _Rig) -> None:
    assert rig.shm.closed is True
    assert rig.engine._initialized is False
    assert rig.engine._connection.shm_handle is None
    assert rig.engine._cached_shape is None


def test_initialize_when_already_initialized_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment())
    rig.engine._initialized = True

    assert rig.engine.initialize_receiver() is True
    assert rig.params == []
    assert rig.cuda.streams_created == []


def test_initialize_missing_segment_reports_sender_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    import TDReceiver as TDReceiverMod

    monkeypatch.setattr(TDReceiverMod, "get_cuda_runtime", lambda device: _FakeCuda())

    def _absent(name: str):
        raise FileNotFoundError(name)

    monkeypatch.setattr(TDReceiverMod, "SharedMemory", _absent)
    engine = _make_receiver("cuda_link_test_absent_segment")
    logs: list[str] = []
    engine._log = lambda msg, *a, **k: logs.append(msg)

    assert engine.initialize_receiver() is False
    assert any("cuda_link_test_absent_segment" in m and "not found" in m for m in logs)
    assert engine._connection.shm_handle is None


def test_initialize_disables_numslots_and_mirrors_sender_slot_count(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment(num_slots=3))

    assert rig.engine.initialize_receiver() is True

    assert ("enabled", "Numslots", False) in rig.params
    assert ("value", "Numslots", 3) in rig.params


@pytest.mark.parametrize(("num_slots", "accepted"), [(10, True), (11, False)], ids=["max-slots", "above-max"])
def test_initialize_slot_count_upper_bound_is_ten(
    monkeypatch: pytest.MonkeyPatch, num_slots: int, accepted: bool
) -> None:
    rig = _rig(monkeypatch, _segment(num_slots=num_slots))

    assert rig.engine.initialize_receiver() is accepted

    assert rig.engine._initialized is accepted
    assert rig.cuda.streams_created == ([0x01] if accepted else [])


def test_initialize_unreadable_magic_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, bytearray(2))

    assert rig.engine.initialize_receiver() is False

    assert any("Cannot read protocol magic" in m for m in rig.logs)
    _assert_rejected_cleanly(rig)


def test_initialize_segment_without_metadata_region_is_rejected_as_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = SHMLayout(num_slots=1)
    data = layout.build_buffer(version=7)[: layout.metadata_offset + METADATA_SIZE - 1]
    rig = _rig(monkeypatch, data)

    assert rig.engine.initialize_receiver() is False

    assert any("legacy sender" in m for m in rig.logs)
    assert rig.cuda.streams_created == []
    _assert_rejected_cleanly(rig)


def test_initialize_metadata_decode_error_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import TDReceiver as TDReceiverMod

    class _Undecodable:
        @staticmethod
        def read_from(buf, layout):
            raise struct.error("injected metadata decode failure")

    rig = _rig(monkeypatch, _segment())
    monkeypatch.setattr(TDReceiverMod, "Metadata", _Undecodable)

    assert rig.engine.initialize_receiver() is False

    assert any("Failed to read SHM metadata" in m and "injected metadata decode failure" in m for m in rig.logs)
    _assert_rejected_cleanly(rig)


@pytest.mark.parametrize(
    ("overrides", "expected_bytes"),
    [
        pytest.param({"data_size": 64 * 64 * 4 * 4 - 1}, 64 * 64 * 4 * 4, id="data-size-off-by-one"),
        pytest.param({"data_size": 64 * 64 * 4 * 4 + 1}, 64 * 64 * 4 * 4, id="data-size-too-large"),
        pytest.param({"bits": 0, "data_size": 0}, 0, id="zero-bits"),
    ],
)
def test_initialize_size_invariant_violation_is_rejected(
    monkeypatch: pytest.MonkeyPatch, overrides: dict, expected_bytes: int
) -> None:
    rig = _rig(monkeypatch, _segment(**overrides))

    assert rig.engine.initialize_receiver() is False

    assert any("size invariant failed" in m and f"={expected_bytes} " in m for m in rig.logs)
    assert rig.cuda.streams_created == []
    _assert_rejected_cleanly(rig)


@pytest.mark.parametrize("zeroed", ["width", "height"], ids=["width", "height"])
def test_initialize_zero_geometry_means_no_frame_written_yet(monkeypatch: pytest.MonkeyPatch, zeroed: str) -> None:
    rig = _rig(monkeypatch, _segment(**{zeroed: 0}))

    assert rig.engine.initialize_receiver() is False

    assert any("sender may not have written frame yet" in m for m in rig.logs)
    assert rig.cuda.streams_created == []
    _assert_rejected_cleanly(rig)


def test_initialize_shutdown_flag_rejects_before_any_cuda_work(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment(shutdown=True))

    assert rig.engine.initialize_receiver() is False

    assert any("Shutdown flag is set" in m for m in rig.logs)
    assert rig.cuda.streams_created == []
    assert rig.cuda.mem_opened == []
    _assert_rejected_cleanly(rig)


def test_initialize_zero_mem_handle_destroys_fresh_stream_and_closes_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment(num_slots=1, mem_tags=[0]))

    assert rig.engine.initialize_receiver() is False

    assert any("Slot 0: IPC mem handle is all zeros" in m for m in rig.logs)
    assert rig.cuda.mem_opened == []
    assert [s.value for s in rig.cuda.streams_destroyed] == [0x5000]
    _assert_rejected_cleanly(rig)


def test_initialize_zero_handle_on_later_slot_releases_earlier_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment(num_slots=3, mem_tags=[1, 2, 0], event_tags=[9, 9, 9]))

    assert rig.engine.initialize_receiver() is False

    assert len(rig.cuda.mem_opened) == 2
    assert sorted(rig.cuda.mem_closed) == [0x11000, 0x12000]
    assert len(rig.cuda.events_destroyed) == 2
    _assert_rejected_cleanly(rig)


def test_initialize_mem_handle_open_failure_releases_earlier_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment(num_slots=3), _FakeCuda(fail_mem_slot=1))

    assert rig.engine.initialize_receiver() is False

    assert any("Slot 1: cudaIpcOpenMemHandle failed" in m and "injected" in m for m in rig.logs)
    assert rig.cuda.mem_closed == [0x11000]
    assert [s.value for s in rig.cuda.streams_destroyed] == [0x5000]
    _assert_rejected_cleanly(rig)


def test_initialize_event_open_failure_fails_init_and_releases_that_slots_mem_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _rig(
        monkeypatch,
        _segment(num_slots=2, event_tags=[5, 6]),
        _FakeCuda(fail_event_slot=1),
    )

    assert rig.engine.initialize_receiver() is False

    assert any("Slot 1: failed to open IPC event" in m and "injected" in m for m in rig.logs)
    # Slot 0 fully opened, slot 1's mem handle opened before its event failed: both released.
    assert sorted(rig.cuda.mem_closed) == [0x11000, 0x12000]
    assert len(rig.cuda.events_destroyed) == 1
    _assert_rejected_cleanly(rig)


def test_initialize_reuses_existing_stream_and_keeps_it_alive_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment(num_slots=1, mem_tags=[0]))
    reused = ctypes.c_void_p(0x7777)
    rig.engine._connection.stream = reused

    assert rig.engine.initialize_receiver() is False

    assert rig.cuda.streams_created == []
    assert rig.cuda.streams_destroyed == []
    assert any("Reusing receiver stream: 0x0000000000007777" in m for m in rig.logs)


def test_initialize_reuses_existing_stream_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment())
    reused = ctypes.c_void_p(0x7777)
    rig.engine._connection.stream = reused

    assert rig.engine.initialize_receiver() is True

    assert rig.cuda.streams_created == []
    assert rig.engine._connection.stream is reused


def test_initialize_commits_connection_for_every_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(
        monkeypatch,
        _segment(num_slots=3, mem_tags=[1, 2, 3], event_tags=[0xA, 0, 0xC], version=42),
    )
    rig.engine._diag_frames_since_reinit = 99
    rig.engine._last_fmt_status = "stale"

    assert rig.engine.initialize_receiver() is True

    conn = rig.engine._connection
    layout = SHMLayout(num_slots=3)
    assert rig.cuda.streams_created == [0x01]
    assert conn.shm_handle is rig.shm and rig.shm.closed is False
    assert conn.num_slots == 3
    assert conn.ipc_version == 42
    assert conn.shutdown_offset == layout.shutdown_offset
    assert conn.layout.total_size == layout.total_size
    assert conn.last_write_idx == 0
    assert conn.stream.value == 0x5000
    assert [p.value for p in conn.dev_ptrs] == [0x11000, 0x12000, 0x13000]
    assert [bytes(h) for h in conn.ipc_handles] == [bytes([t]) * 64 for t in (1, 2, 3)]
    assert rig.cuda.mem_opened == [bytes([t]) * 64 for t in (1, 2, 3)]
    assert [e is not None for e in conn.ipc_events] == [True, False, True]
    assert rig.cuda.event_opened == [bytes([0xA]) * 64, bytes([0xC]) * 64]
    assert rig.engine._initialized is True
    assert rig.engine._diag_frames_since_reinit == 0
    assert rig.engine._last_fmt_status == ""
    assert rig.engine._retry.needs_resolution_update is True
    assert any("Receiver initialized: 3 slots, 64x64x4" in m for m in rig.logs)


@pytest.mark.parametrize(
    ("fmt", "needs_format_update", "shape_dtype"),
    [
        pytest.param(
            {"kind": FORMAT_KIND_FLOAT, "bits": 32, "comps": 4},
            False,
            "float32",
            id="rgba32float-is-script-top-default",
        ),
        pytest.param(
            {"kind": FORMAT_KIND_FLOAT, "bits": 32, "comps": 2, "flags": FLAGS_MONO_ALPHA},
            True,
            "float32",
            id="monoalpha32float",
        ),
        pytest.param({"kind": FORMAT_KIND_UNSIGNED, "bits": 8, "comps": 4}, True, "uint8", id="rgba8fixed"),
        pytest.param({"kind": FORMAT_KIND_UNSIGNED, "bits": 16, "comps": 1}, True, "uint16", id="r16fixed"),
    ],
)
def test_initialize_resolves_format_and_shape_dtype_from_metadata(
    monkeypatch: pytest.MonkeyPatch, fmt: dict, needs_format_update: bool, shape_dtype: str
) -> None:
    rig = _rig(monkeypatch, _segment(width=32, height=16, **fmt))

    assert rig.engine.initialize_receiver() is True

    shape = rig.engine._cached_shape
    assert (shape.width, shape.height, shape.numComps) == (32, 16, fmt["comps"])
    assert shape.dataType.__name__ == shape_dtype
    assert rig.engine._format.bits == fmt["bits"]
    assert rig.engine._format.num_comps == fmt["comps"]
    assert rig.engine._retry.needs_format_update is needs_format_update


def test_initialize_float16_presents_float32_shape_and_stages_conversion_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import TDReceiver as TDReceiverMod

    # CuPy is optional: pin the no-CuPy branch so the test does not depend on the host install.
    monkeypatch.setattr(TDReceiverMod, "cp", object())
    monkeypatch.setattr(TDReceiverMod, "CUPY_AVAILABLE", False)
    rig = _rig(monkeypatch, _segment(width=8, height=4, comps=4, kind=FORMAT_KIND_FLOAT, bits=16))

    assert rig.engine.initialize_receiver() is True

    assert rig.engine._cached_shape.dataType.__name__ == "float32"
    assert rig.engine._format.dtype_str == "float16"
    assert rig.engine._f16_cpu_buf.dtype.name == "float16"
    assert rig.engine._f16_cpu_buf.size == 8 * 4 * 4
    assert rig.engine._f32_cpu_buf.shape == (4, 8, 4)
    assert rig.engine._f32_cpu_buf.dtype.name == "float32"
    # Pinned alloc was refused, so the pageable fallback is page-locked.
    assert rig.cuda.host_registered == [(rig.engine._f16_cpu_buf.ctypes.data, 8 * 4 * 4 * 2)]


def test_initialize_cuda_runtime_failure_returns_false_without_touching_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import TDReceiver as TDReceiverMod

    def _boom(device):
        raise RuntimeError("no cudart")

    monkeypatch.setattr(TDReceiverMod, "get_cuda_runtime", _boom)
    engine = _make_receiver("cuda_link_test_unused")
    logs: list[str] = []
    engine._log = lambda msg, *a, **k: logs.append(msg)

    assert engine.initialize_receiver() is False

    assert any("Receiver initialization failed: no cudart" in m for m in logs)
    assert engine._initialized is False


def test_initialize_failure_after_commit_releases_connection_but_keeps_backoff_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _rig(monkeypatch, _segment(num_slots=2))
    rig.engine._retry.connect_attempts = 3

    def _fail(*args, **kwargs):
        raise RuntimeError("shape factory failed")

    rig.engine._host.make_cuda_shape = _fail

    assert rig.engine.initialize_receiver() is False

    assert any("Receiver initialization failed: shape factory failed" in m for m in rig.logs)
    assert sorted(rig.cuda.mem_closed) == [0x11000, 0x12000]
    assert [s.value for s in rig.cuda.streams_destroyed] == [0x5000]
    assert rig.shm.closed is True
    assert rig.engine._connection.shm_handle is None
    assert rig.engine._initialized is False
    assert rig.engine._cached_shape is None
    assert rig.engine._retry.connect_attempts == 3


def test_initialize_failure_before_commit_closes_the_local_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _rig(monkeypatch, _segment())

    def _boom(flags: int):
        raise RuntimeError("stream create failed")

    rig.cuda.create_stream_with_priority = _boom

    assert rig.engine.initialize_receiver() is False

    assert rig.shm.closed is True
    assert rig.engine._connection.shm_handle is None


def test_initialize_against_real_shared_memory_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """One end-to-end pass through a genuine OS segment (the in-memory fake is not the product)."""
    import uuid

    import TDReceiver as TDReceiverMod

    data = _segment(num_slots=2, width=16, height=16, event_tags=[3, 4])
    shm = SharedMemory(create=True, name=f"test_init_{uuid.uuid4().hex[:8]}", size=len(data))
    try:
        shm.buf[: len(data)] = data
        cuda = _FakeCuda()
        monkeypatch.setattr(TDReceiverMod, "get_cuda_runtime", lambda device: cuda)
        engine = _make_receiver(shm.name)
        engine._log = lambda *a, **k: None

        assert engine.initialize_receiver() is True
        assert engine._connection.num_slots == 2
        assert [bytes(h) for h in engine._connection.ipc_handles] == [bytes([1]) * 64, bytes([2]) * 64]
    finally:
        engine._release_resources()
        shm.close()
        shm.unlink()
