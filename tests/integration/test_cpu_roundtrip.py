"""
CPU-only IPC round-trip tests: Exporter → SHM → acquire_slot → Importer.get_frame_numpy().

All tests are unmarked and run in the no-GPU CI suite.  A real Importer.open() connects
to a real exporter-written SharedMemory; FakeCUDAAdapter stands in for every CUDA call.
TORCH_AVAILABLE and CUPY_AVAILABLE are monkeypatched False so the importer's eager torch
buffer build (importer.py:1021 → TorchBuffers.build, device="cuda") never fires.
NUMPY_AVAILABLE is left True — the numpy path is what's under test.

Verified feasibility: Exporter.open + Importer.open round-trip was confirmed against
current source (2026-06-10).  Key invariants noted in individual tests below.
"""

from __future__ import annotations

import dataclasses
import time
from ctypes import c_void_p
from multiprocessing.shared_memory import SharedMemory

import pytest

import cuda_link.importer as _imp_mod
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link._exporter_port import ExportPolicy, FrameSpec, GpuFrame
from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
from cuda_link.exporter import Exporter
from cuda_link.importer import Importer
from cuda_link.shm_protocol import (
    PROTOCOL_MAGIC,
    DtypeCodec,
    Metadata,
    SHMLayout,
    SlotState,
    acquire_slot,
    read_magic,
    read_num_slots,
    read_version,
    read_write_idx,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class _RecordingFake(FakeCUDAAdapter):
    """FakeCUDAAdapter that records memcpy_async calls.

    Appends (dst, src, count, kind) to self.memcpy_calls on each call, then
    delegates to the base no-op implementation.  Must not be used with graph
    mode (for_testing() already disables graphs).
    """

    def __init__(self, device: int = 0) -> None:
        super().__init__(device)
        self.memcpy_calls: list[tuple[c_void_p, c_void_p, int, int]] = []

    def memcpy_async(
        self,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
        stream: object,
    ) -> None:
        self.memcpy_calls.append((dst, src, count, kind))
        super().memcpy_async(dst, src, count, kind, stream)


def _open_exporter(
    shm_name: str,
    num_slots: int = 3,
    h: int = 8,
    w: int = 8,
    dtype: str = "uint8",
    fake: FakeCUDAAdapter | None = None,
) -> tuple[Exporter, FakeCUDAAdapter]:
    """Open a real Exporter against a FakeCUDAAdapter. Returns (exporter, fake).

    GpuFrame exports use ptr=0x5000 + producer_stream=0 to arm ordering and
    suppress the unordered-export warning (exporter.py:571 guards on
    _source_sync_recorded, which record_source_sync sets via producer_stream).
    """
    if fake is None:
        fake = FakeCUDAAdapter()
    spec = FrameSpec(shm_name=shm_name, height=h, width=w, channels=4, dtype=dtype, num_slots=num_slots)
    exp = Exporter.open(spec, policy=ExportPolicy.for_testing(), cuda=fake)
    return exp, fake


def _open_importer(
    shm_name: str,
    *,
    reconnect: bool = False,
    fake: FakeCUDAAdapter | None = None,
) -> tuple[Importer, FakeCUDAAdapter]:
    """Open a real Importer against a FakeCUDAAdapter. Returns (importer, fake).

    reconnect=True builds a policy with reconnect_enabled=True (default is False
    in ImportPolicy.for_testing() to avoid retry delays in unit tests).
    reconnect_enabled is an ImportPolicy field, not an Importer kwarg (C1).
    """
    if fake is None:
        fake = FakeCUDAAdapter()
    base_policy = ImportPolicy.for_testing()
    policy = dataclasses.replace(base_policy, reconnect_enabled=True) if reconnect else base_policy
    spec = ImportSpec(shm_name=shm_name)
    imp = Importer.open(spec, policy=policy, cuda=fake)
    return imp, fake


def _export_frame(exp: Exporter) -> None:
    """Export one fake frame, arming ordering to suppress the unordered warning."""
    exp.export(GpuFrame(ptr=0x5000, size=exp.data_size, producer_stream=0))


# ---------------------------------------------------------------------------
# Autouse fixture: disable torch/cupy GPU buffer builds globally per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_gpu_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch importer module-level flags so the eager device='cuda' buffer
    builds (importer.py:1021 → TorchBuffers.build) never fire on no-GPU hosts."""
    monkeypatch.setattr(_imp_mod, "TORCH_AVAILABLE", False)
    monkeypatch.setattr(_imp_mod, "CUPY_AVAILABLE", False)


# ---------------------------------------------------------------------------
# Section A — Wire-format round-trip
# ---------------------------------------------------------------------------


def test_wire_header_after_open(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """After Exporter.open(), a raw SHM observer sees a valid protocol header."""
    shared_memory_cleanup.append(temp_shm_name)
    exp, _ = _open_exporter(temp_shm_name)
    try:
        layout = SHMLayout(exp._spec.num_slots)
        obs = SharedMemory(name=temp_shm_name)
        try:
            buf = obs.buf
            assert read_magic(buf) == PROTOCOL_MAGIC
            assert read_version(buf) == 1
            assert read_num_slots(buf) == exp._spec.num_slots
            assert read_write_idx(buf) == 0
            assert buf[layout.shutdown_offset] == 0  # shutdown byte — no read_shutdown helper; index directly
            assert obs.size >= layout.total_size
        finally:
            obs.close()
    finally:
        exp.close()


def test_wire_metadata_round_trip(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Metadata.read_from() fields match the FrameSpec and DtypeCodec encoding."""
    shared_memory_cleanup.append(temp_shm_name)
    dtype = "uint8"
    h, w, ch = 8, 8, 4
    exp, _ = _open_exporter(temp_shm_name, h=h, w=w, dtype=dtype)
    try:
        layout = SHMLayout(exp._spec.num_slots)
        obs = SharedMemory(name=temp_shm_name)
        try:
            meta = Metadata.read_from(obs.buf, layout)
            assert meta.width == w
            assert meta.height == h
            assert meta.num_comps == ch
            assert meta.data_size == exp.data_size
            expected_kind, expected_bits, expected_flags = DtypeCodec.encode(dtype)
            assert meta.format_kind == expected_kind
            assert meta.bits_per_comp == expected_bits
            assert meta.flags == expected_flags
        finally:
            obs.close()
    finally:
        exp.close()


def test_acquire_slot_sees_published_frame(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """After one export(), a raw acquire_slot call returns NEW_FRAME."""
    shared_memory_cleanup.append(temp_shm_name)
    exp, _ = _open_exporter(temp_shm_name)
    try:
        layout = SHMLayout(exp._spec.num_slots)
        obs = SharedMemory(name=temp_shm_name)
        try:
            t_before = time.perf_counter()
            _export_frame(exp)
            t_after = time.perf_counter()

            result = acquire_slot(obs.buf, layout, last_write_idx=0, last_version=1)

            assert result.state is SlotState.NEW_FRAME
            assert result.slot == 0  # first frame → write_idx=1 → slot = (1-1) % num_slots = 0
            assert read_write_idx(obs.buf) == 1
            # timestamp is a producer wall-clock perf_counter; verify it's within bracket
            assert t_before <= result.timestamp <= t_after + 0.1
        finally:
            obs.close()
    finally:
        exp.close()


def test_ipc_handle_bytes_cross_process_offsets(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Exporter writes per-slot distinctive IPC handle bytes; importer picks them up.

    Patches ipc_get_mem_handle on the exporter fake to return handles with
    known per-slot patterns.  Verifies (a) the bytes appear at the correct SHM
    offset, and (b) the importer opened exactly num_slots handles.
    """
    shared_memory_cleanup.append(temp_shm_name)
    num_slots = 3
    exp_fake = FakeCUDAAdapter()

    # Build per-slot pattern objects before opening; counter tracks call order.
    call_count = [0]
    patterns: dict[int, bytes] = {}  # slot_index → 64-byte pattern

    class _PatternHandle:
        """Minimal IPC handle shape — both .internal and .reserved required by exporter."""

        def __init__(self, slot: int) -> None:
            self.internal = bytes([slot + 1] * 64)  # slot 0 → 0x01*64, slot 1 → 0x02*64, …
            self.reserved = bytes(64)

    def _patched_ipc_get_mem_handle(dev_ptr: c_void_p) -> _PatternHandle:
        s = call_count[0]
        call_count[0] += 1
        h = _PatternHandle(s)
        patterns[s] = h.internal
        return h

    exp_fake.ipc_get_mem_handle = _patched_ipc_get_mem_handle  # instance-level override

    exp, _ = _open_exporter(temp_shm_name, num_slots=num_slots, fake=exp_fake)
    try:
        layout = SHMLayout(num_slots)
        obs = SharedMemory(name=temp_shm_name)
        try:
            # (a) SHM bytes at each slot offset match the injected patterns.
            for slot in range(num_slots):
                off = layout.mem_handle_offset(slot)
                actual = bytes(obs.buf[off : off + 64])
                assert actual == patterns[slot], f"slot {slot}: SHM bytes mismatch"

            # (b) Open an importer and verify it opened exactly num_slots mem handles.
            imp, imp_fake = _open_importer(temp_shm_name)
            try:
                assert len(imp_fake.opened_mem_handles) == num_slots
                # Each cudaIpcMemHandle_t round-trips: bytes(struct) == original pattern.
                opened_patterns = {bytes(h) for h in imp_fake.opened_mem_handles.values()}
                expected_patterns = set(patterns.values())
                assert opened_patterns == expected_patterns
            finally:
                imp.close()
        finally:
            obs.close()
    finally:
        exp.close()


def test_real_importer_full_path_no_gpu(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Full Importer.get_frame_numpy() path without a GPU.

    ImportSpec(shm_name=...) only — shape/dtype auto-detected from SHM metadata.
    First call after one export → NEW_FRAME with correct shape/dtype.
    Second call without new export → NO_FRAME.
    """
    shared_memory_cleanup.append(temp_shm_name)
    h, w, ch, dtype = 8, 8, 4, "uint8"
    exp, _ = _open_exporter(temp_shm_name, h=h, w=w, dtype=dtype)
    try:
        _export_frame(exp)

        imp, _ = _open_importer(temp_shm_name)
        try:
            assert imp.is_ready()
            # Auto-detected geometry must match the exporter's FrameSpec.
            assert imp._format.shape == (h, w, ch)
            assert imp._format.dtype_str == dtype

            r1 = imp.get_frame_numpy()
            assert r1.outcome is ImportOutcome.NEW_FRAME
            assert r1.frame is not None
            assert r1.frame.shape == (h, w, ch)
            assert str(r1.frame.dtype) == dtype

            # No new export → no new frame.
            r2 = imp.get_frame_numpy()
            assert r2.outcome is ImportOutcome.NO_FRAME
        finally:
            imp.close()
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# Section B — Producer restart / version change
# ---------------------------------------------------------------------------


def test_version_changed_when_producer_reopens_live_segment(
    temp_shm_name: str, shared_memory_cleanup: list[str]
) -> None:
    """Opening a second Exporter on the same SHM name while the first is open bumps
    the version; a raw acquire_slot call returns VERSION_CHANGED.

    _initialize() attaches to the existing segment (exporter.py:309-314);
    _write_handles_to_shm() calls bump_version (exporter.py:349).
    """
    shared_memory_cleanup.append(temp_shm_name)
    exp_a, _ = _open_exporter(temp_shm_name)
    try:
        layout = SHMLayout(exp_a._spec.num_slots)
        obs = SharedMemory(name=temp_shm_name)
        try:
            assert read_version(obs.buf) == 1

            # Open a second exporter on the same SHM segment while A is still open.
            exp_b, _ = _open_exporter(temp_shm_name)
            try:
                assert read_version(obs.buf) == 2

                # A consumer that last saw version=1 sees VERSION_CHANGED.
                result = acquire_slot(obs.buf, layout, last_write_idx=0, last_version=1)
                assert result.state is SlotState.VERSION_CHANGED
                assert result.new_version == 2
            finally:
                exp_b.close()
        finally:
            obs.close()
    finally:
        exp_a.close()


def test_importer_reconnecting_on_version_change(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """When the producer re-initializes (version bump), the importer returns RECONNECTING
    and then _reinitialize() picks up the new version.

    Pins the retained-_last_write_idx behavior:
    - _reinitialize does NOT reset _last_write_idx (importer.py:1420-1456).
    - After version change and one reconnect, first export from the new producer
      yields NO_FRAME (write_idx=1 == _last_write_idx=1), second yields NEW_FRAME.
    This is a characterization of current behavior; it may be a product bug.
    """
    shared_memory_cleanup.append(temp_shm_name)
    exp_a, _ = _open_exporter(temp_shm_name)
    try:
        _export_frame(exp_a)  # write_idx → 1

        imp, _ = _open_importer(temp_shm_name)
        try:
            r1 = imp.get_frame_numpy()
            assert r1.outcome is ImportOutcome.NEW_FRAME
            assert imp._last_write_idx == 1  # consumed frame #1

            # Open exporter B on the same SHM → version bumps to 2, write_idx reset to 0.
            exp_b, _ = _open_exporter(temp_shm_name)
            try:
                # Next call sees VERSION_CHANGED and triggers _reinitialize → RECONNECTING.
                r2 = imp.get_frame_numpy()
                assert r2.outcome is ImportOutcome.RECONNECTING
                assert imp._conn.ipc_version == 2

                # _reinitialize does NOT reset _last_write_idx (characterization test).
                assert imp._last_write_idx == 1

                # First B export: write_idx=1 == _last_write_idx=1 → NO_FRAME.
                _export_frame(exp_b)
                r3 = imp.get_frame_numpy()
                assert r3.outcome is ImportOutcome.NO_FRAME  # characterization of retained _last_write_idx

                # Second B export: write_idx=2 != _last_write_idx=1 → NEW_FRAME.
                _export_frame(exp_b)
                r4 = imp.get_frame_numpy()
                assert r4.outcome is ImportOutcome.NEW_FRAME
            finally:
                exp_b.close()
        finally:
            imp.close()
    finally:
        exp_a.close()


def test_pipelined_drains_and_reprimes_on_reconnect(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """d2h_pipelined=True across a producer restart: drain before handle close, then re-prime.

    Pipelined mode leaves one async D2H copy in flight after every get_frame_numpy().
    On version change _reinitialize() must:
    - synchronize the D2H stream BEFORE cudaIpcCloseMemHandle releases the copy's
      source mapping (CUDA 719-class source-lifetime race otherwise), and
    - reset priming so the first frame of the new session is a priming call
      (NO_FRAME) rather than the dead session's last frame surfaced as NEW_FRAME.
    """

    class _OrderRecordingFake(FakeCUDAAdapter):
        def __init__(self, device: int = 0) -> None:
            super().__init__(device)
            self.ops: list[str] = []

        def stream_synchronize(self, stream: object) -> None:
            self.ops.append("stream_synchronize")
            super().stream_synchronize(stream)

        def ipc_close_mem_handle(self, ptr: c_void_p) -> None:
            self.ops.append("ipc_close_mem_handle")
            super().ipc_close_mem_handle(ptr)

    shared_memory_cleanup.append(temp_shm_name)
    exp_a, _ = _open_exporter(temp_shm_name)
    try:
        _export_frame(exp_a)  # write_idx → 1

        fake = _OrderRecordingFake()
        policy = dataclasses.replace(ImportPolicy.for_testing(), d2h_pipelined=True)
        imp = Importer.open(ImportSpec(shm_name=temp_shm_name), policy=policy, cuda=fake)
        try:
            # Session A: priming call → NO_FRAME; steady state on the next export.
            r1 = imp.get_frame_numpy()
            assert r1.outcome is ImportOutcome.NO_FRAME  # priming sentinel
            assert imp._numpy is not None and imp._numpy.priming is False

            _export_frame(exp_a)  # write_idx → 2
            r2 = imp.get_frame_numpy()
            assert r2.outcome is ImportOutcome.NEW_FRAME

            # Producer restart: exporter B on the same SHM bumps the version.
            exp_b, _ = _open_exporter(temp_shm_name)
            try:
                fake.ops.clear()
                r3 = imp.get_frame_numpy()
                assert r3.outcome is ImportOutcome.RECONNECTING

                # Drain ordering: the in-flight copy's source mapping must not be
                # closed before the stream is synchronized.
                assert "stream_synchronize" in fake.ops and "ipc_close_mem_handle" in fake.ops
                assert fake.ops.index("stream_synchronize") < fake.ops.index("ipc_close_mem_handle")

                # The surviving pipelined buffer must be re-primed for session B.
                assert imp._numpy.priming is True

                # First visible B frame is a priming call (NO_FRAME), NOT session A's
                # last frame replayed as NEW_FRAME.
                _export_frame(exp_b)
                r4 = imp.get_frame_numpy()
                assert r4.outcome is ImportOutcome.NO_FRAME
                assert imp._numpy.priming is False

                # Steady state resumes: next export yields NEW_FRAME again.
                _export_frame(exp_b)
                r5 = imp.get_frame_numpy()
                assert r5.outcome is ImportOutcome.NEW_FRAME
            finally:
                exp_b.close()
        finally:
            imp.close()
    finally:
        exp_a.close()


def test_close_sets_shutdown_zeroes_handles_and_unlinks(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """Exporter.close() sets the shutdown byte, zeroes IPC handle slots, and unlinks SHM.

    The unlink happens in _do_cleanup STEP 7 (exporter.py:831-838) via a fresh
    SharedMemory(name=...) handle, NOT the exporter's own handle.

    On POSIX the segment stays alive until all handles are closed; on Windows
    the name stays accessible in the namespace until all handles are released.
    So we read the buffer contents (assertions a and b) while the observer is
    still open, then close the observer and ONLY THEN verify the name is gone
    (assertion c).  This ordering is correct on both platforms.
    """
    shared_memory_cleanup.append(temp_shm_name)
    exp, _ = _open_exporter(temp_shm_name)
    layout = SHMLayout(exp._spec.num_slots)
    num_slots = exp._spec.num_slots

    obs = SharedMemory(name=temp_shm_name)
    try:
        exp.close()  # includes 100ms sleep (STEP 5) + SHM unlink (STEP 7)

        # (a) Shutdown byte == 1 (set in STEP 1 via set_shutdown).
        assert obs.buf[layout.shutdown_offset] == 1

        # (b) IPC handle slots are zeroed (STEP 1).
        for slot in range(num_slots):
            base = layout.slot_offset(slot)
            slot_bytes = bytes(obs.buf[base : base + 128])  # 128 = SLOT_SIZE
            assert slot_bytes == b"\x00" * 128, f"slot {slot} handles not zeroed after close"
    finally:
        obs.close()

    # (c) After observer is closed, the name must be gone.
    # On POSIX this holds immediately after unlink; on Windows the name stays in
    # the kernel namespace until the last handle (the observer) is released, so
    # the check must come AFTER obs.close().
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=temp_shm_name)


def test_importer_full_restart_cycle_shutdown_then_reconnect(
    temp_shm_name: str, shared_memory_cleanup: list[str]
) -> None:
    """Importer with reconnect_enabled=True survives a full producer shutdown→restart.

    Sequence:
    1. Open exporter A + connected importer → NEW_FRAME.
    2. Close exporter A → importer sees SHUTDOWN.
    3. While exporter is down → importer sees RECONNECTING.
    4. Reopen exporter B on same name → importer eventually reconnects.
    5. Export one frame → importer sees NEW_FRAME.

    reconnect_enabled is an ImportPolicy field (C1 correction).
    The backoff schedule (reconnect_backoff_frames=(1,2,4,8,16,32,64,120)) means the
    importer drives reconnect on the next call after retry_interval_frames passes —
    so we loop up to ~6 get_frame_numpy() calls; a single-call assert is brittle.
    """
    shared_memory_cleanup.append(temp_shm_name)
    exp_a, _ = _open_exporter(temp_shm_name)
    _export_frame(exp_a)

    imp, _ = _open_importer(temp_shm_name, reconnect=True)
    try:
        r = imp.get_frame_numpy()
        assert r.outcome is ImportOutcome.NEW_FRAME

        # Shut down producer A.
        exp_a.close()

        # Next call detects SHUTDOWN and arms retry state.
        r = imp.get_frame_numpy()
        assert r.outcome is ImportOutcome.SHUTDOWN

        # While producer is down, subsequent calls return RECONNECTING.
        r = imp.get_frame_numpy()
        assert r.outcome is ImportOutcome.RECONNECTING

        # Restart producer B on the same SHM name.
        exp_b, _ = _open_exporter(temp_shm_name)
        try:
            # Drive the retry loop until connected.  BackoffSchedule starts at
            # retry_interval_frames=1 so connection should happen within a few calls.
            MAX_RETRIES = 20
            for _ in range(MAX_RETRIES):
                if imp._initialized:
                    break
                imp.get_frame_numpy()
            else:
                pytest.fail(f"Importer did not reconnect within {MAX_RETRIES} get_frame_numpy() calls")

            assert imp._initialized

            # One export → NEW_FRAME from reconnected importer.
            _export_frame(exp_b)
            r = imp.get_frame_numpy()
            # First call after reconnect may return NO_FRAME if _last_write_idx >= write_idx.
            # If so, export one more frame.
            if r.outcome is ImportOutcome.NO_FRAME:
                _export_frame(exp_b)
                r = imp.get_frame_numpy()
            assert r.outcome is ImportOutcome.NEW_FRAME
        finally:
            exp_b.close()
    finally:
        imp.close()


# ---------------------------------------------------------------------------
# Section C — Ring-buffer slot wrap
# ---------------------------------------------------------------------------


def test_ring_wrap_write_idx_monotonic_and_slot_mapping(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """write_idx is monotonically increasing; slot maps as (write_idx-1) % num_slots."""
    shared_memory_cleanup.append(temp_shm_name)
    num_slots = 3
    num_frames = 8
    exp, _ = _open_exporter(temp_shm_name, num_slots=num_slots)
    try:
        layout = SHMLayout(num_slots)
        obs = SharedMemory(name=temp_shm_name)
        try:
            last_write_idx = 0
            for k in range(num_frames):
                _export_frame(exp)
                wi = read_write_idx(obs.buf)
                assert wi == k + 1, f"frame {k}: expected write_idx={k + 1}, got {wi}"

                result = acquire_slot(obs.buf, layout, last_write_idx=last_write_idx, last_version=1)
                assert result.state is SlotState.NEW_FRAME
                assert result.slot == k % num_slots, f"frame {k}: expected slot={k % num_slots}, got {result.slot}"
                last_write_idx = wi
        finally:
            obs.close()
    finally:
        exp.close()


def test_ring_wrap_memcpy_destinations_cycle(temp_shm_name: str, shared_memory_cleanup: list[str]) -> None:
    """memcpy_async destinations cycle through dev_ptrs[k % num_slots].

    Uses _RecordingFake to capture all memcpy_async calls.  Verifies:
    - dst.value == exp.dev_ptrs[k % num_slots].value  (slot cycling)
    - fake.allocations[dst.value] == exp.buffer_size   (2 MiB-aligned allocation — C4)
    - count == exp.data_size                           (copy is exactly data_size bytes)

    ExportPolicy.for_testing() sets use_graphs=False, guaranteeing the legacy
    memcpy path (no CUDA Graphs) so every export issues exactly one memcpy_async.
    """
    shared_memory_cleanup.append(temp_shm_name)
    num_slots = 3
    num_frames = 8
    recording_fake = _RecordingFake()
    exp, _ = _open_exporter(temp_shm_name, num_slots=num_slots, fake=recording_fake)
    try:
        for _ in range(num_frames):
            _export_frame(exp)

        calls = recording_fake.memcpy_calls
        assert len(calls) == num_frames, f"Expected {num_frames} memcpy calls, got {len(calls)}"

        for k, (dst, _src, count, _kind) in enumerate(calls):
            expected_slot = k % num_slots
            expected_dst_val = exp.dev_ptrs[expected_slot].value

            assert dst.value == expected_dst_val, (
                f"call {k}: expected dst=0x{expected_dst_val:x} (slot {expected_slot}), got 0x{dst.value:x}"
            )
            # Buffer size is 2 MiB-aligned (exporter.py:278); data_size is raw byte count.
            assert recording_fake.allocations.get(dst.value) == exp.buffer_size, (
                f"call {k}: allocation size != buffer_size ({exp.buffer_size})"
            )
            assert count == exp.data_size, f"call {k}: copy count {count} != data_size {exp.data_size}"
    finally:
        exp.close()
