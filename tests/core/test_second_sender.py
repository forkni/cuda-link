"""
A second sender on a SharedMemory name that a running sender holds is refused.

Before, Exporter.open() attached to the existing segment and wrote its own header and IPC
handles over the first sender's, with no error on either side.

All tests use FakeCUDAAdapter — no GPU required.
"""

from __future__ import annotations

import contextlib
import gc
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from multiprocessing.shared_memory import SharedMemory

import pytest

from cuda_link import ExportPolicy, FrameSpec
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.cuda_runtime_types import CudaIpcError, SenderConflictError
from cuda_link.exporter import _OPEN_EXPORTERS, Exporter, _process_is_running
from cuda_link.shm_protocol import (
    MAGIC_OFFSET,
    NUM_SLOTS_OFFSET,
    SHM_HEADER_SIZE,
    SHMLayout,
    read_sender_pid,
    running_sender_pid,
    set_shutdown,
)

_OTHER_PID = 0x7FFF_FFF0  # not a pid this machine gives out; its liveness is patched per test


def _spec(*, num_slots: int = 2) -> FrameSpec:
    return FrameSpec(
        shm_name=f"test_second_sender_{uuid.uuid4().hex[:8]}",
        height=4,
        width=4,
        channels=4,
        dtype="uint8",
        num_slots=num_slots,
        device=0,
    )


def _open(spec: FrameSpec, cuda: FakeCUDAAdapter | None = None) -> Exporter:
    return Exporter.open(spec, policy=ExportPolicy.for_testing(), cuda=cuda or FakeCUDAAdapter(device=0))


@contextlib.contextmanager
def _segment_of_a_sender(spec: FrameSpec, *, sender_pid: int, closed: bool = False) -> Iterator[SharedMemory]:
    """A segment as a sender of process *sender_pid* leaves it: publishing, or closed."""
    layout = SHMLayout(spec.num_slots)
    shm = SharedMemory(name=spec.shm_name, create=True, size=layout.total_size)
    try:
        image = layout.build_buffer(sender_pid=sender_pid)
        shm.buf[: len(image)] = image
        if closed:
            set_shutdown(shm.buf, layout)
        yield shm
    finally:
        shm.close()
        with contextlib.suppress(FileNotFoundError):
            shm.unlink()


# ---------------------------------------------------------------------------
# running_sender_pid: what the segment says
# ---------------------------------------------------------------------------


def _image(*, sender_pid: int = 4242, num_slots: int = 2) -> bytearray:
    return SHMLayout(num_slots).build_buffer(sender_pid=sender_pid)


def test_running_sender_pid_reads_the_pid_of_a_publishing_sender() -> None:
    assert running_sender_pid(memoryview(_image(sender_pid=4242))) == 4242


def test_running_sender_pid_is_zero_after_the_sender_set_the_shutdown_flag() -> None:
    image = _image()
    set_shutdown(memoryview(image), SHMLayout(2))

    assert running_sender_pid(memoryview(image)) == 0


def test_running_sender_pid_is_zero_when_the_segment_names_no_sender() -> None:
    assert running_sender_pid(memoryview(_image(sender_pid=0))) == 0


def test_running_sender_pid_is_zero_for_another_protocol() -> None:
    image = _image()
    image[MAGIC_OFFSET] ^= 0xFF

    assert running_sender_pid(memoryview(image)) == 0


@pytest.mark.parametrize("num_slots", [0, 0xFFFF], ids=["zero-slots", "too-many-slots"])
def test_running_sender_pid_is_zero_for_an_invalid_slot_count(num_slots: int) -> None:
    image = _image()
    image[NUM_SLOTS_OFFSET : NUM_SLOTS_OFFSET + 4] = num_slots.to_bytes(4, "little")

    assert running_sender_pid(memoryview(image)) == 0


@pytest.mark.parametrize(
    "size",
    [0, SHM_HEADER_SIZE - 1, SHMLayout(2).shutdown_offset],
    ids=["empty", "short-header", "no-shutdown-flag"],
)
def test_running_sender_pid_is_zero_for_a_segment_too_short_to_read(size: int) -> None:
    assert running_sender_pid(memoryview(_image())[:size]) == 0


def test_running_sender_pid_reads_a_segment_that_ends_at_the_shutdown_flag() -> None:
    size = SHMLayout(2).shutdown_offset + 1

    assert running_sender_pid(memoryview(_image(sender_pid=7))[:size]) == 7


# ---------------------------------------------------------------------------
# _process_is_running
# ---------------------------------------------------------------------------


def test_this_process_is_running() -> None:
    assert _process_is_running(os.getpid()) is True


@pytest.mark.parametrize("pid", [0, -1], ids=["zero", "negative"])
def test_a_pid_that_is_not_a_process_id_is_not_running(pid: int) -> None:
    assert _process_is_running(pid) is False


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill is the POSIX branch")
def test_a_pid_above_the_system_pid_type_is_not_running() -> None:
    """The header field is 32 bits unsigned; os.kill raises OverflowError for 0xFFFFFFF0."""
    assert _process_is_running(0xFFFF_FFF0) is False


def test_a_pid_that_os_kill_cannot_take_is_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """The POSIX branch on any platform: os.kill raising OverflowError means no such process."""

    def kill(pid: int, sig: int) -> None:
        raise OverflowError("signed integer is greater than maximum")

    monkeypatch.setattr("cuda_link.exporter.sys.platform", "linux")
    monkeypatch.setattr("cuda_link.exporter.os.kill", kill)

    assert _process_is_running(0xFFFF_FFF0) is False


@pytest.mark.parametrize(
    ("error", "running"),
    [(PermissionError("not ours"), True), (ProcessLookupError("gone"), False), (None, True)],
    ids=["not-allowed-to-signal", "no-such-process", "signal-accepted"],
)
def test_the_posix_branch_reads_the_answer_of_os_kill(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None, running: bool
) -> None:
    def kill(pid: int, sig: int) -> None:
        assert (pid, sig) == (4242, 0)
        if error is not None:
            raise error

    monkeypatch.setattr("cuda_link.exporter.sys.platform", "linux")
    monkeypatch.setattr("cuda_link.exporter.os.kill", kill)

    assert _process_is_running(4242) is running


class _FakeKernel32:
    """kernel32 as _process_is_running uses it; records the handles it closed."""

    def __init__(self, *, handle: int, exit_code: int | None) -> None:
        self._handle = handle
        self._exit_code = exit_code
        self.closed: list[int] = []
        # Plain functions, like ctypes function objects: the caller sets argtypes and restype
        # on them, and a bound method takes no attributes.
        self.OpenProcess = lambda *args: self._open_process(*args)
        self.GetExitCodeProcess = lambda *args: self._get_exit_code_process(*args)
        self.CloseHandle = lambda *args: self._close_handle(*args)

    def _open_process(self, access: int, inherit: bool, pid: int) -> int:
        assert (access, inherit, pid) == (0x1000, False, 4242)
        return self._handle

    def _get_exit_code_process(self, handle: int, exit_code_ref: object) -> int:
        if self._exit_code is None:
            return 0
        exit_code_ref._obj.value = self._exit_code  # type: ignore[attr-defined]
        return 1

    def _close_handle(self, handle: int) -> int:
        self.closed.append(handle)
        return 1


def _windows_branch(monkeypatch: pytest.MonkeyPatch, kernel32: _FakeKernel32, last_error: int = 0) -> None:
    import ctypes

    monkeypatch.setattr("cuda_link.exporter.sys.platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error: kernel32, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error, raising=False)


@pytest.mark.parametrize(
    ("exit_code", "running"),
    [(259, True), (0, False), (1, False), (None, True)],
    ids=["still-active", "exited-clean", "exited-with-error", "exit-code-unreadable"],
)
def test_the_windows_branch_reads_the_exit_code_and_closes_the_handle(
    monkeypatch: pytest.MonkeyPatch, exit_code: int | None, running: bool
) -> None:
    kernel32 = _FakeKernel32(handle=77, exit_code=exit_code)
    _windows_branch(monkeypatch, kernel32)

    assert _process_is_running(4242) is running
    assert kernel32.closed == [77]


@pytest.mark.parametrize(
    ("last_error", "running"),
    [(5, True), (87, False)],
    ids=["access-denied", "no-such-process"],
)
def test_the_windows_branch_without_a_handle_reads_the_last_error(
    monkeypatch: pytest.MonkeyPatch, last_error: int, running: bool
) -> None:
    """A process this one may not open still exists; any other error means no process."""
    kernel32 = _FakeKernel32(handle=0, exit_code=None)
    _windows_branch(monkeypatch, kernel32, last_error)

    assert _process_is_running(4242) is running
    assert kernel32.closed == []


def test_a_process_that_exited_is_not_running() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 -- our own interpreter
    child.wait(timeout=60)

    assert _process_is_running(child.pid) is False


def test_a_child_that_still_runs_is_running() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)  # noqa: S603
    try:
        assert _process_is_running(child.pid) is True
    finally:
        child.communicate(timeout=60)


# ---------------------------------------------------------------------------
# Exporter.open: two senders of one process
# ---------------------------------------------------------------------------


def test_a_second_sender_of_this_process_is_refused_before_any_gpu_allocation() -> None:
    spec = _spec()
    first = _open(spec)
    second_cuda = FakeCUDAAdapter(device=0)
    try:
        with pytest.raises(SenderConflictError, match="already has a sender in this process"):
            _open(spec, second_cuda)

        assert second_cuda.allocations == {}
        assert read_sender_pid(first.shm_handle.buf) == os.getpid()  # the first sender's header is intact
        assert first._initialized
    finally:
        first.close()


def test_the_conflict_is_a_cuda_ipc_error() -> None:
    """Callers that catch CudaIpcError or RuntimeError keep working."""
    assert issubclass(SenderConflictError, CudaIpcError)
    assert issubclass(SenderConflictError, RuntimeError)


def test_the_name_is_free_again_after_the_first_sender_closes() -> None:
    spec = _spec()
    _open(spec).close()

    second = _open(spec)
    try:
        assert _OPEN_EXPORTERS.get(spec.shm_name) is second
    finally:
        second.close()
    assert spec.shm_name not in _OPEN_EXPORTERS


def test_a_sender_on_another_name_is_not_affected() -> None:
    first, second = _open(_spec()), _open(_spec())
    try:
        assert first._initialized
        assert second._initialized
    finally:
        first.close()
        second.close()


def test_a_sender_dropped_without_close_does_not_block_the_name() -> None:
    """The record of open senders is weak. The segment still names this process as its
    sender with the shutdown flag clear, and that alone is not a conflict."""
    spec = _spec()
    dropped = _open(spec)
    keep_segment = SharedMemory(name=spec.shm_name)  # a receiver that still holds it
    try:
        del dropped
        gc.collect()

        second = _open(spec)
        try:
            assert _OPEN_EXPORTERS.get(spec.shm_name) is second
        finally:
            second.close()
    finally:
        keep_segment.close()


def test_a_sender_that_failed_to_open_does_not_block_the_name() -> None:
    spec = _spec()
    failing = FakeCUDAAdapter(device=1)  # device mismatch: open() raises
    with pytest.raises(CudaIpcError, match="Device mismatch"):
        _open(spec, failing)
    assert spec.shm_name not in _OPEN_EXPORTERS

    second = _open(spec)
    try:
        assert second._initialized
    finally:
        second.close()


# ---------------------------------------------------------------------------
# Exporter.open: a segment that names another process
# ---------------------------------------------------------------------------


def test_a_running_sender_of_another_process_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec()
    asked: list[int] = []
    monkeypatch.setattr("cuda_link.exporter._process_is_running", lambda pid: asked.append(pid) or True)
    cuda = FakeCUDAAdapter(device=0)

    with _segment_of_a_sender(spec, sender_pid=_OTHER_PID) as segment:
        with pytest.raises(SenderConflictError, match=rf"running sender \(process {_OTHER_PID}\)"):
            _open(spec, cuda)

        assert asked == [_OTHER_PID]
        assert cuda.allocations == {}
        assert read_sender_pid(segment.buf) == _OTHER_PID  # the other sender's header is intact


def test_the_segment_of_a_sender_that_crashed_is_taken_over(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pid no longer runs: a receiver kept the segment alive after the sender died."""
    spec = _spec()
    monkeypatch.setattr("cuda_link.exporter._process_is_running", lambda pid: False)

    with _segment_of_a_sender(spec, sender_pid=_OTHER_PID) as segment:
        exporter = _open(spec)
        try:
            assert read_sender_pid(segment.buf) == os.getpid()
        finally:
            exporter.close()


def test_the_segment_of_a_sender_that_closed_is_taken_over(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shutdown flag is set: the other process still runs, but its sender is closed."""
    spec = _spec()
    asked: list[int] = []
    monkeypatch.setattr("cuda_link.exporter._process_is_running", lambda pid: asked.append(pid) or True)

    with _segment_of_a_sender(spec, sender_pid=_OTHER_PID, closed=True) as segment:
        exporter = _open(spec)
        try:
            assert read_sender_pid(segment.buf) == os.getpid()
        finally:
            exporter.close()

    assert asked == []


def test_a_segment_that_names_this_process_is_not_a_conflict_by_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two senders of one process are told apart by the record of open senders, not by the pid."""
    spec = _spec()
    asked: list[int] = []
    monkeypatch.setattr("cuda_link.exporter._process_is_running", lambda pid: asked.append(pid) or True)

    with _segment_of_a_sender(spec, sender_pid=os.getpid()):
        _open(spec).close()

    assert asked == []


def test_a_segment_that_cannot_be_read_is_taken_over(monkeypatch: pytest.MonkeyPatch) -> None:
    """The header read fails: nothing says that a sender runs, and the open writes a new header."""
    spec = _spec()

    def unreadable(buf: memoryview) -> int:
        raise ValueError("released buffer")

    monkeypatch.setattr("cuda_link.exporter.running_sender_pid", unreadable)
    monkeypatch.setattr("cuda_link.exporter._process_is_running", lambda pid: True)

    with _segment_of_a_sender(spec, sender_pid=_OTHER_PID) as segment:
        exporter = _open(spec)
        try:
            assert read_sender_pid(segment.buf) == os.getpid()
        finally:
            exporter.close()


def test_a_segment_of_another_protocol_is_taken_over(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec()
    monkeypatch.setattr("cuda_link.exporter._process_is_running", lambda pid: True)

    with _segment_of_a_sender(spec, sender_pid=_OTHER_PID) as segment:
        segment.buf[MAGIC_OFFSET] ^= 0xFF
        exporter = _open(spec)
        try:
            assert read_sender_pid(segment.buf) == os.getpid()
        finally:
            exporter.close()
