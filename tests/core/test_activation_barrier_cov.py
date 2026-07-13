"""Coverage-focused tests for RealShmAdapter in src/cuda_link/activation_barrier.py.

test_activation_barrier.py already covers the module-level SHM-IO functions
(open_or_create/read_state/increment/decrement/bump_skip) directly. The
Checker/Holder role-class test files exercise RealShmAdapter only via an
isinstance() structural check, never calling its methods. This file drives
RealShmAdapter's methods directly against a real (but test-local) named
SharedMemory segment — no GPU required.
"""

from __future__ import annotations

import contextlib
from multiprocessing.shared_memory import SharedMemory

import pytest

from cuda_link.activation_barrier import _STRUCT, MAGIC, SHM_NAME, VERSION, RealShmAdapter

# ---------------------------------------------------------------------------
# Helpers (duplicated from test_activation_barrier.py — new test files must
# not edit existing ones)
# ---------------------------------------------------------------------------


def _cleanup(name: str) -> None:
    try:
        shm = SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def cleanup_barrier():
    _cleanup(SHM_NAME)
    # On Windows a live TD session (or a not-yet-GC'd handle elsewhere in this
    # process) can keep the named SHM alive past unlink() because SHM lifetime
    # is handle-bound (not name-bound like POSIX). If the segment persists,
    # re-zero its contents so each test starts from a deterministic zero state.
    with contextlib.suppress(FileNotFoundError):
        shm = SharedMemory(name=SHM_NAME)
        try:
            _STRUCT.pack_into(shm.buf, 0, MAGIC, VERSION, 0, 0, 0, 0, 0, b"\x00" * 32)
        finally:
            shm.close()
    yield
    _cleanup(SHM_NAME)


# ---------------------------------------------------------------------------
# is_attached / attach
# ---------------------------------------------------------------------------


def test_is_attached_false_initially():
    adapter = RealShmAdapter()
    assert adapter.is_attached is False


def test_attach_sets_is_attached_true():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        assert adapter.is_attached is True
    finally:
        adapter.close()


def test_attach_is_idempotent_does_not_recreate():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        first_shm = adapter._shm
        adapter.attach(create=True)  # second call must be a no-op
        assert adapter._shm is first_shm
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# read_state
# ---------------------------------------------------------------------------


def test_read_state_before_attach_raises():
    adapter = RealShmAdapter()
    with pytest.raises(RuntimeError, match="attach\\(\\) not called"):
        adapter.read_state()


def test_read_state_after_attach_returns_tuple():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        state = adapter.read_state()
        assert state == (0, 0, 0)
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# bump_skip
# ---------------------------------------------------------------------------


def test_bump_skip_before_attach_is_noop():
    adapter = RealShmAdapter()
    adapter.bump_skip()  # must not raise even though never attached
    assert adapter._shm is None


def test_bump_skip_after_attach_increments():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        adapter.bump_skip()
        _, _, skips = adapter.read_state()
        assert skips == 1
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# open_and_increment
# ---------------------------------------------------------------------------


def test_open_and_increment_creates_when_absent():
    adapter = RealShmAdapter()
    try:
        count = adapter.open_and_increment(pid=1234)
        assert count == 1
        assert adapter.is_attached is True
    finally:
        adapter.close()


def test_open_and_increment_reuses_existing_shm():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        first_shm = adapter._shm
        adapter.open_and_increment(pid=1)
        assert adapter._shm is first_shm
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# decrement
# ---------------------------------------------------------------------------


def test_decrement_before_open_raises():
    adapter = RealShmAdapter()
    with pytest.raises(RuntimeError, match="open_and_increment\\(\\) not called"):
        adapter.decrement(pid=1234)


def test_decrement_after_open_and_increment_works():
    adapter = RealShmAdapter()
    try:
        adapter.open_and_increment(pid=1)
        count = adapter.decrement(pid=1)
        assert count == 0
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_before_attach_is_noop():
    adapter = RealShmAdapter()
    adapter.close()  # must not raise
    assert adapter._shm is None


def test_close_is_idempotent():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    adapter.close()
    adapter.close()  # second call must be a no-op, not raise
    assert adapter._shm is None


def test_close_swallows_oserror_from_underlying_shm(monkeypatch):
    adapter = RealShmAdapter()
    adapter.attach(create=True)

    def _raise_oserror():
        raise OSError("simulated close failure")

    monkeypatch.setattr(adapter._shm, "close", _raise_oserror)
    adapter.close()  # must not raise — OSError is suppressed
    assert adapter._shm is None
