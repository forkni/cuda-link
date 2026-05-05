"""Tests for src/cuda_link/activation_barrier.py — F9 cross-process SHM barrier."""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import pytest

from cuda_link.activation_barrier import (
    _STRUCT,
    MAGIC,
    SHM_NAME,
    SHM_SIZE,
    VERSION,
    bump_skip,
    decrement,
    increment,
    open_or_create,
    read_state,
)

# ---------------------------------------------------------------------------
# Helpers
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
    yield
    _cleanup(SHM_NAME)


# ---------------------------------------------------------------------------
# Layout tests
# ---------------------------------------------------------------------------


def test_layout_struct_size() -> None:
    assert _STRUCT.size == SHM_SIZE == 64


def test_layout_magic_version_roundtrip() -> None:
    shm = open_or_create(create=True)
    try:
        data = bytes(shm.buf[:SHM_SIZE])
        fields = _STRUCT.unpack(data)
        assert fields[0] == MAGIC
        assert fields[1] == VERSION
    finally:
        shm.close()
        _cleanup(SHM_NAME)


# ---------------------------------------------------------------------------
# open_or_create
# ---------------------------------------------------------------------------


def test_open_or_create_creates_on_missing() -> None:
    shm = open_or_create(create=True)
    assert shm is not None
    shm.close()


def test_open_or_create_opens_existing() -> None:
    # On Windows, SHM is alive only while at least one handle is open; keep shm1.
    shm1 = open_or_create(create=True)
    try:
        shm2 = open_or_create(create=False)
        assert shm2 is not None
        shm2.close()
    finally:
        shm1.close()


def test_open_or_create_raises_when_missing_and_no_create() -> None:
    with pytest.raises(FileNotFoundError):
        open_or_create(create=False)


# ---------------------------------------------------------------------------
# increment / decrement
# ---------------------------------------------------------------------------


def test_increment_decrement_roundtrip() -> None:
    shm = open_or_create(create=True)
    try:
        count = increment(shm, pid=1234)
        assert count == 1

        count2, ts, _ = read_state(shm)
        assert count2 == 1
        assert ts > 0

        result = decrement(shm, pid=1234)
        assert result == 0

        count3, _, _ = read_state(shm)
        assert count3 == 0
    finally:
        shm.close()


def test_decrement_clamps_at_zero() -> None:
    shm = open_or_create(create=True)
    try:
        decrement(shm, pid=1)
        decrement(shm, pid=1)
        count, _, _ = read_state(shm)
        assert count == 0, "active_count must never go below zero"
    finally:
        shm.close()


def test_multi_sender_stacking() -> None:
    shm = open_or_create(create=True)
    try:
        increment(shm, pid=100)
        c2 = increment(shm, pid=101)
        assert c2 == 2

        count, _, _ = read_state(shm)
        assert count == 2

        decrement(shm, pid=100)
        count2, _, _ = read_state(shm)
        assert count2 == 1

        decrement(shm, pid=101)
        count3, _, _ = read_state(shm)
        assert count3 == 0
    finally:
        shm.close()


# ---------------------------------------------------------------------------
# bump_skip
# ---------------------------------------------------------------------------


def test_bump_skip_increments_counter() -> None:
    shm = open_or_create(create=True)
    try:
        bump_skip(shm)
        bump_skip(shm)
        bump_skip(shm)
        _, _, skips = read_state(shm)
        assert skips == 3
    finally:
        shm.close()
