"""Tests for the R2 Win32 named-event doorbell.

Three groups:
    1. Pure (platform-agnostic): doorbell_event_name() formatting.
    2. Importer primitive (GPU-free, FakeCUDAAdapter): wait_for_doorbell() logic
       exercised via make_connected_importer() without real Win32 handles.
    3. Real kernel32 round-trip (Windows-only, no GPU): create / open / signal /
       wait / close sequence using actual Win32 API calls.
"""

from __future__ import annotations

import dataclasses
import os
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# 1. Pure tests — platform-agnostic
# ---------------------------------------------------------------------------


def test_doorbell_event_name_format() -> None:
    """Name function returns the expected Local\\ prefix + suffix."""
    from cuda_link._doorbell import doorbell_event_name

    assert doorbell_event_name("foo") == r"Local\cudalink_db_foo"
    assert doorbell_event_name("my_shm") == r"Local\cudalink_db_my_shm"
    assert doorbell_event_name("") == r"Local\cudalink_db_"


# ---------------------------------------------------------------------------
# 2. Importer wait_for_doorbell() primitive — GPU-free, FakeCUDAAdapter
# ---------------------------------------------------------------------------


def _make_importer(write_idx: int = 1, last_write_idx: int = 0) -> object:
    """Return a connected Importer via fakes.make_connected_importer()."""
    from fakes import make_connected_importer  # type: ignore[import]

    return make_connected_importer(write_idx=write_idx, last_write_idx=last_write_idx)


def test_wait_for_doorbell_returns_false_when_handle_none() -> None:
    """wait_for_doorbell returns False immediately when doorbell_handle is None."""
    imp = _make_importer()
    # conn.doorbell_handle defaults to None — disabled path
    assert imp._conn.doorbell_handle is None
    result = imp.wait_for_doorbell(2.0)
    assert result is False


def test_wait_for_doorbell_returns_false_when_policy_disabled_even_with_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A handle opened for the native wait backend must not make a disabled doorbell block."""
    import cuda_link.importer as _imp_mod

    imp = _make_importer(write_idx=1, last_write_idx=1)  # no frame waiting
    imp._conn.doorbell_handle = 0xFEED
    assert imp._policy.doorbell is False
    waits: list = []
    monkeypatch.setattr(_imp_mod._doorbell, "wait", lambda h, ms: waits.append((h, ms)) or True)
    try:
        result = imp.wait_for_doorbell(250.0)
    finally:
        imp._conn.doorbell_handle = None
    assert result is False
    assert waits == []


def test_wait_for_doorbell_returns_true_on_advanced_write_idx() -> None:
    """Early-return path: returns True immediately when a frame is already waiting.

    The early-return check (cur != 0 and cur != _last_write_idx) only runs when
    the doorbell handle is non-None (doorbell is enabled). We inject a sentinel
    object as the handle so the method enters the frame-check branch.
    """
    # write_idx=2 (one frame available), last_write_idx=1 (already consumed one)
    imp = _make_importer(write_idx=2, last_write_idx=1)
    # Inject a sentinel so the method doesn't short-circuit on handle=None.
    # The early-return path doesn't call any Win32 functions, so any truthy
    # object works as the handle here.
    sentinel = object()
    imp._conn.doorbell_handle = sentinel
    imp._policy = dataclasses.replace(imp._policy, doorbell=True)
    try:
        result = imp.wait_for_doorbell(2.0)
    finally:
        # Restore to None so __del__ doesn't try CloseHandle(sentinel).
        imp._conn.doorbell_handle = None
    assert result is True, "Should return True without blocking when frame already waiting"


def test_wait_for_doorbell_returns_false_when_conn_none() -> None:
    """wait_for_doorbell returns False when _conn is None (not connected)."""
    from cuda_link._cuda_adapters import FakeCUDAAdapter
    from cuda_link._importer_port import ImportPolicy, ImportSpec
    from cuda_link.importer import Importer

    imp = Importer(ImportSpec(shm_name="test_db_disconnected"), ImportPolicy(), FakeCUDAAdapter())
    # _conn is None before open() → should not raise, just return False
    assert imp._conn is None
    result = imp.wait_for_doorbell(2.0)
    assert result is False


def _armed_importer(monkeypatch: pytest.MonkeyPatch, waits: list, wait_result=None, write_idx_seq=None):
    """Connected Importer with no frame waiting, a fake doorbell handle and doorbell on.

    _doorbell.wait records each timeout and returns wait_result(call_number) (default
    False: a slice that times out). write_idx_seq, if given, is what read_write_idx returns
    on successive polls (the early check first).
    """
    import cuda_link.importer as _imp_mod

    imp = _make_importer(write_idx=1, last_write_idx=1)
    imp._conn.doorbell_handle = 0xFEED
    imp._policy = dataclasses.replace(imp._policy, doorbell=True)

    def fake_wait(h, ms):
        waits.append(ms)
        return bool(wait_result(len(waits))) if wait_result else False

    monkeypatch.setattr(_imp_mod._doorbell, "wait", fake_wait)
    if write_idx_seq is not None:
        seq = iter(write_idx_seq)
        monkeypatch.setattr(_imp_mod, "read_write_idx", lambda buf: next(seq))
    return imp


def test_wait_for_doorbell_runs_in_one_ms_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    """No signal and no frame: 1 ms slices until the timeout, then False."""
    waits: list = []
    imp = _armed_importer(monkeypatch, waits)
    try:
        assert imp.wait_for_doorbell(5.0) is False
    finally:
        imp._conn.doorbell_handle = None
    assert waits, "must wait at least one slice"
    assert all(ms == 1 for ms in waits)
    assert len(waits) <= 5 + 2  # iteration cap: timeout / slice + 2


def test_wait_for_doorbell_signal_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list = []
    imp = _armed_importer(monkeypatch, waits, wait_result=lambda n: n == 2)
    try:
        assert imp.wait_for_doorbell(50.0) is True
    finally:
        imp._conn.doorbell_handle = None
    assert len(waits) == 2


def test_wait_for_doorbell_taken_signal_costs_one_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another consumer took the signal: the slice times out, write_idx moved -> True."""
    waits: list = []
    # Early check: 1 (no frame). After slice 1: 1. After slice 2: 2 (a frame landed).
    imp = _armed_importer(monkeypatch, waits, write_idx_seq=[1, 1, 2])
    try:
        assert imp.wait_for_doorbell(2000.0) is True
    finally:
        imp._conn.doorbell_handle = None
    assert waits == [1, 1]


def test_wait_for_doorbell_failed_wait_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead handle (wait returns False at once) stops at the iteration cap."""
    waits: list = []
    imp = _armed_importer(monkeypatch, waits)
    try:
        assert imp.wait_for_doorbell(20.0) is False
    finally:
        imp._conn.doorbell_handle = None
    assert len(waits) <= 20 + 2


def test_wait_for_doorbell_zero_timeout_polls_the_event(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list = []
    imp = _armed_importer(monkeypatch, waits, wait_result=lambda n: True)
    try:
        assert imp.wait_for_doorbell(0) is True
    finally:
        imp._conn.doorbell_handle = None
    assert waits == [0]


def test_import_policy_doorbell_default_on_and_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from cuda_link._importer_port import ImportPolicy

    monkeypatch.delenv("CUDALINK_DOORBELL", raising=False)
    assert ImportPolicy().doorbell is True
    assert ImportPolicy.from_env().doorbell is True
    assert ImportPolicy.for_testing().doorbell is False
    monkeypatch.setenv("CUDALINK_DOORBELL", "0")
    assert ImportPolicy.from_env().doorbell is False


# ---------------------------------------------------------------------------
# 3. Real kernel32 round-trip — Windows-only, no GPU
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_doorbell_roundtrip_signal_and_wait() -> None:
    """create → open → signal → wait returns True (signaled)."""
    from cuda_link._doorbell import (
        close,
        create_doorbell,
        doorbell_event_name,
        open_doorbell,
        wait,
    )

    name = doorbell_event_name("test_roundtrip_pytest")
    producer_h = create_doorbell(name)
    assert producer_h is not None, "CreateEventW failed"

    consumer_h = open_doorbell(name)
    assert consumer_h is not None, "OpenEventW failed — event not found after create"

    try:
        # Signal and immediately wait — should fire within 1 s
        from cuda_link import _doorbell as db_mod

        db_mod.signal(producer_h)
        result = wait(consumer_h, 1000)
        assert result is True, "WaitForSingleObject should return WAIT_OBJECT_0 after SetEvent"
    finally:
        close(producer_h)
        close(consumer_h)


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_doorbell_wait_timeout_on_unsignaled() -> None:
    """wait on an unsignaled event with a short timeout returns False."""
    from cuda_link._doorbell import (
        close,
        create_doorbell,
        doorbell_event_name,
        wait,
    )

    name = doorbell_event_name("test_timeout_pytest")
    handle = create_doorbell(name)
    assert handle is not None

    try:
        # 10 ms timeout on a never-signaled event → should return False quickly
        result = wait(handle, 10)
        assert result is False, "wait on unsignaled event should return False (WAIT_TIMEOUT)"
    finally:
        close(handle)


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_doorbell_auto_reset_single_wake() -> None:
    """Auto-reset: after one wait() returns True the event is cleared — second wait times out."""
    from cuda_link._doorbell import (
        close,
        create_doorbell,
        doorbell_event_name,
        open_doorbell,
        signal,
        wait,
    )

    name = doorbell_event_name("test_autoreset_pytest")
    producer_h = create_doorbell(name)
    consumer_h = open_doorbell(name)
    assert producer_h is not None and consumer_h is not None

    try:
        signal(producer_h)
        assert wait(consumer_h, 1000) is True, "First wait should be signaled"
        # Auto-reset: event is now clear — second wait should time out
        assert wait(consumer_h, 10) is False, "Second wait after auto-reset should time out"
    finally:
        close(producer_h)
        close(consumer_h)


@pytest.mark.skipif(os.name != "nt", reason="Win32 named events require Windows")
def test_doorbell_close_none_never_touches_closehandle(monkeypatch: pytest.MonkeyPatch) -> None:
    """close(None) is the non-Windows / disabled no-op path: it must short-circuit
    before CloseHandle, which would otherwise be handed a NULL handle."""
    import cuda_link._doorbell as db

    mock_close_handle = MagicMock()
    monkeypatch.setattr(db._k32, "CloseHandle", mock_close_handle)
    db.close(None)
    mock_close_handle.assert_not_called()
