"""Unit tests for the `.tox` Receiver Framewaitms doorbell wait.

TDReceiverEngine.wait_for_new_frame() runs from onFrameStart after has_new_frame()
returned False. It waits on the sender's doorbell in 1 ms slices and polls write_idx after
each slice (same rules as the C++ In TOP's waitForFreshFrame()).

Harness: _make_receiver / _inject_receiver_connection from test_tdreceiver_has_new_frame.
The Doorbell module is replaced by a fake, so no Win32 event and no real CUDA is needed.
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import pytest

from tests.td.test_tdreceiver_has_new_frame import (
    _inject_receiver_connection,
    _make_receiver,
    _set_shm_write_idx,
    _write_shm_frame,
    shm_cleanup,  # noqa: F401 -- pytest fixture
)


class _FakeDoorbell:
    """Stand-in for the Doorbell module: counts calls and runs a hook on each wait."""

    def __init__(self, handle: object = 1234, on_wait=None) -> None:
        self.handle = handle
        self.on_wait = on_wait
        self.open_calls: list[str] = []
        self.wait_calls: list[int] = []
        self.closed: list[object] = []

    def doorbell_event_name(self, shm_name: str) -> str:
        return "Local\\cudalink_db_" + shm_name

    def open_doorbell(self, name: str) -> object:
        self.open_calls.append(name)
        return self.handle

    def wait(self, handle: object, timeout_ms: int) -> bool:
        self.wait_calls.append(timeout_ms)
        return bool(self.on_wait(len(self.wait_calls))) if self.on_wait else False

    def close(self, handle: object) -> None:
        self.closed.append(handle)


def _connected_engine(shm: SharedMemory, monkeypatch: pytest.MonkeyPatch, fake: _FakeDoorbell):
    import TDReceiver

    from cuda_link.shm_protocol import FORMAT_KIND_FLOAT

    _write_shm_frame(shm, 1, 64, 64, 4, FORMAT_KIND_FLOAT, 32, 0, 64 * 64 * 4 * 4)
    engine = _make_receiver("wait_test")
    _inject_receiver_connection(engine, shm, ipc_version=1, write_idx=0)
    monkeypatch.setattr(TDReceiver, "_doorbell", fake)
    return engine


def test_next_wait_slice_ms_rounds_up_and_stops() -> None:
    from TDReceiver import next_wait_slice_ms

    assert next_wait_slice_ms(0) == 0
    assert next_wait_slice_ms(-5) == 0
    assert next_wait_slice_ms(1) == 1  # a remainder under 1 ms still waits
    assert next_wait_slice_ms(999) == 1
    assert next_wait_slice_ms(3000) == 1  # never longer than one slice
    assert next_wait_slice_ms(3000, slice_ms=2) == 2
    assert next_wait_slice_ms(1500, slice_ms=4) == 2
    assert next_wait_slice_ms(1000, slice_ms=0) == 0


def test_frame_wait_ms_default_clamp_and_bad_value(shm_cleanup: SharedMemory) -> None:  # noqa: F811
    from TDReceiver import FRAME_WAIT_DEFAULT_MS, FRAME_WAIT_MAX_MS

    engine = _make_receiver("wait_test")
    assert engine.frame_wait_ms() == FRAME_WAIT_DEFAULT_MS == 3  # par missing -> default

    for value, expected in ((0, 0), (5, 5), (99, FRAME_WAIT_MAX_MS), (-4, 0), ("x", 3)):
        engine._host.param_value = lambda name, v=value: v if name == "Framewaitms" else None
        assert engine.frame_wait_ms() == expected


def test_wait_off_or_not_connected_returns_at_once(
    shm_cleanup: SharedMemory,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDoorbell()
    engine = _connected_engine(shm_cleanup, monkeypatch, fake)

    assert engine.wait_for_new_frame(0) is False
    engine._initialized = False
    assert engine.wait_for_new_frame(3) is False
    assert fake.open_calls == [] and fake.wait_calls == []


def test_no_doorbell_keeps_poll_once_and_retries_open(
    shm_cleanup: SharedMemory,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDoorbell(handle=None)
    engine = _connected_engine(shm_cleanup, monkeypatch, fake)

    assert engine.wait_for_new_frame(3) is False
    assert engine.wait_for_new_frame(3) is False
    assert len(fake.open_calls) == 2  # the open is tried again on each waiting frame start
    assert fake.wait_calls == []


def test_wait_gets_frame_published_during_the_wait(
    shm_cleanup: SharedMemory,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def publish_on_second_wait(n: int) -> bool:
        if n == 2:
            _set_shm_write_idx(shm_cleanup, 1)
            return True
        return False  # first slice: timeout (or a signal another consumer took)

    fake = _FakeDoorbell(on_wait=publish_on_second_wait)
    engine = _connected_engine(shm_cleanup, monkeypatch, fake)

    assert engine.has_new_frame() is False
    assert engine.wait_for_new_frame(8) is True
    assert fake.wait_calls == [1, 1]
    assert fake.open_calls == ["Local\\cudalink_db_wait_test"]
    assert engine.frame_wait_rescued_count == 1
    assert engine.get_stats()["rx_frame_wait_rescued"] == 1


def test_stale_wake_without_new_frame_waits_again_then_gives_up(
    shm_cleanup: SharedMemory,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDoorbell(on_wait=lambda n: True)  # every wake is stale: write_idx never moves
    engine = _connected_engine(shm_cleanup, monkeypatch, fake)

    assert engine.wait_for_new_frame(2) is False
    # A fake wait returns at once, so the budget never runs out: the iteration cap ends it.
    assert len(fake.wait_calls) == 2 * 2 + 2
    assert engine.frame_wait_rescued_count == 0


def test_budget_end_stops_the_wait(
    shm_cleanup: SharedMemory,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import TDReceiver

    clock = iter([0.0, 0.0005, 0.0012, 0.0021, 0.0030, 0.0040, 0.0050])
    monkeypatch.setattr(TDReceiver.time, "perf_counter", lambda: next(clock))
    fake = _FakeDoorbell()
    engine = _connected_engine(shm_cleanup, monkeypatch, fake)

    assert engine.wait_for_new_frame(2) is False  # deadline 2.0 ms; third check is at 2.1 ms
    assert fake.wait_calls == [1, 1]


def test_handle_is_reused_and_closed_on_cleanup(
    shm_cleanup: SharedMemory,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDoorbell()
    engine = _connected_engine(shm_cleanup, monkeypatch, fake)

    engine.wait_for_new_frame(1)
    engine.wait_for_new_frame(1)
    assert len(fake.open_calls) == 1

    engine._close_doorbell()
    engine._close_doorbell()
    assert fake.closed == [1234]
    assert engine._doorbell_handle is None
