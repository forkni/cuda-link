"""Tests for the SpoutBackend seam and FakeSpoutBackend."""

from __future__ import annotations

import pytest
from cuda_link_spout import FakeSpoutBackend, SpoutBackend
from cuda_link_spout._backend import NativeReceiveResult


def test_fake_satisfies_protocol():
    assert isinstance(FakeSpoutBackend(), SpoutBackend)


def test_create_and_close_sender_tracks_state():
    b = FakeSpoutBackend()
    h = b.create_sender("s", 640, 480, 28, 0)
    assert h in b.created_senders
    assert b.created_senders[h]["width"] == 640
    b.close_sender(h)
    assert h not in b.created_senders
    assert h in b.closed_senders


def test_send_records_args_and_rejects_unknown_handle():
    b = FakeSpoutBackend()
    h = b.create_sender("s", 8, 8, 28, 0)
    b.send(h, 0x1234, 32, 8, 8, 4, 0)
    assert b.sent[-1]["src_ptr"] == 0x1234
    assert b.sent[-1]["bytes_per_pixel"] == 4
    with pytest.raises(RuntimeError):
        b.send(0xDEAD, 0, 0, 0, 0, 0, 0)


def test_receive_returns_dst_ptr_only_on_new_frame():
    b = FakeSpoutBackend()
    h = b.create_receiver("r", 0)
    r = b.receive(h, 0, 0, 0)
    assert isinstance(r, NativeReceiveResult)
    assert r.connected and r.new_frame
    assert r.dst_ptr == b.fake_dst_ptr

    b.fake_new_frame = False
    r2 = b.receive(h, 0, 0, 0)
    assert r2.connected and not r2.new_frame
    assert r2.dst_ptr == 0


def test_receive_not_connected():
    b = FakeSpoutBackend()
    h = b.create_receiver("r", 0)
    b.fake_connected = False
    r = b.receive(h, 0, 0, 0)
    assert not r.connected and not r.new_frame


def test_failure_injection():
    b = FakeSpoutBackend()
    b.fail_on_create_sender = True
    with pytest.raises(RuntimeError):
        b.create_sender("s", 8, 8, 28, 0)
