"""Tests for SpoutReceiver against FakeSpoutBackend (no GPU)."""

from __future__ import annotations

from cuda_link_spout import (
    FakeSpoutBackend,
    ReceivedFrame,
    ReceiveOutcome,
    SpoutReceiver,
    SpoutReceiverSpec,
)
from cuda_link_spout._format import DXGI_FORMAT_R32G32B32A32_FLOAT


def test_open_default_spec_binds_active_sender():
    b = FakeSpoutBackend()
    with SpoutReceiver.open(backend=b) as rx:
        assert rx.is_open
        (info,) = b.created_receivers.values()
        assert info["name"] == ""  # active sender


def test_receive_new_frame_resolves_format_and_ptr():
    b = FakeSpoutBackend()
    b.fake_width, b.fake_height = 800, 600
    b.fake_dxgi_format = DXGI_FORMAT_R32G32B32A32_FLOAT
    with SpoutReceiver.open(SpoutReceiverSpec("resolume_out"), backend=b) as rx:
        frame = rx.receive()
        assert isinstance(frame, ReceivedFrame)
        assert frame.outcome is ReceiveOutcome.NEW_FRAME
        assert frame.width == 800 and frame.height == 600
        assert frame.ptr == b.fake_dst_ptr
        assert frame.fmt is not None and frame.fmt.name == "RGBA32F"


def test_receive_no_new_frame():
    b = FakeSpoutBackend()
    b.fake_new_frame = False
    with SpoutReceiver.open(SpoutReceiverSpec("s"), backend=b) as rx:
        frame = rx.receive()
        assert frame.outcome is ReceiveOutcome.NO_FRAME
        assert frame.ptr == 0


def test_receive_not_connected():
    b = FakeSpoutBackend()
    b.fake_connected = False
    with SpoutReceiver.open(SpoutReceiverSpec("s"), backend=b) as rx:
        assert rx.receive().outcome is ReceiveOutcome.NOT_CONNECTED


def test_receive_after_close_returns_failed():
    b = FakeSpoutBackend()
    rx = SpoutReceiver.open(SpoutReceiverSpec("s"), backend=b)
    rx.close()
    assert rx.receive().outcome is ReceiveOutcome.FAILED


def test_context_manager_closes_backend_receiver():
    b = FakeSpoutBackend()
    with SpoutReceiver.open(SpoutReceiverSpec("s"), backend=b):
        pass
    assert b.closed_receivers
