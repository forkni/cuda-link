"""Tests for SpoutSender against FakeSpoutBackend (no GPU)."""

from __future__ import annotations

import pytest
from cuda_link_spout import FakeSpoutBackend, SendOutcome, SpoutFrame, SpoutSender, SpoutSenderSpec
from cuda_link_spout._format import DXGI_FORMAT_R8G8B8A8_UNORM


def test_spec_validation():
    with pytest.raises(ValueError):
        SpoutSenderSpec("", 8, 8)  # empty name
    with pytest.raises(ValueError):
        SpoutSenderSpec("s", 0, 8)  # bad dims
    with pytest.raises(ValueError):
        SpoutSenderSpec("s", 8, 8, fmt="RGB8")  # bad format


def test_open_creates_sender_with_resolved_dxgi_format():
    b = FakeSpoutBackend()
    with SpoutSender.open(SpoutSenderSpec("ai_out", 320, 240, "RGBA8"), backend=b) as tx:
        assert tx.is_open
        (info,) = b.created_senders.values()
        assert info["name"] == "ai_out"
        assert info["dxgi_format"] == DXGI_FORMAT_R8G8B8A8_UNORM
        assert info["width"] == 320 and info["height"] == 240


def test_send_spoutframe_forwards_geometry_and_bpp():
    b = FakeSpoutBackend()
    with SpoutSender.open(SpoutSenderSpec("o", 100, 50, "RGBA16F"), backend=b) as tx:
        out = tx.send(SpoutFrame(ptr=0xABC0, width=100, height=50, pitch=100 * 8, stream=7))
        assert out is SendOutcome.SENT
        rec = b.sent[-1]
        assert rec["src_ptr"] == 0xABC0
        assert rec["bytes_per_pixel"] == 8  # RGBA16F
        assert rec["src_pitch"] == 100 * 8
        assert rec["stream"] == 7


class _FakeTensor:
    """Duck-typed torch-like tensor exposing .data_ptr()."""

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr


def test_send_tensor_derives_geometry_from_spec():
    b = FakeSpoutBackend()
    with SpoutSender.open(SpoutSenderSpec("o", 64, 64, "RGBA8"), backend=b) as tx:
        tx.send(_FakeTensor(0x9999), stream=3)
        rec = b.sent[-1]
        assert rec["src_ptr"] == 0x9999
        assert rec["width"] == 64 and rec["height"] == 64
        assert rec["src_pitch"] == 64 * 4  # tightly packed
        assert rec["stream"] == 3


def test_send_via_cuda_array_interface():
    b = FakeSpoutBackend()

    class _Cupyish:
        __cuda_array_interface__ = {"data": (0x4242, False), "shape": (64, 64, 4)}

    with SpoutSender.open(SpoutSenderSpec("o", 64, 64, "RGBA8"), backend=b) as tx:
        tx.send(_Cupyish())
        assert b.sent[-1]["src_ptr"] == 0x4242


def test_close_is_idempotent_and_send_after_close_fails():
    b = FakeSpoutBackend()
    tx = SpoutSender.open(SpoutSenderSpec("o", 8, 8), backend=b)
    tx.close()
    tx.close()  # no raise
    assert not tx.is_open
    assert tx.send(SpoutFrame(0, 8, 8, 32)) is SendOutcome.FAILED


def test_context_manager_closes_backend_sender():
    b = FakeSpoutBackend()
    with SpoutSender.open(SpoutSenderSpec("o", 8, 8), backend=b):
        pass
    assert b.closed_senders  # close_sender was called on exit
