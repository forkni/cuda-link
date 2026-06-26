"""Tests for the bridge CLI arg parsing and spec-building (no cuda_link / GPU needed)."""

from __future__ import annotations

import pytest
from cuda_link_spout._format import DXGI_FORMAT_R8G8B8A8_UNORM
from cuda_link_spout.bridge import BridgeArgs, parse_args, receiver_spec, sender_spec


def test_parse_out_direction():
    args = parse_args(["--dir", "out", "--ipc", "ipc_a", "--spout", "spo", "--width", "1024", "--height", "768"])
    assert isinstance(args, BridgeArgs)
    assert args.direction == "out"
    assert args.ipc == "ipc_a" and args.spout == "spo"
    assert args.width == 1024 and args.height == 768
    assert args.fmt == "RGBA8"  # default


def test_parse_in_direction_allows_zero_dims():
    args = parse_args(["--dir", "in", "--ipc", "ai_in", "--spout", "resolume_out"])
    assert args.direction == "in"
    assert args.width == 0 and args.height == 0  # geometry learned from the sender


def test_out_requires_positive_dims():
    with pytest.raises(ValueError, match="requires positive"):
        parse_args(["--dir", "out", "--ipc", "i", "--spout", "s"])  # no width/height


def test_missing_required_args_exits():
    with pytest.raises(SystemExit):
        parse_args(["--dir", "out"])  # missing --ipc/--spout


def test_invalid_direction_exits():
    with pytest.raises(SystemExit):
        parse_args(["--dir", "sideways", "--ipc", "i", "--spout", "s"])


def test_sender_spec_from_args():
    args = parse_args(
        ["--dir", "out", "--ipc", "i", "--spout", "s", "--width", "640", "--height", "480", "--fmt", "RGBA8"]
    )
    spec = sender_spec(args)
    assert spec.name == "s" and spec.width == 640 and spec.height == 480
    assert spec.resolved_format.dxgi_format == DXGI_FORMAT_R8G8B8A8_UNORM


def test_receiver_spec_from_args():
    args = parse_args(["--dir", "in", "--ipc", "i", "--spout", "s", "--device", "1"])
    spec = receiver_spec(args)
    assert spec.name == "s" and spec.device == 1


def test_bad_fmt_in_out_spec_raises():
    args = parse_args(["--dir", "out", "--ipc", "i", "--spout", "s", "--width", "8", "--height", "8", "--fmt", "RGB8"])
    with pytest.raises(ValueError):
        sender_spec(args)  # SpoutSenderSpec validates the format
