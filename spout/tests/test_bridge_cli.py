"""Tests for the bridge CLI arg parsing and spec-building (no cuda_link / GPU needed)."""

from __future__ import annotations

import pytest
from cuda_link_spout._format import DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, format_from_dtype
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


def test_parse_out_direction_no_dims():
    """--dir out no longer requires --width/--height; geometry is derived from the IPC frame."""
    args = parse_args(["--dir", "out", "--ipc", "i", "--spout", "s"])
    assert args.direction == "out"
    assert args.width == 0 and args.height == 0  # lazy-derive from the first frame


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


# ---------------------------------------------------------------------------
# format_from_dtype — auto-geometry helper
# ---------------------------------------------------------------------------


def test_format_from_dtype_uint8_gives_bgra8():
    # TD's cudaMemory() returns BGRA bytes for uint8 4ch; the auto-derived tag must match.
    fmt = format_from_dtype("uint8")
    assert fmt.name == "BGRA8"
    assert fmt.dxgi_format == DXGI_FORMAT_B8G8R8A8_UNORM
    assert fmt.bgra is True


def test_format_from_dtype_float16_gives_rgba16f():
    from cuda_link_spout._format import DXGI_FORMAT_R16G16B16A16_FLOAT

    fmt = format_from_dtype("float16")
    assert fmt.name == "RGBA16F"
    assert fmt.dxgi_format == DXGI_FORMAT_R16G16B16A16_FLOAT


def test_format_from_dtype_float32_gives_rgba32f():
    from cuda_link_spout._format import DXGI_FORMAT_R32G32B32A32_FLOAT

    fmt = format_from_dtype("float32")
    assert fmt.name == "RGBA32F"
    assert fmt.dxgi_format == DXGI_FORMAT_R32G32B32A32_FLOAT


def test_format_from_dtype_unknown_raises():
    with pytest.raises(ValueError, match="Cannot derive Spout format"):
        format_from_dtype("bfloat16")


def test_format_from_dtype_uint8_matches_td_bgra_convention():
    """uint8 auto-derives BGRA8 to match TD's cudaMemory() uint8=BGRA wire format."""
    fmt = format_from_dtype("uint8")
    assert fmt.name == "BGRA8", "uint8 auto-derive must return BGRA8 (TD cudaMemory() convention)"
    assert fmt.bgra is True


# ---------------------------------------------------------------------------
# --verbose / --debug flag
# ---------------------------------------------------------------------------


def test_verbose_defaults_to_false():
    args = parse_args(["--dir", "out", "--ipc", "i", "--spout", "s"])
    assert args.verbose is False


def test_verbose_flag_sets_verbose_true():
    args = parse_args(["--dir", "out", "--ipc", "i", "--spout", "s", "--verbose"])
    assert args.verbose is True


def test_debug_alias_sets_verbose_true():
    """--debug is an alias for --verbose."""
    args = parse_args(["--dir", "in", "--ipc", "i", "--spout", "s", "--debug"])
    assert args.verbose is True


def test_verbose_not_in_bridgeargs_repr_when_false():
    """BridgeArgs is a frozen dataclass — verbose=False is the default; check field presence."""
    args = parse_args(["--dir", "out", "--ipc", "i", "--spout", "s"])
    assert hasattr(args, "verbose")
    assert args.verbose is False
