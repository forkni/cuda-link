"""Tests for the inbound (spout->ipc) bridge spec-key helper (no cuda_link / GPU needed)."""

from __future__ import annotations

from dataclasses import dataclass

from cuda_link_spout._format import resolve_format
from cuda_link_spout.bridge import _exporter_spec_key


@dataclass
class _FakeFrame:
    """Minimal stand-in for ReceivedFrame — only the fields _exporter_spec_key reads."""

    width: int
    height: int
    fmt: object


def _frame(width: int, height: int, fmt_name: str) -> _FakeFrame:
    return _FakeFrame(width=width, height=height, fmt=resolve_format(fmt_name))


def test_spec_key_equal_for_identical_geometry():
    a = _frame(1920, 1080, "RGBA8")
    b = _frame(1920, 1080, "RGBA8")
    assert _exporter_spec_key(a) == _exporter_spec_key(b)


def test_spec_key_differs_on_width_change():
    a = _frame(1920, 1080, "RGBA8")
    b = _frame(1280, 1080, "RGBA8")
    assert _exporter_spec_key(a) != _exporter_spec_key(b)


def test_spec_key_differs_on_height_change():
    a = _frame(1920, 1080, "RGBA8")
    b = _frame(1920, 720, "RGBA8")
    assert _exporter_spec_key(a) != _exporter_spec_key(b)


def test_spec_key_differs_on_format_change_same_channels():
    """RGBA8 → RGBA16F: same channel count (4) but different dtype."""
    a = _frame(1024, 1024, "RGBA8")
    b = _frame(1024, 1024, "RGBA16F")
    assert _exporter_spec_key(a) != _exporter_spec_key(b)


def test_spec_key_differs_on_format_change_to_float32():
    a = _frame(512, 512, "RGBA16F")
    b = _frame(512, 512, "RGBA32F")
    assert _exporter_spec_key(a) != _exporter_spec_key(b)


def test_spec_key_is_a_tuple():
    frame = _frame(640, 480, "BGRA8")
    key = _exporter_spec_key(frame)
    assert isinstance(key, tuple)
    assert len(key) == 4  # (width, height, channels, dtype)
    width, height, channels, dtype = key
    assert width == 640
    assert height == 480
    assert channels == 4  # all Spout formats are 4-channel
    assert dtype == "uint8"  # BGRA8 → uint8
