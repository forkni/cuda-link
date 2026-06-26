"""Tests for the pure pixel-format mapping (no GPU)."""

from __future__ import annotations

import pytest
from cuda_link_spout import SUPPORTED_FORMATS, format_from_dxgi, resolve_format
from cuda_link_spout._format import (
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    frame_nbytes,
    row_pitch,
)


@pytest.mark.parametrize(
    "name,dxgi,bpp,dtype,bgra",
    [
        ("RGBA8", DXGI_FORMAT_R8G8B8A8_UNORM, 4, "uint8", False),
        ("BGRA8", DXGI_FORMAT_B8G8R8A8_UNORM, 4, "uint8", True),
        ("RGBA16F", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, "float16", False),
        ("RGBA32F", DXGI_FORMAT_R32G32B32A32_FLOAT, 16, "float32", False),
    ],
)
def test_resolve_format_fields(name, dxgi, bpp, dtype, bgra):
    fmt = resolve_format(name)
    assert fmt.dxgi_format == dxgi
    assert fmt.bytes_per_pixel == bpp
    assert fmt.dtype == dtype
    assert fmt.bgra is bgra
    assert fmt.channels == 4


def test_resolve_format_is_case_insensitive():
    assert resolve_format("rgba8") is resolve_format("RGBA8")


def test_resolve_format_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported Spout format"):
        resolve_format("RGB8")  # 3-channel not allowed


def test_supported_formats_exposed():
    assert set(SUPPORTED_FORMATS) == {"RGBA8", "BGRA8", "RGBA16F", "RGBA32F"}


def test_format_from_dxgi_roundtrip():
    for name in SUPPORTED_FORMATS:
        fmt = resolve_format(name)
        assert format_from_dxgi(fmt.dxgi_format) is fmt


def test_format_from_dxgi_rejects_unimportable():
    with pytest.raises(ValueError, match="cannot import"):
        format_from_dxgi(115)  # an arbitrary non-importable DXGI format


def test_pitch_and_nbytes():
    fmt = resolve_format("RGBA32F")  # 16 bpp
    assert row_pitch(1920, fmt) == 1920 * 16
    assert frame_nbytes(1920, 1080, fmt) == 1920 * 1080 * 16
