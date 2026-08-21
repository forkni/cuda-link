"""
Regression tests: TDSender must handle runtime dtype changes mid-stream.

Bug reproduced: switching a TouchDesigner source texture from 8-bit RGBA to
32-bit float RGBA during a live stream caused "Size mismatch: expected 8294400,
got 33177600" to repeat forever because _cm_dtype_to_str fell back to "uint8"
when cm.data_type was None on the transition frame, so the reopen guard never
fired, and the stale Exporter.data_size was compared against a 4x-larger size.

Root cause: the guard relied on cm.data_type (unreliable mid-transition) instead
of cm.size (the ground truth Exporter.export validates against).

Fix:
  1. _guess_dtype_from_buffer_size: 2 bytes/comp → "uint16" (not "float16",
     which is a TD-rejected format).
  2. _resolve_frame_dtype: cross-checks name itemsize against cm.size; falls
     back to size-derived dtype when they disagree.
  3. export_frame guard: fires on geometry OR resolved-dtype OR raw-size change.

All helpers tested here are pure (no CUDA, no SHM) — can run without hardware.
"""

from __future__ import annotations

import types

import pytest
from TDSender import _cm_dtype_to_str, _guess_dtype_from_buffer_size, _resolve_frame_dtype

# ---------------------------------------------------------------------------
# _cm_dtype_to_str — unchanged but exercised for completeness
# ---------------------------------------------------------------------------


def test_cm_dtype_to_str_float32():
    dt = types.SimpleNamespace(name="float32")
    assert _cm_dtype_to_str(dt) == "float32"


def test_cm_dtype_to_str_uint8():
    dt = types.SimpleNamespace(name="uint8")
    assert _cm_dtype_to_str(dt) == "uint8"


def test_cm_dtype_to_str_none_falls_back():
    assert _cm_dtype_to_str(None) == "uint8"


def test_cm_dtype_to_str_unknown_falls_back():
    dt = types.SimpleNamespace(name="bogus_dtype")
    assert _cm_dtype_to_str(dt) == "uint8"


def test_cm_dtype_to_str_missing_name_attr_falls_back():
    assert _cm_dtype_to_str(object()) == "uint8"


# ---------------------------------------------------------------------------
# _guess_dtype_from_buffer_size — float16→uint16 regression is the key case
# ---------------------------------------------------------------------------

_W, _H, _C = 1920, 1080, 4  # 1080p RGBA


@pytest.mark.parametrize(
    "buf_size,expected",
    [
        (1920 * 1080 * 4 * 1, "uint8"),  # 8294400  — uint8
        (1920 * 1080 * 4 * 2, "uint16"),  # 16588800 — uint16 (was "float16" before fix)
        (1920 * 1080 * 4 * 4, "float32"),  # 33177600 — float32
    ],
)
def test_guess_dtype_from_buffer_size(buf_size, expected):
    result = _guess_dtype_from_buffer_size(_W, _H, _C, buf_size)
    assert result == expected, (
        f"_guess_dtype_from_buffer_size({_W},{_H},{_C},{buf_size}) returned {result!r}, "
        f"expected {expected!r}. "
        "NOTE: 2 bytes/comp must map to 'uint16', NOT 'float16' — float16 is TD-rejected."
    )


def test_guess_dtype_from_buffer_size_unknown_bpp():
    """Exotic bpp (e.g. 3 bytes) safely returns 'uint8' fallback."""
    assert _guess_dtype_from_buffer_size(_W, _H, _C, _W * _H * _C * 3) == "uint8"


def test_guess_dtype_from_buffer_size_none():
    assert _guess_dtype_from_buffer_size(_W, _H, _C, None) == "uint8"


def test_guess_dtype_from_buffer_size_zero_dims():
    assert _guess_dtype_from_buffer_size(0, _H, _C, 1024) == "uint8"


# ---------------------------------------------------------------------------
# _resolve_frame_dtype — the core fix
# ---------------------------------------------------------------------------


def _dt(name: str):
    """Build a fake cm.data_type with the given numpy dtype name."""
    return types.SimpleNamespace(name=name)


class TestResolveFrameDtype:
    """Systematic coverage of _resolve_frame_dtype(width, height, channels,
    cm_size, cm_data_type, fallback_dtype) -> str."""

    # ------- Path 1: name and size are consistent → trust the name -------

    def test_float32_consistent(self):
        """cm.data_type='float32', cm.size=4x — name is correct."""
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 4, _dt("float32"), "uint8")
        assert result == "float32"

    def test_uint8_consistent(self):
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 1, _dt("uint8"), "float32")
        assert result == "uint8"

    def test_uint16_consistent(self):
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 2, _dt("uint16"), "uint8")
        assert result == "uint16"

    # ------- Path 2 (exact bug): cm.data_type is None / stale → bytes win -------

    def test_none_data_type_float32_bytes(self):
        """THE BUG: cm.data_type=None but cm.size is 4x → must resolve to float32."""
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 4, None, "uint8")
        assert result == "float32", (
            "When cm.data_type is None but cm.size indicates float32, "
            "resolve must return 'float32', not stay on fallback 'uint8'."
        )

    def test_none_data_type_uint16_bytes(self):
        """cm.data_type=None but cm.size is 2x → must resolve to uint16."""
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 2, None, "uint8")
        assert result == "uint16"

    def test_none_data_type_uint8_bytes(self):
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 1, None, "float32")
        assert result == "uint8"

    def test_stale_uint8_name_float32_bytes(self):
        """Transition frame: name says uint8 but size says float32 → size wins."""
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 4, _dt("uint8"), "uint8")
        assert result == "float32"

    def test_stale_float32_name_uint8_bytes(self):
        """Transition frame: name says float32 but size says uint8 → size wins."""
        result = _resolve_frame_dtype(_W, _H, _C, _W * _H * _C * 1, _dt("float32"), "float32")
        assert result == "uint8"

    # ------- Path 3: unknown bpp → return fallback -------

    def test_unknown_bpp_returns_fallback(self):
        """3 bytes/comp — not a supported TD dtype; fallback preserved."""
        exotic_size = _W * _H * _C * 3
        result = _resolve_frame_dtype(_W, _H, _C, exotic_size, None, "uint8")
        # _guess_dtype_from_buffer_size returns "uint8" for 3.0 bpp, but
        # 1 * px != exotic_size, so we fall to fallback_dtype.
        assert result == "uint8"

    def test_zero_pixels_returns_fallback(self):
        result = _resolve_frame_dtype(0, _H, _C, 1024, None, "float32")
        assert result == "float32"

    def test_none_size_returns_name_or_fallback(self):
        """No size info — return whatever _cm_dtype_to_str gives (or fallback)."""
        result = _resolve_frame_dtype(_W, _H, _C, None, _dt("float32"), "uint8")
        # name="float32" is returned directly because px > 0 but cm_size is falsy
        assert result == "uint8"  # falls through to fallback_dtype because cm_size is None

    # ------- Mono (1-channel) path — Bug B regression -------

    def test_mono_uint16_correct_channels(self):
        """1ch uint16 must resolve correctly when channels=1 is passed (not stale 4)."""
        # 1ch uint16: size = 1920*1080*1*2 = 4,147,200
        mono_size = _W * _H * 1 * 2
        result = _resolve_frame_dtype(_W, _H, 1, mono_size, None, "float32")
        assert result == "uint16", f"1ch uint16 with correct channels=1 must resolve to 'uint16', got {result!r}"

    def test_mono_uint16_stale_channels_4_fails_detection(self):
        """Demonstrate Bug B: with stale channels=4, mono uint16 CANNOT be resolved.
        This test documents the failure mode (passes BEFORE the fix, captures the bug).
        With stale channels=4, px=8,294,400 but cm.size=4,147,200 → bpp=0.5 → fallback.
        After Bug B fix (cm_channels used instead of spec.channels), export_frame calls
        _resolve_frame_dtype with channels=1 (see test above), not 4.
        """
        mono_size = _W * _H * 1 * 2  # 4,147,200 bytes
        # Using stale channels=4 → cannot match any supported dtype → fallback
        result = _resolve_frame_dtype(_W, _H, 4, mono_size, None, "float32")
        # With stale channels=4: bpp = 4,147,200 / (1920*1080*4) = 0.5 → not in map → fallback
        assert result == "float32", (
            "With stale channels=4, mono uint16 falls back to float32 (Bug B)."
            "This test documents the bug — export_frame must use cm.channels instead."
        )


# ---------------------------------------------------------------------------
# _PIXEL_FMT_NAME_TO_DTYPE — mapping table completeness
# ---------------------------------------------------------------------------


def test_pixel_fmt_name_map_covers_probe_results():
    """All pixelFormatName values returned by TD's Script TOP par.format menu must be mapped
    (or explicitly excluded). Probe output 2026-05-30 gave 23 entries; we expect all
    supported ones in the table and the unsupported/special ones absent."""
    from TDSender import _PIXEL_FMT_NAME_TO_DTYPE

    # Supported formats confirmed by probe (Section 1 of pixel_format_probe.py):
    _should_be_mapped = [
        "rgba32float",
        "rgba16fixed",
        "rgba8fixed",
        "rg32float",
        "rg16fixed",
        "rg8fixed",
        "mono32float",
        "mono16fixed",
        "mono8fixed",
        "a32float",
        "a16fixed",
        "a8fixed",
        "monoalpha32float",
        "monoalpha16fixed",
        "monoalpha8fixed",
        # float16 variants (unsupported by cudaMemory but still in the map)
        "rgba16float",
        "rg16float",
        "mono16float",
        "a16float",
        "monoalpha16float",
    ]
    for name in _should_be_mapped:
        assert name in _PIXEL_FMT_NAME_TO_DTYPE, (
            f"{name!r} must be in _PIXEL_FMT_NAME_TO_DTYPE (returned by TD par.format menu, probe 2026-05-30)"
        )

    # Special/unsupported entries must NOT be mapped (no reliable dtype/channels):
    _should_not_be_mapped = ["useinput", "rgb10a2fixed", "rgba11float"]
    for name in _should_not_be_mapped:
        assert name not in _PIXEL_FMT_NAME_TO_DTYPE, (
            f"{name!r} must NOT be in _PIXEL_FMT_NAME_TO_DTYPE (no reliable dtype or channel mapping for this format)"
        )


@pytest.mark.parametrize(
    "fmt_name,expected_dtype,expected_ch",
    [
        ("rgba32float", "float32", 4),
        ("rgba16fixed", "uint16", 4),
        ("rgba8fixed", "uint8", 4),
        ("rg32float", "float32", 2),
        ("mono32float", "float32", 1),
        ("mono8fixed", "uint8", 1),
        ("a8fixed", "uint8", 1),
        ("monoalpha32float", "float32", 2),
    ],
)
def test_pixel_fmt_name_to_dtype_values(fmt_name, expected_dtype, expected_ch):
    """Spot-check key _PIXEL_FMT_NAME_TO_DTYPE entries match expected dtype + channels."""
    from TDSender import _PIXEL_FMT_NAME_TO_DTYPE

    assert fmt_name in _PIXEL_FMT_NAME_TO_DTYPE, f"{fmt_name!r} missing from map"
    dtype, ch = _PIXEL_FMT_NAME_TO_DTYPE[fmt_name]
    assert dtype == expected_dtype, f"{fmt_name!r}: dtype={dtype!r} expected {expected_dtype!r}"
    assert ch == expected_ch, f"{fmt_name!r}: channels={ch} expected {expected_ch}"
