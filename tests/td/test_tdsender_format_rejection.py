"""
Regression test: _is_unsupported_format must return True for all 6 formats that
cudaMemory() cannot handle correctly in TD 2025.32820, and False for the 11 that work.

Source of truth: verification/results/cuda_memory_probe_20260510_090919.json
(empirical probe run on TD 2025.32820 / CUDA 12.8 with a cupy non-blocking stream).

Rejected formats (5 hard exceptions + 1 silent corruption):
  - All 4 float16 variants → "Texture is invalid pixel format to be shared with CUDA."
  - 10-bit RGB / 2-bit Alpha fixed → "Source TOP has unsupported pixel format."
  - 11-bit float (RGB) → "succeeds" but returns uint8/4ch (raw byte layout); silent
    receiver corruption if not converted.
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "td_exporter"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _make_engine(shm_name: str):
    from TDConfig import TDSenderConfig
    from TDHost import TDHost
    from TDSender import TDSenderEngine

    class _NullHost(TDHost):
        def param_value(self, name):
            return {"Active": True, "Debug": False}.get(name)

        def set_param_value(self, name, value):
            pass

        def set_param_enabled(self, name, enabled):
            pass

        def show_custom_only(self, value):
            pass

        def is_active(self):
            return True

        def find_top(self, name):
            return None

        def set_warning_status(self, msg):
            pass

        def set_error_status(self, msg):
            pass

        def clear_status(self):
            pass

        def set_info_status(self, msg):
            pass

    config = TDSenderConfig()
    return TDSenderEngine(
        host=_NullHost(),
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=shm_name,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Format classification table — taken verbatim from the probe artifact.
# Each entry: (pixel_format_string, should_need_conversion)
# ---------------------------------------------------------------------------

_FORMATS = [
    # Supported — cudaMemory() returns correct dtype/shape
    ("8-bit fixed (RGBA)", False),
    ("16-bit fixed (RGBA)", False),
    ("32-bit float (RGBA)", False),
    ("8-bit fixed (R)", False),
    ("16-bit fixed (R)", False),
    ("32-bit float (R)", False),
    ("8-bit fixed (RG)", False),
    ("16-bit fixed (RG)", False),
    ("32-bit float (RG)", False),
    ("8-bit fixed (A)", False),
    ("32-bit float (A)", False),
    # Rejected outright — "Texture is invalid pixel format to be shared with CUDA."
    ("16-bit float (RGBA)", True),
    ("16-bit float (R)", True),
    ("16-bit float (RG)", True),
    ("16-bit float (A)", True),
    # Rejected outright — "Source TOP has unsupported pixel format." (Bug A)
    ("10-bit RGB, 2-bit Alpha", True),
    # Silent corruption — returns uint8/4ch instead of 11:11:10 packed float (Bug B)
    ("11-bit float (RGB)", True),
    # Abbreviated substring variants also present in _CUDA_UNSUPPORTED_PIXEL_FORMATS
    ("16float", True),
    ("10bit", True),
    ("11bit", True),
]


@pytest.fixture
def engine():
    eng = _make_engine(f"test_fmt_rej_{uuid.uuid4().hex[:8]}")
    yield eng


@pytest.mark.parametrize("pixel_fmt,expected", _FORMATS, ids=[f[0] for f in _FORMATS])
def test_is_unsupported_format(engine, pixel_fmt, expected):
    """_is_unsupported_format must match probe-derived expected value for every TD format."""
    fake_top = types.SimpleNamespace(pixel_format=pixel_fmt)
    result = engine._is_unsupported_format(fake_top)
    assert result is expected, (
        f"_is_unsupported_format({pixel_fmt!r}) returned {result}, expected {expected}.\n"
        "Source of truth: verification/results/cuda_memory_probe_20260510_090919.json"
    )


def test_cache_is_consistent(engine):
    """Calling _is_unsupported_format twice with the same format returns the same result."""
    fake_top = types.SimpleNamespace(pixel_format="16-bit float (RGBA)")
    first = engine._is_unsupported_format(fake_top)
    second = engine._is_unsupported_format(fake_top)
    assert first is second is True


def test_cache_resets_on_format_change(engine):
    """Switching pixel_format on the same engine clears the cache correctly."""
    fake_top = types.SimpleNamespace(pixel_format="32-bit float (RGBA)")
    assert engine._is_unsupported_format(fake_top) is False
    fake_top.pixel_format = "16-bit float (RGBA)"
    assert engine._is_unsupported_format(fake_top) is True
