"""Python <-> C++ parity for the doorbell name and the Framewaitms rules.

The C++ Custom TOPs and the Python side hold these values in separate files with no build
link between them. A value changed on one side only gives a receiver that waits on the wrong
event name, or two receivers that follow different wait rules. This test reads the C++
headers as text (no C++ toolchain) and compares each constant with its Python twin.

Sources:
- cpp_top/src/core/doorbell_name.h   <-> cuda_link._doorbell (the event name prefix)
- cpp_top/src/core/frame_wait.h      <-> TDReceiver (cadence guard, wait slice) and
                                         cuda_link.importer (wait slice)
- cpp_top/src/in_top/Parameters.h    <-> TDReceiver (Framewaitms default and maximum)

Skips if the cpp_top tree is not in the checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CPP_SRC = Path(__file__).parent.parent.parent / "cpp_top" / "src"

# "constexpr uint32_t kName = 4;", "constexpr double kName = 0.001;", "constexpr static int kName = 8;"
_NUMBER_RE = re.compile(r"constexpr\s+(?:static\s+)?(?:u?int(?:32_t)?|double)\s+(k\w+)\s*=\s*([0-9.]+)\s*;")
_PREFIX_RE = re.compile(r'constexpr\s+wchar_t\s+kDoorbellPrefix\[\]\s*=\s*L"([^"]*)"\s*;')


def _read(relative: str) -> str:
    path = _CPP_SRC / relative
    if not path.is_file():
        pytest.skip(f"cpp_top C++ tree not present in this checkout: {path}")
    return path.read_text(encoding="utf-8")


def _numbers(relative: str) -> dict[str, float]:
    return {name: float(value) for name, value in _NUMBER_RE.findall(_read(relative))}


def test_doorbell_prefix_matches_python() -> None:
    from cuda_link import _doorbell

    match = _PREFIX_RE.search(_read("core/doorbell_name.h"))
    assert match is not None, "kDoorbellPrefix not found in doorbell_name.h"
    cpp_prefix = match.group(1).replace("\\\\", "\\")  # C++ string escape -> the real text

    assert cpp_prefix == _doorbell._PREFIX
    assert _doorbell.doorbell_event_name("abc") == cpp_prefix + "abc"


@pytest.mark.parametrize(
    ("cpp_name", "py_name"),
    [
        ("kCadenceMinSamples", "CADENCE_MIN_SAMPLES"),
        ("kCadenceMaxOutliers", "CADENCE_MAX_OUTLIERS"),
        ("kCadenceMaxPeriodS", "CADENCE_MAX_PERIOD_S"),
        ("kCadenceSlackS", "CADENCE_SLACK_S"),
        ("kFrameWaitSliceMs", "_FRAME_WAIT_SLICE_MS"),
    ],
    ids=["min_samples", "max_outliers", "max_period", "slack", "wait_slice"],
)
def test_frame_wait_constant_matches_receiver(cpp_name: str, py_name: str) -> None:
    import TDReceiver

    cpp = _numbers("core/frame_wait.h")
    assert cpp_name in cpp, f"{cpp_name} not found in frame_wait.h"
    assert cpp[cpp_name] == getattr(TDReceiver, py_name)


def test_wait_slice_matches_importer() -> None:
    from cuda_link import importer

    assert _numbers("core/frame_wait.h")["kFrameWaitSliceMs"] == importer._DOORBELL_SLICE_MS


@pytest.mark.parametrize(
    ("cpp_name", "py_name"),
    [("kFramewaitDefault", "FRAME_WAIT_DEFAULT_MS"), ("kFramewaitMax", "FRAME_WAIT_MAX_MS")],
    ids=["default", "max"],
)
def test_framewaitms_bounds_match_receiver(cpp_name: str, py_name: str) -> None:
    import TDReceiver

    cpp = _numbers("in_top/Parameters.h")
    assert cpp_name in cpp, f"{cpp_name} not found in in_top/Parameters.h"
    assert cpp[cpp_name] == getattr(TDReceiver, py_name)
