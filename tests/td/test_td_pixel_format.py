"""TDPixelFormat: TD pixel format names <-> the wire format, both directions in one table."""

from __future__ import annotations

from dataclasses import replace

import pytest
import TDPixelFormat
from Importer import Format
from SHMProtocol import FLAGS_BGRA, FLAGS_MONO_ALPHA, DtypeCodec

# name -> (dtype, channels, flags on the wire, the name the receiver gives the Script TOP)
# The last column differs from the first for 1-channel names: the receiver cannot tell
# "mono" from "a" (no flag carries it) and labels both "r...".
_CASES: dict[str, tuple[str, int, int, str]] = {
    "rgba32float": ("float32", 4, 0, "rgba32float"),
    "rgba16fixed": ("uint16", 4, 0, "rgba16fixed"),
    "rgba8fixed": ("uint8", 4, FLAGS_BGRA, "rgba8fixed"),
    "rg32float": ("float32", 2, 0, "rg32float"),
    "rg16fixed": ("uint16", 2, 0, "rg16fixed"),
    "rg8fixed": ("uint8", 2, 0, "rg8fixed"),
    "mono32float": ("float32", 1, 0, "r32float"),
    "mono16fixed": ("uint16", 1, 0, "r16fixed"),
    "mono8fixed": ("uint8", 1, 0, "r8fixed"),
    "a32float": ("float32", 1, 0, "r32float"),
    "a16fixed": ("uint16", 1, 0, "r16fixed"),
    "a8fixed": ("uint8", 1, 0, "r8fixed"),
    "monoalpha32float": ("float32", 2, FLAGS_MONO_ALPHA, "monoalpha32float"),
    "monoalpha16fixed": ("uint16", 2, FLAGS_MONO_ALPHA, "monoalpha16fixed"),
    "monoalpha8fixed": ("uint8", 2, FLAGS_MONO_ALPHA, "monoalpha8fixed"),
}


def test_cases_cover_every_shareable_name() -> None:
    shareable = set(TDPixelFormat.NAME_TO_DTYPE) - set(TDPixelFormat.F16_CONVERT_TARGET)
    assert set(_CASES) == shareable


@pytest.mark.parametrize("name", sorted(_CASES), ids=sorted(_CASES))
def test_name_to_wire_and_back(name: str) -> None:
    dtype, channels, flags, receiver_name = _CASES[name]

    assert TDPixelFormat.to_wire(name) == (dtype, channels, flags)

    kind, bits, dtype_flags = DtypeCodec.encode(dtype)
    on_wire = replace(Format.from_overrides((2, 2, channels), dtype), kind=kind, bits=bits, flags=flags | dtype_flags)
    assert TDPixelFormat.to_name(on_wire) == receiver_name


@pytest.mark.parametrize(
    "name", ["", "useinput", "rgb10a2fixed", "rgba11float"], ids=["empty", "useinput", "10bit", "11bit"]
)
def test_unmapped_name_has_no_wire_format(name: str) -> None:
    assert TDPixelFormat.to_wire(name) is None
    # The Status text then shows what the CUDA memory says.
    assert TDPixelFormat.label(name, "uint8", 4) == "uint8 4ch"


def test_label_of_a_mapped_name_is_the_name() -> None:
    assert TDPixelFormat.label("rgba16fixed", "uint16", 4) == "rgba16fixed"


@pytest.mark.parametrize(
    "f16_name", sorted(TDPixelFormat.F16_CONVERT_TARGET), ids=sorted(TDPixelFormat.F16_CONVERT_TARGET)
)
def test_float16_share_target_is_the_float32_twin(f16_name: str) -> None:
    target = TDPixelFormat.F16_CONVERT_TARGET[f16_name]

    f16_dtype, f16_channels = TDPixelFormat.NAME_TO_DTYPE[f16_name]
    flags = TDPixelFormat.channel_flags(target, f16_channels, "float32")

    assert f16_dtype == "float16"
    assert TDPixelFormat.to_wire(target) == ("float32", f16_channels, flags)
    assert target == f16_name.replace("16float", "32float")


def test_bgra_flag_follows_the_frame_not_the_name() -> None:
    """An unmapped name on a 4-channel uint8 frame still gets FLAGS_BGRA."""
    assert TDPixelFormat.channel_flags("useinput", 4, "uint8") == FLAGS_BGRA
    assert TDPixelFormat.channel_flags("useinput", 4, "uint16") == 0
    assert TDPixelFormat.channel_flags("useinput", 3, "uint8") == 0


def test_engines_use_the_one_mapping() -> None:
    """The old names in the two engines are the TDPixelFormat objects, not copies."""
    import TDReceiver
    import TDSender

    assert TDSender._PIXEL_FMT_NAME_TO_DTYPE is TDPixelFormat.NAME_TO_DTYPE
    assert TDSender._F16_CONVERT_TARGET is TDPixelFormat.F16_CONVERT_TARGET
    assert TDSender._channel_flags is TDPixelFormat.channel_flags
    assert TDReceiver.td_format_string is TDPixelFormat.to_name
