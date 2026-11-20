# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
TDPixelFormat — the mapping between TouchDesigner pixel format names and the wire format.

One place for what the .tox knows about TD pixel formats:
  - Sender direction: a TOP's pixelFormatName → (dtype, channels, flags) on the wire.
  - Receiver direction: a wire Format → the Script TOP par.format name.
  - Float16: cudaMemory() cannot share it, so each float16 name has a float32 share target.

The two directions are not a full round trip, and that is intended:
  - 1-channel names ("mono...", "a...") come back as "r..." on the receiver.
  - FLAGS_BGRA has no name of its own; it only tells the receiver the channel order.

The C++ Custom TOPs hold their own mapping (cpp_top/src/out_top/pixel_format_map.h); the
intended differences are listed in docs/CPP_TOP.md.

textDAT name: TDPixelFormat  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from SHMProtocol import FLAGS_BGRA, FLAGS_MONO_ALPHA, FORMAT_KIND_FLOAT  # noqa: E402

if TYPE_CHECKING:
    from Importer import Format

# Map TD pixelFormatName values (op.pixelFormatName) → (dtype_str, num_channels).
# The sender uses the name to detect dtype changes that are invisible to cm.size/cm.data_type —
# specifically dtype-shrink transitions (e.g. float32→uint8) where TD keeps the old
# (larger) CUDA allocation unchanged, so neither cm.size nor cm.data_type reflects the change.
# pixelFormatName updates immediately on any format change (grow or shrink).
# Built from the TD Script TOP par.format menu names (Section 1 of pixel_format_probe.py).
# Float16 variants are included for completeness; the sender swaps them for their
# F16_CONVERT_TARGET name before the lookup, so they never resolve to float16.
NAME_TO_DTYPE: dict[str, tuple[str, int]] = {
    # RGBA
    "rgba32float": ("float32", 4),
    "rgba16fixed": ("uint16", 4),
    "rgba8fixed": ("uint8", 4),
    # RG
    "rg32float": ("float32", 2),
    "rg16fixed": ("uint16", 2),
    "rg8fixed": ("uint8", 2),
    # Mono (1 component)
    "mono32float": ("float32", 1),
    "mono16fixed": ("uint16", 1),
    "mono8fixed": ("uint8", 1),
    # Alpha-only (1 component)
    "a32float": ("float32", 1),
    "a16fixed": ("uint16", 1),
    "a8fixed": ("uint8", 1),
    # Mono+Alpha (2 components)
    "monoalpha32float": ("float32", 2),
    "monoalpha16fixed": ("uint16", 2),
    "monoalpha8fixed": ("uint8", 2),
    # Float16 variants (cudaMemory cannot share them; converted via F16_CONVERT_TARGET)
    "rgba16float": ("float16", 4),
    "rg16float": ("float16", 2),
    "mono16float": ("float16", 1),
    "a16float": ("float16", 1),
    "monoalpha16float": ("float16", 2),
    # "useinput", "rgb10a2fixed", "rgba11float" → NOT mapped (no reliable dtype/channel count)
}

# Float16 source → 32-bit float target requested via cudaMemory(pixelFormat=...).
# cudaMemory() cannot share float16 textures, but TD converts on the GPU before sharing,
# and the float32 result matches numpyArray() bit-exactly with the full range kept
# (pixel-format matrix probe, TD 2025.33230). Measured at 1280x720 RGBA, TD 2025.33230:
# cudaMemory() median 0.89 ms converted vs 0.86 ms native float32 (numpyArray() readback
# 3.4 ms), so no CPU round trip. The shared frame is 2x the bytes of the float16 source.
F16_CONVERT_TARGET: dict[str, str] = {
    "rgba16float": "rgba32float",
    "rg16float": "rg32float",
    "mono16float": "mono32float",
    "a16float": "a32float",
    "monoalpha16float": "monoalpha32float",
}


def channel_flags(pixel_format_name: str, channels: int, dtype: str) -> int:
    """Return the SHM channel-semantic flag bits for a TD source frame.

    FLAGS_MONO_ALPHA: 2-channel Mono+Alpha source (R+A, not R+G).
    FLAGS_BGRA: 4-channel uint8 frame. TD's cudaMemory() hands 8-bit RGBA textures
    over in B,G,R,A order (pixel-format matrix probe, TD 2025.33230); 16-bit and
    32-bit RGBA come back R,G,B,A.
    """
    flags = FLAGS_MONO_ALPHA if pixel_format_name.startswith("monoalpha") else 0
    if channels == 4 and dtype == "uint8":
        flags |= FLAGS_BGRA
    return flags


def to_wire(pixel_format_name: str) -> tuple[str, int, int] | None:
    """Map a TD pixelFormatName → (dtype, channels, flags), or None for a name with no mapping.

    None covers "", "useinput" and the packed formats: the caller then takes dtype and
    channels from the CUDA memory itself and calls channel_flags() with those.
    """
    mapped = NAME_TO_DTYPE.get(pixel_format_name)
    if mapped is None:
        return None
    dtype, channels = mapped
    return dtype, channels, channel_flags(pixel_format_name, channels, dtype)


def label(pixel_format_name: str, dtype: str, channels: int) -> str:
    """The format part of the sender's Status text: the TD name when it is a mapped one,
    otherwise the dtype and channel count taken from the CUDA memory."""
    if pixel_format_name in NAME_TO_DTYPE:
        return pixel_format_name
    return f"{dtype} {channels}ch"


def to_name(fmt: Format) -> str:
    """Map a Format value object → TouchDesigner Script TOP par.format string.

    Produces strings matching TD's internal par.format menu values, e.g.:
      (FLOAT, 32, 4, flags=0)               → "rgba32float"
      (UNSIGNED, 16, 4, flags=0)            → "rgba16fixed"
      (UNSIGNED, 8, 1, flags=0)             → "r8fixed"
      (FLOAT, 32, 2, flags=FLAGS_MONO_ALPHA) → "monoalpha32float"
      (FLOAT, 32, 2, flags=0)               → "rg32float"

    TD uses these to allocate the Script TOP's output texture. Setting
    par.format before copyCUDAMemory ensures the texture is the right size
    for the incoming dtype, preventing partial writes into an oversized buffer.

    FLAGS_MONO_ALPHA is preserved in fmt.flags so that mono+alpha sources are
    correctly labelled in the par.format string and in Info/Status messages —
    the user's only signal that the channel layout is R+A (not R+G).
    """
    format_kind, bits_per_comp, num_comps, flags = fmt.kind, fmt.bits, fmt.num_comps, fmt.flags
    if num_comps == 2 and flags & FLAGS_MONO_ALPHA:
        ch = "monoalpha"
    else:
        ch = {1: "r", 2: "rg", 3: "rgb", 4: "rgba"}.get(num_comps, "rgba")
    suffix = "float" if format_kind == FORMAT_KIND_FLOAT else "fixed"
    return f"{ch}{bits_per_comp}{suffix}"
