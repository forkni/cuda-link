"""
Pixel-format mapping between cuda-link frame formats and Spout/DXGI texture formats.

Pure-stdlib logic — no GPU, no native module — so the mapping is fully unit-testable.
The native bridge consumes the resolved `SpoutFormat` (DXGI enum + bytes-per-pixel +
channel order) to create the shared D3D11 texture and to size the de-swizzle copy.

Only fully-typed 4-channel DXGI formats are supported, because CUDA↔D3D11 interop
cannot import 3-channel or ``*_TYPELESS``/sRGB-typed formats (see
docs/competitive/spout-bridge-design.md §2, §5.4).
"""

from __future__ import annotations

from dataclasses import dataclass

# DXGI_FORMAT enum values (winerror-independent constants from dxgiformat.h).
DXGI_FORMAT_R32G32B32A32_FLOAT = 2
DXGI_FORMAT_R16G16B16A16_FLOAT = 10
DXGI_FORMAT_R8G8B8A8_UNORM = 28
DXGI_FORMAT_B8G8R8A8_UNORM = 87


@dataclass(frozen=True)
class SpoutFormat:
    """Resolved description of one Spout-compatible texture format."""

    name: str  # canonical cuda-link name, e.g. "RGBA8"
    dxgi_format: int  # DXGI_FORMAT_* enum value
    bytes_per_pixel: int  # row pitch = width * bytes_per_pixel
    channels: int  # always 4 (Spout is 4-channel)
    dtype: str  # cuda-link dtype string: "uint8" | "float16" | "float32"
    bgra: bool  # True when channel order is BGRA (needs a swap from RGBA sources)


# Canonical name → resolved format. Names are case-insensitive at the API boundary.
_FORMATS: dict[str, SpoutFormat] = {
    "RGBA8": SpoutFormat("RGBA8", DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, "uint8", bgra=False),
    "BGRA8": SpoutFormat("BGRA8", DXGI_FORMAT_B8G8R8A8_UNORM, 4, 4, "uint8", bgra=True),
    "RGBA16F": SpoutFormat("RGBA16F", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 4, "float16", bgra=False),
    "RGBA32F": SpoutFormat("RGBA32F", DXGI_FORMAT_R32G32B32A32_FLOAT, 16, 4, "float32", bgra=False),
}

# Reverse map: DXGI enum → canonical format (for resolving a received sender's format).
_BY_DXGI: dict[int, SpoutFormat] = {fmt.dxgi_format: fmt for fmt in _FORMATS.values()}

#: Tuple of every accepted format name (stable order, for error messages / docs).
SUPPORTED_FORMATS: tuple[str, ...] = tuple(_FORMATS)


def resolve_format(name: str) -> SpoutFormat:
    """Resolve a cuda-link format name to a :class:`SpoutFormat`.

    Args:
        name: One of :data:`SUPPORTED_FORMATS` (case-insensitive).

    Raises:
        ValueError: if the name is not a supported Spout format.
    """
    try:
        return _FORMATS[name.upper()]
    except KeyError:
        raise ValueError(
            f"Unsupported Spout format {name!r}. Spout textures must be fully-typed "
            f"4-channel formats; supported: {', '.join(SUPPORTED_FORMATS)}."
        ) from None


def format_from_dxgi(dxgi_format: int) -> SpoutFormat:
    """Resolve a DXGI_FORMAT enum value (e.g. from a received sender) to a :class:`SpoutFormat`.

    Raises:
        ValueError: if the DXGI format is not one CUDA↔D3D11 interop can import.
    """
    try:
        return _BY_DXGI[dxgi_format]
    except KeyError:
        raise ValueError(
            f"Sender uses DXGI format {dxgi_format} which CUDA↔D3D11 interop cannot import. "
            f"Importable formats: {', '.join(SUPPORTED_FORMATS)}."
        ) from None


def row_pitch(width: int, fmt: SpoutFormat) -> int:
    """Tightly-packed row pitch in bytes for *width* pixels of *fmt*."""
    return width * fmt.bytes_per_pixel


def frame_nbytes(width: int, height: int, fmt: SpoutFormat) -> int:
    """Total tightly-packed byte size of a *width*×*height* frame of *fmt*."""
    return row_pitch(width, fmt) * height


def format_from_dtype(dtype: str) -> SpoutFormat:
    """Derive a :class:`SpoutFormat` from a cuda-link dtype string, matching TD's channel order.

    Used by the ``--dir out`` bridge auto-geometry path to open a
    :class:`~cuda_link_spout.sender.SpoutSender` from the IPC frame's dtype
    when no explicit ``--fmt`` was supplied.

    Args:
        dtype: one of ``"uint8"``, ``"float16"``, ``"float32"``.

    Returns:
        :data:`BGRA8` for ``"uint8"``, :data:`RGBA16F` for ``"float16"``,
        :data:`RGBA32F` for ``"float32"``.

    Note:
        TD's ``cudaMemory()`` returns **BGRA** bytes for uint8 4-channel textures and
        **RGBA** bytes for float types (documented in ``CUDAMemoryShape_Class`` at
        derivative.ca/UserGuide/CUDAMemoryShape_Class).  ``uint8`` therefore
        auto-derives to ``BGRA8`` (``B8G8R8A8_UNORM``) so the Spout DXGI tag matches
        the wire bytes.  Pass ``--fmt RGBA8`` explicitly when the IPC source is a
        non-TD uint8 sender that emits RGBA-ordered bytes.

    Raises:
        ValueError: if *dtype* is not a supported auto-derivable dtype.
    """
    _dtype_to_fmt: dict[str, str] = {
        "uint8": "BGRA8",
        "float16": "RGBA16F",
        "float32": "RGBA32F",
    }
    try:
        return _FORMATS[_dtype_to_fmt[dtype]]
    except KeyError:
        supported = ", ".join(f'"{k}"' for k in _dtype_to_fmt)
        raise ValueError(
            f"Cannot derive Spout format from dtype {dtype!r}. "
            f"Supported dtypes for auto-derivation: {supported}. "
            "To use RGBA8 for a non-TD uint8 source, pass --fmt RGBA8 explicitly."
        ) from None
