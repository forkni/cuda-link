"""
Public value objects for the Spout bridge — specs, outcomes, and frame inputs.

Frozen dataclasses + enums, mirroring cuda-link's FrameSpec / ImportSpec / outcome
conventions so the two APIs feel the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ._format import SpoutFormat, resolve_format


@dataclass(frozen=True)
class SpoutSenderSpec:
    """Immutable description of a Spout sender channel.

    *fmt* is one of cuda_link_spout._format.SUPPORTED_FORMATS (case-insensitive).
    """

    name: str
    width: int
    height: int
    fmt: str = "RGBA8"
    device: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SpoutSenderSpec.name must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"SpoutSenderSpec dims must be positive, got {self.width}x{self.height}")
        # Validate the format eagerly so a bad name fails at construction, not mid-stream.
        resolve_format(self.fmt)

    @property
    def resolved_format(self) -> SpoutFormat:
        """The resolved :class:`SpoutFormat` for this spec's ``fmt``."""
        return resolve_format(self.fmt)


@dataclass(frozen=True)
class SpoutReceiverSpec:
    """Immutable description of a Spout receiver.

    *name* is the sender to connect to; empty string binds to the host's active sender.
    """

    name: str = ""
    device: int = 0


class SendOutcome(Enum):
    """Result of a single SpoutSender.send()."""

    SENT = auto()
    FAILED = auto()


class ReceiveOutcome(Enum):
    """Result of a single SpoutReceiver.receive()."""

    NEW_FRAME = auto()
    NO_FRAME = auto()  # connected, but the sender has not advanced since last call
    NOT_CONNECTED = auto()  # no sender of the requested name is present
    FAILED = auto()


@dataclass(frozen=True)
class ReceivedFrame:
    """A frame received from a Spout sender, copied into a cuda-link device buffer.

    ``ptr`` is a CUDA device pointer (int) to linear memory of ``width*height*fmt.bytes_per_pixel``
    bytes, valid until the next receive() call.
    """

    outcome: ReceiveOutcome
    ptr: int = 0
    width: int = 0
    height: int = 0
    fmt: SpoutFormat | None = None


@dataclass(frozen=True)
class SpoutFrame:
    """A single GPU frame to send: a CUDA device pointer + its row pitch.

    Use :meth:`from_tensor` to build one from a torch / cupy GPU tensor by duck typing,
    without importing torch or cupy here.
    """

    ptr: int
    width: int
    height: int
    pitch: int  # source row pitch in bytes (width*bpp for tightly-packed data)
    stream: int = 0  # producer CUDA stream handle (0 = default), for pre-copy ordering

    @classmethod
    def from_tensor(cls, tensor: Any, *, width: int, height: int, pitch: int, stream: int = 0) -> SpoutFrame:
        """Build a SpoutFrame from a GPU tensor exposing ``.data_ptr()`` (torch) or
        ``__cuda_array_interface__`` (cupy/numba). Geometry must be supplied by the caller
        (the bridge cannot infer channel layout from a flat pointer).
        """
        ptr = _device_ptr_of(tensor)
        return cls(ptr=ptr, width=width, height=height, pitch=pitch, stream=stream)


def _device_ptr_of(tensor: Any) -> int:
    """Extract a CUDA device pointer (int) from a torch/cupy-like object, no hard deps."""
    # torch.Tensor
    data_ptr = getattr(tensor, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    # cupy.ndarray / numba / __cuda_array_interface__ providers
    cai = getattr(tensor, "__cuda_array_interface__", None)
    if isinstance(cai, dict) and "data" in cai:
        return int(cai["data"][0])
    raise TypeError(
        "Cannot extract a CUDA device pointer: object exposes neither .data_ptr() "
        "(torch) nor __cuda_array_interface__ (cupy). Pass a SpoutFrame(ptr=...) instead."
    )
