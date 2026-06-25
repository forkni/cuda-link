"""
cuda-link-spout — bridge cuda-link's CUDA-IPC GPU frames to/from Spout.

Optional, separately-distributed native add-on (relaxes cuda-link's pure-Python
rule per docs/competitive/spout-bridge-design.md §8). The core ``cuda_link`` wheel
is unaffected; install this only to reach Spout-speaking apps (Resolume, Unreal,
OBS, Notch, Unity, TouchDesigner, …).

Public API::

    from cuda_link_spout import (
        SpoutSender, SpoutSenderSpec,
        SpoutReceiver, SpoutReceiverSpec,
        SendOutcome, ReceiveOutcome, ReceivedFrame, SpoutFrame,
        SUPPORTED_FORMATS,
    )
"""

from __future__ import annotations

from ._backend import FakeSpoutBackend, NativeReceiveResult, SpoutBackend
from ._format import SUPPORTED_FORMATS, SpoutFormat, format_from_dxgi, resolve_format
from ._types import (
    ReceivedFrame,
    ReceiveOutcome,
    SendOutcome,
    SpoutFrame,
    SpoutReceiverSpec,
    SpoutSenderSpec,
)
from .receiver import SpoutReceiver
from .sender import SpoutSender

__version__ = "0.1.0"
__all__ = [
    # high-level API
    "SpoutSender",
    "SpoutSenderSpec",
    "SpoutReceiver",
    "SpoutReceiverSpec",
    "SendOutcome",
    "ReceiveOutcome",
    "ReceivedFrame",
    "SpoutFrame",
    # formats
    "SUPPORTED_FORMATS",
    "SpoutFormat",
    "resolve_format",
    "format_from_dxgi",
    # backend seam (advanced / testing)
    "SpoutBackend",
    "FakeSpoutBackend",
    "NativeReceiveResult",
]
