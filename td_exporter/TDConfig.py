"""
TDConfig — frozen configuration dataclasses for CUDAIPCExtension.

Centralises all os.environ reads so the interaction matrix between toggles
is visible in one place and the extension body only references self._config.<field>.

textDAT name: TDConfig  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TDSenderConfig:
    """Immutable sender configuration, resolved once at init time.

    All booleans map 1-to-1 to CUDALINK_* env vars.  Defaults match
    the validated production stack described in docs/ARCHITECTURE.md.
    """

    export_sync: bool = True
    export_profile: bool = False
    export_flush_probe: bool = True
    use_graphs: bool = False
    graphs_deferred: bool = False
    stream_high_prio: bool = False
    init_pace: bool = False
    persist_stream: bool = True
    activation_barrier: bool = True
    barrier_settle_frames: int = 30
    nvml: bool = False

    @classmethod
    def from_env(cls) -> TDSenderConfig:
        """Build a config from environment variables (production path)."""
        return cls(
            export_sync=os.environ.get("CUDALINK_EXPORT_SYNC", "1") != "0",
            export_profile=os.environ.get("CUDALINK_EXPORT_PROFILE", "0") == "1",
            export_flush_probe=os.environ.get("CUDALINK_EXPORT_FLUSH_PROBE", "1") == "1",
            use_graphs=os.environ.get("CUDALINK_TD_USE_GRAPHS", "0") == "1",
            graphs_deferred=os.environ.get("CUDALINK_TD_GRAPHS_DEFERRED", "0") == "1",
            stream_high_prio=os.environ.get("CUDALINK_TD_STREAM_PRIO", "normal") == "high",
            init_pace=os.environ.get("CUDALINK_TD_INIT_PACE", "0") == "1",
            persist_stream=os.environ.get("CUDALINK_TD_PERSIST_STREAM", "1") != "0",
            activation_barrier=os.environ.get("CUDALINK_TD_ACTIVATION_BARRIER", "1") != "0",
            barrier_settle_frames=int(os.environ.get("CUDALINK_TD_BARRIER_SETTLE_FRAMES", "30")),
            nvml=os.environ.get("CUDALINK_NVML", "0") == "1",
        )

    def __post_init__(self) -> None:
        # export_flush_probe only takes effect when export_sync is False;
        # no error, just a documented no-op.
        if self.barrier_settle_frames < 0:
            raise ValueError(f"barrier_settle_frames must be >= 0, got {self.barrier_settle_frames}")


@dataclass(frozen=True)
class TDReceiverConfig:
    """Immutable receiver configuration.

    No env vars are Receiver-only at present; placeholder for future additions.
    """
