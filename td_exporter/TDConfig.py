"""
TDConfig — frozen configuration dataclasses for CUDAIPCExtension.

Centralises all os.environ reads so the interaction matrix between toggles
is visible in one place and the extension body only references self._config.<field>.

textDAT name: TDConfig  (must match the importable module name inside the COMP namespace)
"""

from __future__ import annotations

from dataclasses import dataclass

from Env import env_bool, env_int, env_str


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
            export_sync=env_bool("CUDALINK_EXPORT_SYNC", default=True),
            export_profile=env_bool("CUDALINK_EXPORT_PROFILE", default=False),
            export_flush_probe=env_bool("CUDALINK_EXPORT_FLUSH_PROBE", default=True),
            use_graphs=env_bool("CUDALINK_TD_USE_GRAPHS", default=False),
            graphs_deferred=env_bool("CUDALINK_TD_GRAPHS_DEFERRED", default=False),
            stream_high_prio=env_str("CUDALINK_TD_STREAM_PRIO", default="normal") == "high",
            init_pace=env_bool("CUDALINK_TD_INIT_PACE", default=False),
            persist_stream=env_bool("CUDALINK_TD_PERSIST_STREAM", default=True),
            activation_barrier=env_bool("CUDALINK_TD_ACTIVATION_BARRIER", default=True),
            barrier_settle_frames=env_int("CUDALINK_TD_BARRIER_SETTLE_FRAMES", default=30),
            nvml=env_bool("CUDALINK_NVML", default=False),
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


@dataclass
class TDRuntimeState:
    """Mutable runtime config — single source of truth for fields that change at runtime.

    Pairs with frozen TDSenderConfig (knobs read once at engine construction and never mutate).
    Owned by CUDAIPCExtension; engines receive copies of the values at construction time.
    """

    shm_name: str
    num_slots: int
    verbose: bool

    def update(self, field: str, value: object) -> None:
        if field not in {"shm_name", "num_slots", "verbose"}:
            raise KeyError(f"TDRuntimeState has no field {field!r}")
        setattr(self, field, value)
