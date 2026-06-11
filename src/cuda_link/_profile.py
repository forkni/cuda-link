"""Per-frame profiling accumulator for cuda-link hot paths."""

from __future__ import annotations

__all__ = ["FrameProfile", "ReportWindow"]


class FrameProfile:
    """Accumulates microsecond-level timing per named region.

    Pass a fixed tuple of region names at construction; the internal dict is
    pre-populated so .record() is a single dict lookup + float add on the hot
    path — no setdefault, no allocation.

    *ptr_cache_miss* and similar count-based regions store dimensionless counts
    (each hit calls record(region, 1.0)). report()/avg() present them as
    average counts per frame, which is the useful diagnostic unit.
    """

    __slots__ = ("regions", "_totals")

    def __init__(self, regions: tuple[str, ...]) -> None:
        self.regions: tuple[str, ...] = regions
        self._totals: dict[str, float] = dict.fromkeys(regions, 0.0)

    def record(self, region: str, us: float) -> None:
        """Accumulate *us* microseconds (or a count) for *region*."""
        self._totals[region] += us

    def avg(self, region: str, n: int) -> float:
        """Per-frame average for *region* over *n* frames."""
        return self._totals[region] / n if n > 0 else 0.0

    def report(self, n: int) -> str:
        """Space-separated 'region=N.N' averages over *n* frames."""
        if n <= 0:
            return ""
        return " ".join(f"{r}={self._totals[r] / n:.1f}" for r in self.regions)


class ReportWindow:
    """Windowed-FPS bookkeeping for periodic status lines.

    Shared by the TD sender/receiver engines (and usable by example scripts):
    tracks a *session baseline* — seeded via start() at the first frame after
    (re)connect so one-time IPC-open latency doesn't dilute the first window,
    and anchored to the lifetime frame counter so reconnects don't inflate it
    (frame counts are never reset across sessions).

    fps() returns frames/second since the previous report (the first window is
    seeded from the session baseline) and advances the window.
    """

    __slots__ = ("start_t", "start_frame", "last_t", "last_frame")

    def __init__(self) -> None:
        self.reset()

    @property
    def started(self) -> bool:
        """True once start() has seeded the session baseline."""
        return self.start_t != 0.0

    def start(self, now: float, frame: int) -> None:
        """Seed the session baseline at the first frame after (re)connect."""
        self.start_t = now
        self.start_frame = frame

    def fps(self, now: float, frame: int) -> float:
        """Windowed FPS since the previous report; advances the window."""
        if self.last_t == 0.0:
            self.last_t = self.start_t
            self.last_frame = self.start_frame
        dt = now - self.last_t
        fps = (frame - self.last_frame) / dt if dt > 0 else 0.0
        self.last_t = now
        self.last_frame = frame
        return fps

    def reset(self) -> None:
        """Clear all state so the next session starts with a fresh window."""
        self.start_t = 0.0
        self.start_frame = 0
        self.last_t = 0.0
        self.last_frame = 0
