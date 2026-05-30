"""Per-frame profiling accumulator for cuda-link hot paths."""

from __future__ import annotations

__all__ = ["FrameProfile"]


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
