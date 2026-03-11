"""
Debug and profiling utilities for CUDA operations.

Extracted subset from StreamDiffusion project.
Requires PyTorch to be installed.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def benchmark_with_events(fn: Callable, *args: Any, warmup: int = 3, iterations: int = 10, **kwargs: Any) -> float:
    """
    GPU-accurate timing using CUDA events.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for fn
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        **kwargs: Keyword arguments for fn

    Returns:
        Average time per iteration in milliseconds

    Raises:
        ImportError: If PyTorch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "benchmark_with_events requires PyTorch. "
            "Install with: pip install cuda-link[torch] or pip install torch>=2.0"
        )

    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


class ProfileSection:
    """
    Context manager for profiling code sections with CUDA events.

    Usage:
        with ProfileSection("UNet Forward"):
            output = unet(input)
        # Prints: [PROFILE] UNet Forward: 45.3ms
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        """Initialize profiling section with a name and optional enable flag.

        Args:
            name: Label shown in the profiling log output.
            enabled: If False, profiling is a no-op. Defaults to True.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "ProfileSection requires PyTorch. Install with: pip install cuda-link[torch] or pip install torch>=2.0"
            )
        self.name = name
        self.enabled = enabled and torch.cuda.is_available()
        self.start_event = None
        self.end_event = None

    def __enter__(self) -> "ProfileSection":
        """Enter profiling context and record CUDA start event."""
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_event.record()
        return self

    def __exit__(self, *args: object) -> None:
        """Exit profiling context, synchronize GPU, and log elapsed time."""
        if self.enabled:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)
            logger.debug("[PROFILE] %s: %.1fms", self.name, elapsed)


# ---------------------------------------------------------------------------
# snoop helpers
# ---------------------------------------------------------------------------


def create_snoop_config(
    out: "str | None" = None,
    *,
    enabled: bool = True,
) -> "object | None":
    """Create a snoop.Config with timestamp column output.

    Note: call-depth tracing is configured per-decorator via ``cfg.snoop(depth=N)``,
    not at Config creation time.

    Args:
        out: Output destination — file path string, or None for stderr.
        enabled: Set to False to get a no-op config object.

    Returns:
        ``snoop.Config`` instance, or ``None`` if snoop is not installed.

    Example::

        cfg = create_snoop_config(out="debug.log")
        if cfg:
            @cfg.snoop(depth=2, watch=("self.write_idx",))
            def _initialize(self): ...
    """
    try:
        import snoop as _snoop

        kwargs: dict[str, object] = {
            "columns": "time",
            "enabled": enabled,
        }
        if out is not None:
            kwargs["out"] = out
        return _snoop.Config(**kwargs)
    except ImportError:
        logger.debug("snoop not installed; create_snoop_config() is a no-op")
        return None


def snoop_decorator(
    fn: "Callable | None" = None,
    *,
    depth: int = 1,
    watch: "tuple[str, ...]" = (),
    enabled: bool = True,
) -> "Callable":
    """Return a @snoop decorator, or a transparent no-op if snoop is unavailable.

    Designed so ``@snoop_decorator`` can be left on functions in development
    branches without breaking production (no snoop installed = zero overhead).

    Args:
        fn: When used as ``@snoop_decorator`` (no args), receives the function
            directly. When used as ``@snoop_decorator(depth=2)``, is None.
        depth: Levels of called functions to trace.
        watch: Extra expressions to evaluate and display (e.g.
            ``("self.write_idx", "slot")``).
        enabled: Pass ``False`` to get a no-op decorator regardless of
            whether snoop is installed.

    Returns:
        Decorator or decorated function.

    Example::

        @snoop_decorator(depth=2, watch=("self.write_idx", "slot"))
        def export_frame(self, gpu_ptr, size):
            ...
    """

    def _noop(f: "Callable") -> "Callable":
        return f

    try:
        import snoop as _snoop
    except ImportError:
        decorator: Callable = _noop
    else:
        if not enabled:
            decorator = _noop
        elif watch:
            decorator = _snoop(depth=depth, watch=watch)
        else:
            decorator = _snoop(depth=depth)

    if fn is not None:
        # Called as @snoop_decorator with no arguments
        return decorator(fn)
    return decorator
