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
        """Initialize profiling section with a name and optional enable flag."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "ProfileSection requires PyTorch. Install with: pip install cuda-link[torch] or pip install torch>=2.0"
            )
        self.name = name
        self.enabled = enabled and torch.cuda.is_available()
        self.start_event = None
        self.end_event = None

    def __enter__(self) -> "ProfileSection":
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_event.record()
        return self

    def __exit__(self, *args: object) -> None:
        if self.enabled:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)
            logger.debug(f"[PROFILE] {self.name}: {elapsed:.1f}ms")
