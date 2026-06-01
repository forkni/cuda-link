"""
CUDAIPCImporter — deprecation shim for the pre-v1.5.x API.

New code should use Importer.open() from cuda_link.importer.
This shim keeps existing callers working until removal in v1.8.

Migration guide: docs/MIGRATION_v1.5.md
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from ._importer_port import ImportOutcome, ImportPolicy, ImportResult, ImportSpec  # noqa: F401
from .cuda_ipc_wrapper import CUDARuntimeAPI, get_cuda_runtime  # noqa: F401
from .cuda_runtime_types import cudaIpcEventHandle_t, cudaIpcMemHandle_t  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports required by existing callers
# ---------------------------------------------------------------------------
from .importer import (  # noqa: F401
    CUPY_AVAILABLE,
    NUMPY_AVAILABLE,
    TORCH_AVAILABLE,
    CupyBuffers,
    Format,
    Importer,  # noqa: F401
    IPCConnection,
    NumpyBuffers,
    TorchBuffers,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deprecation helpers
# ---------------------------------------------------------------------------

_deprecation_warned = False


def _warn_once() -> None:
    global _deprecation_warned
    if not _deprecation_warned:
        warnings.warn(
            "CUDAIPCImporter is deprecated and will be removed in v1.8. "
            "Migrate to Importer.open() from cuda_link.importer. "
            "See docs/MIGRATION_v1.5.md for migration guides.",
            DeprecationWarning,
            stacklevel=3,
        )
        _deprecation_warned = True


# ---------------------------------------------------------------------------
# Legacy shim class
# ---------------------------------------------------------------------------


class CUDAIPCImporter:
    """Deprecated. Use Importer.open() instead.

    This class preserves the pre-v1.5.x API surface:
        CUDAIPCImporter(shm_name, shape, dtype, ...)
        importer.connect()
        CUDAIPCImporter.from_connected(shm_name, ...)
        with CUDAIPCImporter(...) as imp: ...
        imp.get_frame() → torch.Tensor | None
        imp.get_frame_numpy() → np.ndarray | None
        imp.get_frame_cupy() → cp.ndarray | None

    All get_frame* methods return the frame payload directly (not ImportResult).
    None is returned for any non-NEW_FRAME outcome.
    """

    def __init__(
        self,
        shm_name: str = "cudalink_ipc_TD>>Python",
        shape: tuple[int, int, int] | None = None,
        dtype: str | None = None,
        debug: bool = False,
        timeout_ms: float = 5000.0,
        device: int = 0,
    ) -> None:
        # Store construction args; actual connection happens in connect()
        self.shm_name = shm_name
        self.shape = shape
        self.dtype = dtype
        self.debug = debug
        self.timeout_ms = timeout_ms
        self.device = device
        self._importer: Importer | None = None

    @classmethod
    def from_connected(cls, shm_name: str = "cudalink_ipc_TD>>Python", **kwargs) -> CUDAIPCImporter:
        """Deprecated. Use Importer.open() instead."""
        _warn_once()
        imp = cls(shm_name=shm_name, **kwargs)
        imp.connect()
        return imp

    def connect(self) -> None:
        """Open SHM + IPC handles. Idempotent."""
        _warn_once()
        if self._importer is not None:
            return
        policy = ImportPolicy.from_env()
        spec = ImportSpec(
            shm_name=self.shm_name,
            device=self.device,
            shape=self.shape,
            dtype=self.dtype,
            timeout_ms=self.timeout_ms,
        )
        self._importer = Importer.open(spec, policy=policy)
        # Keep shape/dtype in sync after auto-detection
        if self._importer._format is not None:
            self.shape = self._importer._format.shape
            self.dtype = self._importer._format.dtype_str

    def get_frame(self, stream: object | None = None) -> object | None:
        """Deprecated. Returns tensor or None (use Importer.get_frame() for ImportResult)."""
        if self._importer is None:
            logger.warning("Not initialized — call connect() first")
            return None
        result = self._importer.get_frame(stream=stream)
        return result.frame if result.outcome is ImportOutcome.NEW_FRAME else None

    def get_frame_numpy(self) -> object | None:
        """Deprecated. Returns ndarray or None (use Importer.get_frame_numpy() for ImportResult)."""
        if self._importer is None:
            logger.warning("Not initialized — call connect() first")
            return None
        result = self._importer.get_frame_numpy()
        return result.frame if result.outcome is ImportOutcome.NEW_FRAME else None

    def get_frame_cupy(self, stream: object | None = None) -> object | None:
        """Deprecated. Returns cp.ndarray or None (use Importer.get_frame_cupy() for ImportResult)."""
        if self._importer is None:
            logger.warning("Not initialized — call connect() first")
            return None
        result = self._importer.get_frame_cupy(stream=stream)
        return result.frame if result.outcome is ImportOutcome.NEW_FRAME else None

    def cleanup(self) -> None:
        """Release resources. Idempotent."""
        if self._importer is not None:
            self._importer.close()
            self._importer = None

    def is_ready(self) -> bool:
        return self._importer is not None and self._importer.is_ready()

    def get_stats(self) -> dict:
        if self._importer is None:
            return {"initialized": False}
        return self._importer.get_stats()

    def attach_nvml_observer(self, observer: object) -> None:
        """Attach an NVMLObserver for GPU telemetry in get_stats()."""
        self._nvml_observer = observer
        if self._importer is not None:
            self._importer.attach_nvml_observer(observer)

    @property
    def frame_count(self) -> int:
        return self._importer.frame_count if self._importer is not None else 0

    @property
    def last_latency(self) -> float:
        return self._importer.last_latency if self._importer is not None else 0.0

    @property
    def _initialized(self) -> bool:
        return self._importer is not None and self._importer._initialized

    def __del__(self) -> None:
        self.cleanup()

    def __enter__(self) -> CUDAIPCImporter:
        if self._importer is None:
            self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.cleanup()
