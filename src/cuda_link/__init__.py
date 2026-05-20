"""
cuda-link - Zero-copy GPU texture sharing between processes via CUDA IPC.

This package links TouchDesigner and Python processes using CUDA Inter-Process
Communication for zero-copy GPU texture transfer. Supports PyTorch (GPU tensors),
CuPy (GPU arrays), and NumPy (CPU arrays) output modes.
"""

from ._exporter_port import ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from ._importer_port import ImporterCudaPort, ImportOutcome, ImportPolicy, ImportResult, ImportSpec
from .cuda_ipc_exporter import CUDAIPCExporter
from .cuda_ipc_importer import CUPY_AVAILABLE, NUMPY_AVAILABLE, TORCH_AVAILABLE, CUDAIPCImporter
from .cuda_ipc_wrapper import CUDARuntimeAPI, get_cuda_runtime
from .exporter import Exporter
from .importer import Importer
from .nvml_observer import NVML_AVAILABLE, NVMLObserver
from .shm_protocol import (
    AcquireResult,
    DtypeCodec,
    Metadata,
    SHMLayout,
    SlotState,
    acquire_slot,
    publish_frame,
)

__version__ = "1.7.0"
__all__ = [
    # v1.6.0 — Exporter API
    "Exporter",
    "FrameSpec",
    "ExportPolicy",
    "GpuFrame",
    "FrameOutcome",
    # v1.7.0 — new deep Importer API
    "Importer",
    "ImportSpec",
    "ImportPolicy",
    "ImportResult",
    "ImportOutcome",
    "ImporterCudaPort",
    # deprecated — removed in v1.8.0
    "CUDAIPCExporter",
    "CUDAIPCImporter",
    "CUDARuntimeAPI",
    "get_cuda_runtime",
    "CUPY_AVAILABLE",
    "NUMPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "NVML_AVAILABLE",
    "NVMLObserver",
    "AcquireResult",
    "DtypeCodec",
    "Metadata",
    "SHMLayout",
    "SlotState",
    "acquire_slot",
    "publish_frame",
]
