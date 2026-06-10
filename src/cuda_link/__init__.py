"""
cuda-link - Zero-copy GPU texture sharing between processes via CUDA IPC.

This package links TouchDesigner and Python processes using CUDA Inter-Process
Communication for zero-copy GPU texture transfer. Supports PyTorch (GPU tensors),
CuPy (GPU arrays), and NumPy (CPU arrays) output modes.
"""

from ._cuda_adapters import CTypesCUDAAdapter, FakeCUDAAdapter
from ._exporter_port import CudaPort, ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from ._importer_port import ImporterCudaPort, ImportOutcome, ImportPolicy, ImportResult, ImportSpec
from .cuda_ipc_importer import CUPY_AVAILABLE, NUMPY_AVAILABLE, TORCH_AVAILABLE, CUDAIPCImporter
from .cuda_ipc_wrapper import CUDARuntimeAPI, get_cuda_runtime
from .exporter import Exporter
from .importer import Format, Importer, IPCConnection
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

__version__ = "1.10.0"
__all__ = [
    # Exporter API (v1.5.0+)
    "Exporter",
    "FrameSpec",
    "ExportPolicy",
    "GpuFrame",
    "FrameOutcome",
    "CudaPort",
    # Importer API (v1.5.x)
    "Importer",
    "IPCConnection",
    "Format",
    "ImportSpec",
    "ImportPolicy",
    "ImportResult",
    "ImportOutcome",
    "ImporterCudaPort",
    # Adapters (satisfies both CudaPort and ImporterCudaPort)
    "CTypesCUDAAdapter",
    "FakeCUDAAdapter",
    # deprecated — CUDAIPCImporter removed in v1.8.0; migrate to Importer.open(ImportSpec(...))
    "CUDAIPCImporter",
    # infrastructure / low-level symbols
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
