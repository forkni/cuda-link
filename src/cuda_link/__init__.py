"""
cuda-link - Zero-copy GPU texture sharing between processes via CUDA IPC.

This package links TouchDesigner and Python processes using CUDA Inter-Process
Communication for zero-copy GPU texture transfer. Supports PyTorch (GPU tensors),
CuPy (GPU arrays), and NumPy (CPU arrays) output modes.
"""

from .cuda_ipc_exporter import CUDAIPCExporter
from .cuda_ipc_importer import CUPY_AVAILABLE, NUMPY_AVAILABLE, TORCH_AVAILABLE, CUDAIPCImporter
from .cuda_ipc_wrapper import CUDARuntimeAPI, get_cuda_runtime
from .nvml_observer import NVML_AVAILABLE, NVMLObserver

__version__ = "0.9.0"
__all__ = [
    "CUDAIPCExporter",
    "CUDAIPCImporter",
    "CUDARuntimeAPI",
    "get_cuda_runtime",
    "CUPY_AVAILABLE",
    "NUMPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "NVML_AVAILABLE",
    "NVMLObserver",
]
