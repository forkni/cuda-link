"""
Backwards-compat re-export shim — use _cuda_adapters directly in new code.

CTypesCudaAdapter and FakeCudaAdapter now live in _cuda_adapters.py, where they
satisfy both CudaPort (exporter) and ImporterCudaPort (importer) structurally.
This module re-exports them unchanged so existing imports keep working.
"""

from ._cuda_adapters import (  # noqa: F401
    CTypesCudaAdapter,
    FakeCudaAdapter,
    _FakeHandle,
    _FakeIpcHandle,
    _FakePointerAttributes,
)

__all__ = [
    "CTypesCudaAdapter",
    "FakeCudaAdapter",
    "_FakeHandle",
    "_FakeIpcHandle",
    "_FakePointerAttributes",
]
