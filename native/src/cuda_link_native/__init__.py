"""
cuda-link-native — native notification-wait accelerator for cuda-link's Importer.

Optional, separately-distributed native add-on (relaxes cuda-link's pure-Python
rule per ADR-0006, same escape hatch cuda-link-spout exercises). The core
``cuda_link`` wheel is unaffected when this package is absent: ``Importer``
falls back to its existing Python spin-then-sleep wait automatically
(``ImportPolicy.wait_backend="auto"``).

Public API::

    from cuda_link_native import WaitBackend, FakeWaitBackend, WaitResult, WaitStatus
"""

from __future__ import annotations

from ._backend import FakeWaitBackend, WaitBackend, WaitResult, WaitStatus

__version__ = "0.2.0"
__all__ = [
    "WaitBackend",
    "FakeWaitBackend",
    "WaitResult",
    "WaitStatus",
]
