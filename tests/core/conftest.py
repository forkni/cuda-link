"""
Fixtures shared by the fake-adapter core tests.
"""

from __future__ import annotations

import pytest

import cuda_link.importer as _imp_mod


@pytest.fixture(autouse=True)
def _no_real_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep core tests off the real CuPy runtime.

    The importer builds CuPy views whenever CuPy imports, and these tests hand it
    fake dev_ptrs. On a GPU box that happens to work; on a CI runner with CuPy but
    no driver it raises cudaErrorInsufficientDriver. Tests that exercise the CuPy
    path patch CUPY_AVAILABLE (and usually cp) back on themselves.
    """
    monkeypatch.setattr(_imp_mod, "CUPY_AVAILABLE", False)
