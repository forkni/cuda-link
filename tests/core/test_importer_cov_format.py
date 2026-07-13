"""Coverage tests for _numpy_dtype_for(), Format.from_shm(), and Format.from_overrides().

All CPU-only, no GPU/CUDA required. ml_dtypes is not installed in this
environment, so the bfloat16-success branch is exercised by injecting a fake
ml_dtypes module into sys.modules rather than requiring the real package.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import cuda_link.importer as _imp_mod
from cuda_link.importer import Format, _numpy_dtype_for


def test_numpy_dtype_for_returns_none_when_numpy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_imp_mod, "NUMPY_AVAILABLE", False)

    assert _numpy_dtype_for("uint8") is None


def test_numpy_dtype_for_bfloat16_uses_ml_dtypes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a successful `import ml_dtypes` without requiring the real package."""
    fake_ml_dtypes = types.ModuleType("ml_dtypes")
    fake_ml_dtypes.bfloat16 = np.float16  # stand-in scalar type, just needs to be np.dtype()-able
    monkeypatch.setitem(sys.modules, "ml_dtypes", fake_ml_dtypes)

    result = _numpy_dtype_for("bfloat16")

    assert result == np.dtype(np.float16)


def test_numpy_dtype_for_bfloat16_returns_none_without_ml_dtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    """ml_dtypes genuinely absent in this environment -- confirms the ImportError fallback."""
    monkeypatch.setitem(sys.modules, "ml_dtypes", None)

    assert _numpy_dtype_for("bfloat16") is None


def test_format_from_shm_returns_none_on_truncated_buffer() -> None:
    """Buffer too short for Metadata.read_from -> struct.error swallowed -> None."""
    tiny_buf = bytearray(4)

    assert Format.from_shm(tiny_buf, num_slots=1) is None


def test_format_from_overrides_unknown_dtype_falls_back_to_itemsize_4(monkeypatch: pytest.MonkeyPatch) -> None:
    """DtypeCodec.itemsize(KeyError) -> itemsize=4 fallback.

    Uses a real, resolvable dtype ("uint8") so _numpy_dtype_for() (called right
    after the itemsize fallback, on the same dtype_str) doesn't itself raise --
    only DtypeCodec.itemsize is forced to fail here, isolating the branch under test.
    """
    from cuda_link.shm_protocol import DtypeCodec

    def _raise(dtype_str: str) -> int:
        raise KeyError(dtype_str)

    monkeypatch.setattr(DtypeCodec, "itemsize", staticmethod(_raise))

    fmt = Format.from_overrides((2, 3, 4), "uint8")

    assert fmt.frame_nbytes == 2 * 3 * 4 * 4
