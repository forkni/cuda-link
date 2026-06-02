"""
Tests for NumpyBuffers.close() — verifying the UAF guard (item 5 of ctypes sweep).

Background
----------
NumpyBuffers.buffer aliases CUDA-pinned host memory owned by pinned_ptr.  After
close() frees that allocation via cuda.free_host(), any subsequent read of buffer
would silently touch freed GPU memory (use-after-free).  The guard

    self.buffer = None   # importer.py, inside the pinned_ptr-is-not-None branch

makes such accesses fail loudly (AttributeError) instead.

These tests are GPU-free: NumpyBuffers is constructed directly (no build() call),
with a MagicMock cuda adapter and a real small numpy array as the buffer.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from cuda_link.importer import NumpyBuffers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_UINT8 = np.dtype("uint8")  # module-level singleton avoids B008 in defaults


def _make_fmt(shape=(4, 4, 4), dtype=_UINT8):
    """Minimal Format-like namespace for needs_rebuild comparisons."""
    return types.SimpleNamespace(shape=shape, numpy_dtype=dtype)


def _make_nb(
    *,
    pinned_ptr=None,
    buffer_shape=(4, 4, 4),
    buffer_dtype=_UINT8,
    host_registered_arr=None,
    primary_stream=None,
    d2h_streams=None,
    num_streams=1,
    chunk_plan=None,
) -> NumpyBuffers:
    """Build a NumpyBuffers with a MagicMock cuda adapter and real numpy buffer."""
    return NumpyBuffers(
        cuda=MagicMock(),
        fmt=_make_fmt(shape=buffer_shape, dtype=buffer_dtype),
        buffer=np.zeros(buffer_shape, dtype=buffer_dtype),
        pinned_ptr=pinned_ptr,
        host_registered_arr=host_registered_arr,
        pinned_memory_available=(pinned_ptr is not None),
        primary_stream=primary_stream if primary_stream is not None else MagicMock(),
        d2h_streams=d2h_streams if d2h_streams is not None else [MagicMock()],
        num_streams=num_streams,
        chunk_plan=chunk_plan if chunk_plan is not None else [],
    )


# ---------------------------------------------------------------------------
# Test 1: pinned path — buffer and ptr are cleared, streams are destroyed
# ---------------------------------------------------------------------------


def test_close_nulls_buffer_and_ptr_on_pinned_path() -> None:
    """close() on the pinned path must null buffer (UAF guard) and pinned_ptr."""
    extra_stream = MagicMock()
    primary = MagicMock()
    sentinel_ptr = MagicMock()  # non-None pinned_ptr triggers the UAF guard branch

    nb = _make_nb(
        pinned_ptr=sentinel_ptr,
        primary_stream=primary,
        d2h_streams=[primary, extra_stream],  # index 0 skipped; index 1 destroyed
        num_streams=2,
    )
    original_buffer = nb.buffer
    assert original_buffer is not None

    nb.close()

    # Core UAF guard assertions
    assert nb.buffer is None, "buffer must be nulled after freeing pinned allocation"
    assert nb.pinned_ptr is None, "pinned_ptr must be cleared after free_host"

    # free_host called with the original ptr
    nb.cuda.free_host.assert_called_once_with(sentinel_ptr)

    # Streams torn down correctly
    assert nb.d2h_streams == [], "d2h_streams must be cleared"
    assert nb.primary_stream is None, "primary_stream must be cleared"

    # destroy_stream: once for extra_stream (index 1), once for primary
    assert nb.cuda.destroy_stream.call_count == 2
    nb.cuda.destroy_stream.assert_any_call(extra_stream)
    nb.cuda.destroy_stream.assert_any_call(primary)


# ---------------------------------------------------------------------------
# Test 2: needs_rebuild raises AttributeError after close (the "fails loudly" proof)
# ---------------------------------------------------------------------------


def test_needs_rebuild_raises_after_close() -> None:
    """needs_rebuild() after close() (pinned path) raises AttributeError.

    This is the core proof that the UAF guard works: instead of silently
    reading self.buffer.shape on freed memory, callers get AttributeError.
    """
    fmt = _make_fmt(shape=(4, 4, 4))
    nb = _make_nb(pinned_ptr=MagicMock())

    nb.close()

    # buffer is now None; .shape access raises AttributeError
    with pytest.raises(AttributeError):
        nb.needs_rebuild(fmt)


# ---------------------------------------------------------------------------
# Test 3: pageable fallback — buffer is NOT cleared (not a pinned alias)
# ---------------------------------------------------------------------------


def test_close_pageable_keeps_buffer() -> None:
    """close() with pinned_ptr=None (pageable fallback) must NOT null buffer.

    Pageable buffers are owned by Python (np.empty), not by CUDA pinned memory.
    Clearing them would be an over-broad future regression; this test guards it.
    """
    nb = _make_nb(pinned_ptr=None)  # pageable path
    original_buffer = nb.buffer

    nb.close()

    assert nb.buffer is original_buffer, "buffer must NOT be cleared on the pageable path (no UAF risk there)"
    # free_host must NOT have been called
    nb.cuda.free_host.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: idempotency — calling close() twice must not raise
# ---------------------------------------------------------------------------


def test_close_is_idempotent() -> None:
    """close() is safe to call multiple times (all branches are None-guarded)."""
    nb = _make_nb(pinned_ptr=MagicMock())

    nb.close()  # first call — frees everything
    nb.close()  # second call — all fields are None / lists are empty; must not raise


# ---------------------------------------------------------------------------
# Test 5: host_registered_arr path — host_unregister is called and field nulled
# ---------------------------------------------------------------------------


def test_close_unregisters_host_array() -> None:
    """close() on the host_registered_arr path calls host_unregister and nulls the field."""
    # Use a real numpy array so .ctypes.data returns a valid integer address
    host_arr = np.zeros((4, 4, 4), dtype=np.uint8)
    expected_addr = host_arr.ctypes.data

    nb = _make_nb(
        pinned_ptr=None,  # no pinned alloc; buffer stays alive
        host_registered_arr=host_arr,
    )

    nb.close()

    nb.cuda.host_unregister.assert_called_once_with(expected_addr)
    assert nb.host_registered_arr is None, "host_registered_arr must be nulled after unregister"
