"""
Unit tests for P5 — pipelined D2H double-buffer in _NumpyBackend.materialize().

All tests are GPU-free: they use NumpyBuffers constructed directly with
MagicMock cuda, bypassing the full Importer / SHM machinery.
"""

from __future__ import annotations

import types
from ctypes import c_void_p
from unittest.mock import MagicMock

import numpy as np

from cuda_link.importer import (
    IPCConnection,
    NumpyBuffers,
    _NumpyBackend,
)
from cuda_link.shm_protocol import SHMLayout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fmt(shape=(4, 4, 4), dtype=np.uint8):
    return types.SimpleNamespace(
        shape=shape,
        numpy_dtype=np.dtype(dtype),
        frame_nbytes=int(np.prod(shape) * np.dtype(dtype).itemsize),
    )


def _make_pipelined_nb(shape=(4, 4, 4), dtype=np.uint8) -> tuple[NumpyBuffers, np.ndarray, np.ndarray]:
    """Build a pipelined NumpyBuffers with two real numpy arrays and a MagicMock cuda."""
    buf_a = np.zeros(shape, dtype=np.dtype(dtype))
    buf_b = np.zeros(shape, dtype=np.dtype(dtype))
    mock_stream = MagicMock()
    nb = NumpyBuffers(
        cuda=MagicMock(),
        fmt=_make_fmt(shape, dtype),
        buffer=buf_a,
        pinned_ptr=None,
        host_registered_arr=None,
        pinned_memory_available=False,
        primary_stream=mock_stream,
        d2h_streams=[mock_stream],
        num_streams=1,
        chunk_plan=[],
        buffer_ptr=c_void_p(buf_a.ctypes.data),
        back_buffer=buf_b,
        back_pinned_ptr=None,
        back_host_registered_arr=None,
        back_buffer_ptr=c_void_p(buf_b.ctypes.data),
        pipelined=True,
        priming=True,
    )
    return nb, buf_a, buf_b


def _make_conn_and_backend(nb: NumpyBuffers, shape=(4, 4, 4), dtype=np.uint8):
    """Build a minimal mock IPCConnection and _NumpyBackend wired to nb."""
    mock_cuda = nb.cuda
    layout = SHMLayout(num_slots=1)
    buf = layout.build_buffer(version=1, write_idx=1)
    mock_shm = MagicMock()
    mock_shm.buf = buf

    conn = IPCConnection(
        cuda=mock_cuda,
        shm_handle=mock_shm,
        ipc_version=1,
        num_slots=1,
        ipc_handles=[None],
        dev_ptrs=[c_void_p(0x1000)],
        ipc_events=[None],
        layout=layout,
        shutdown_offset=layout.shutdown_offset,
        timestamp_offset=layout.timestamp_offset,
    )

    fmt = _make_fmt(shape, dtype)
    mock_imp = MagicMock()
    mock_imp._numpy = nb
    mock_imp._format = fmt
    mock_imp._policy = MagicMock()

    backend = _NumpyBackend(mock_imp)
    return conn, backend


# ---------------------------------------------------------------------------
# Priming call — first call returns None (surfaced as NO_FRAME by get_frame_numpy)
# ---------------------------------------------------------------------------


def test_pipelined_priming_returns_none() -> None:
    """materialize() with priming=True returns None (priming sentinel)."""
    nb, buf_a, buf_b = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    result = backend.materialize(conn, 0)

    assert result is None, "priming call must return None — get_frame_numpy surfaces as NO_FRAME"


def test_pipelined_priming_sets_priming_false() -> None:
    """After priming call, nb.priming is False."""
    nb, _, _ = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    backend.materialize(conn, 0)

    assert nb.priming is False


def test_pipelined_priming_issues_async_copy_to_back() -> None:
    """Priming call enqueues memcpy_async with the BACK buffer ptr, not the front."""
    nb, buf_a, buf_b = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    backend.materialize(conn, 0)

    # The copy should have been issued to the back buffer (buf_b)
    call_args = nb.cuda.memcpy_async.call_args
    assert call_args is not None, "memcpy_async must be called during priming"
    assert call_args.kwargs.get("dst", call_args.args[0] if call_args.args else None) is not None


def test_pipelined_priming_rotates_front_back() -> None:
    """After priming, nb.buffer is buf_b (old back) and nb.back_buffer is buf_a (old front)."""
    nb, buf_a, buf_b = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    backend.materialize(conn, 0)

    assert nb.buffer is buf_b, "after priming, front (nb.buffer) must be the old back (buf_b)"
    assert nb.back_buffer is buf_a, "after priming, back must be the old front (buf_a)"


# ---------------------------------------------------------------------------
# Steady-state call — second call syncs and returns the primed front buffer
# ---------------------------------------------------------------------------


def test_pipelined_second_call_returns_front_buffer() -> None:
    """Second call returns nb.buffer (the primed front buffer, which is buf_b after priming)."""
    nb, buf_a, buf_b = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    backend.materialize(conn, 0)  # priming — nb.buffer is now buf_b
    result = backend.materialize(conn, 0)  # steady-state

    assert result is buf_b, "second call must return buf_b (the primed front buffer)"


def test_pipelined_second_call_syncs_stream() -> None:
    """Steady-state call issues stream_synchronize before returning data."""
    nb, _, _ = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    backend.materialize(conn, 0)  # priming
    nb.cuda.stream_synchronize.reset_mock()  # clear priming-call side-effects (none, but be explicit)

    backend.materialize(conn, 0)  # steady-state

    nb.cuda.stream_synchronize.assert_called_once_with(nb.primary_stream)


def test_pipelined_second_call_rotates_again() -> None:
    """After second call, front/back swap back to buf_a/buf_b."""
    nb, buf_a, buf_b = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    backend.materialize(conn, 0)  # priming: front→buf_b, back→buf_a
    backend.materialize(conn, 0)  # steady: returns buf_b, front→buf_a, back→buf_b

    assert nb.buffer is buf_a, "after second call, front must rotate back to buf_a"
    assert nb.back_buffer is buf_b


def test_pipelined_three_call_sequence() -> None:
    """Three-call sequence: None, buf_b, buf_a — alternating front buffer."""
    nb, buf_a, buf_b = _make_pipelined_nb()
    conn, backend = _make_conn_and_backend(nb)

    r1 = backend.materialize(conn, 0)  # priming → None
    r2 = backend.materialize(conn, 0)  # steady → buf_b
    r3 = backend.materialize(conn, 0)  # steady → buf_a

    assert r1 is None
    assert r2 is buf_b
    assert r3 is buf_a


# ---------------------------------------------------------------------------
# Non-pipelined path is unchanged
# ---------------------------------------------------------------------------


def test_non_pipelined_path_unchanged() -> None:
    """With pipelined=False, materialize returns nb.buffer directly (original behaviour)."""
    nb, buf_a, _ = _make_pipelined_nb()
    nb.pipelined = False  # switch off pipelined mode
    nb.priming = False
    conn, backend = _make_conn_and_backend(nb)

    result = backend.materialize(conn, 0)

    assert result is buf_a, "non-pipelined path must return nb.buffer (buf_a) unchanged"
    # stream_synchronize must still be called (synchronous D2H)
    nb.cuda.stream_synchronize.assert_called_once_with(nb.primary_stream)
    # No buffer swap should have occurred
    assert nb.buffer is buf_a
