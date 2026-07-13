"""Coverage tests for IPCConnection.close_ipc_handles() / close() error branches.

Builds bare IPCConnection instances directly (bypassing _open_ipc_slots entirely)
so each error path can be triggered in isolation via a MagicMock cuda/shm_handle —
no real SHM or CUDA required.
"""

from __future__ import annotations

from ctypes import c_void_p
from unittest.mock import MagicMock

from cuda_link.importer import IPCConnection
from cuda_link.shm_protocol import SHMLayout


def _make_conn(
    *,
    cuda: object,
    shm_handle: object | None = None,
    dev_ptrs: list | None = None,
    ipc_events: list | None = None,
    num_slots: int = 1,
    doorbell_handle: object | None = None,
) -> IPCConnection:
    layout = SHMLayout(num_slots)
    return IPCConnection(
        cuda=cuda,
        shm_handle=shm_handle,
        ipc_version=1,
        num_slots=num_slots,
        ipc_handles=[None] * num_slots,
        dev_ptrs=dev_ptrs if dev_ptrs is not None else [None] * num_slots,
        ipc_events=ipc_events if ipc_events is not None else [None] * num_slots,
        layout=layout,
        shutdown_offset=layout.shutdown_offset,
        timestamp_offset=layout.timestamp_offset,
        doorbell_handle=doorbell_handle,
    )


def test_close_ipc_handles_skips_none_dev_ptrs_and_events() -> None:
    """None entries in dev_ptrs/ipc_events are skipped -- no cuda calls made for them."""
    cuda = MagicMock()
    conn = _make_conn(cuda=cuda, dev_ptrs=[None], ipc_events=[None])

    conn.close_ipc_handles()

    cuda.ipc_close_mem_handle.assert_not_called()
    cuda.destroy_event.assert_not_called()


def test_close_ipc_handles_logs_and_continues_on_mem_handle_error() -> None:
    """RuntimeError from ipc_close_mem_handle is swallowed; slot is still cleared."""
    cuda = MagicMock()
    cuda.ipc_close_mem_handle.side_effect = RuntimeError("boom")
    conn = _make_conn(cuda=cuda, dev_ptrs=[c_void_p(0x1000)])

    conn.close_ipc_handles()  # must not raise

    assert conn.dev_ptrs[0] is None


def test_close_ipc_handles_logs_and_continues_on_event_destroy_error() -> None:
    """OSError from destroy_event is swallowed; slot is still cleared."""
    cuda = MagicMock()
    cuda.destroy_event.side_effect = OSError("boom")
    conn = _make_conn(cuda=cuda, ipc_events=[MagicMock()])

    conn.close_ipc_handles()  # must not raise

    assert conn.ipc_events[0] is None


def test_close_logs_and_swallows_shm_close_error() -> None:
    """OSError from shm_handle.close() is swallowed; shm_handle is still cleared."""
    cuda = MagicMock()
    shm_handle = MagicMock()
    shm_handle.close.side_effect = OSError("boom")
    conn = _make_conn(cuda=cuda, shm_handle=shm_handle)

    conn.close()  # must not raise

    assert conn.shm_handle is None


def test_close_ipc_handles_closes_doorbell_handle(monkeypatch) -> None:
    """doorbell_handle is closed via _doorbell.close() and reset to None."""
    import cuda_link.importer as _imp_mod

    closed: list[object] = []
    monkeypatch.setattr(_imp_mod._doorbell, "close", lambda h: closed.append(h))

    cuda = MagicMock()
    conn = _make_conn(cuda=cuda, doorbell_handle=0xDEAD)

    conn.close_ipc_handles()

    assert closed == [0xDEAD]
    assert conn.doorbell_handle is None
