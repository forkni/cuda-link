"""Coverage tests for the reconnect/retry state machine, slot acquisition edge
cases, the doorbell wait primitive, get_frame*() dispatch, format-change
propagation, and the _reinitialize() failure path.

All GPU-free: uses fakes.make_connected_importer() (FakeCUDAAdapter-backed)
plus direct private-attribute injection, matching the pattern established in
tests/core/test_importer_cov_backends.py. Any real torch calls are
monkeypatched so results never depend on this dev machine's actual GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from fakes import make_connected_importer, make_fake_ipc_connection

from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link._importer_port import ImportOutcome, ImportPolicy, ImportSpec
from cuda_link.importer import Format, Importer, TorchBuffers
from cuda_link.shm_protocol import DtypeCodec, Metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_importer_with_policy(*, reconnect_enabled: bool, **conn_kwargs) -> Importer:
    """Build a connected Importer with full control over reconnect_enabled.

    fakes.make_connected_importer() always leaves reconnect_enabled at its
    dataclass default (True) and ImportPolicy is frozen, so exercising the
    reconnect_enabled=False paths needs this lower-level builder instead.
    """
    conn, _mock_cuda, _ = make_fake_ipc_connection(**conn_kwargs)
    fmt = Format.from_overrides((8, 8, 4), "uint8")
    spec = ImportSpec(shm_name="fake", device=0, timeout_ms=1000.0, shape=(8, 8, 4), dtype="uint8")
    policy = ImportPolicy(
        wait_spin_us=0,
        allow_pageable_fallback=True,
        reconnect_enabled=reconnect_enabled,
        wait_backend="python",
    )
    return Importer.from_connection(spec, policy, conn, fmt, cuda=FakeCUDAAdapter())


# ---------------------------------------------------------------------------
# Reconnect helpers
# ---------------------------------------------------------------------------


def test_partial_cleanup_builds_retry_state_when_none() -> None:
    """_retry is None (reconnect_enabled=False at construction) -- the guard in
    _partial_cleanup_for_reconnect() must build a fresh _RetryState on demand."""
    imp = _make_importer_with_policy(reconnect_enabled=False)
    assert imp._retry is None

    imp._partial_cleanup_for_reconnect()

    assert imp._retry is not None
    assert imp._initialized is False


def test_request_immediate_reconnect_is_noop_when_retry_is_none() -> None:
    imp = _make_importer_with_policy(reconnect_enabled=False)
    assert imp._retry is None

    imp.request_immediate_reconnect()  # must not raise

    assert imp._retry is None


def test_request_immediate_reconnect_delegates_when_retry_present() -> None:
    imp = make_connected_importer()  # reconnect_enabled=True by default
    assert imp._retry is not None
    imp._retry.retry_interval_frames = 64

    imp.request_immediate_reconnect()

    assert imp._retry.frames_since_last_retry == imp._retry.retry_interval_frames


# ---------------------------------------------------------------------------
# _try_acquire()
# ---------------------------------------------------------------------------


def test_try_acquire_swallows_shm_buffer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.importer as _imp_mod

    imp = make_connected_importer()
    monkeypatch.setattr(_imp_mod, "acquire_slot", MagicMock(side_effect=OSError("buffer gone")))

    result, outcome = imp._try_acquire()

    assert result is None
    assert outcome is ImportOutcome.NO_FRAME


def test_try_acquire_closes_importer_on_shutdown_when_reconnect_disabled() -> None:
    imp = _make_importer_with_policy(reconnect_enabled=False, write_idx=1)
    conn = imp._conn
    conn.shm_handle.buf[conn.shutdown_offset] = 1

    result, outcome = imp._try_acquire()

    assert result is None
    assert outcome is ImportOutcome.SHUTDOWN
    assert imp._conn is None  # close() -> _release_connection_state() cleared it
    assert imp._initialized is False


def test_try_acquire_keeps_provisional_flag_when_metadata_still_absent() -> None:
    """SHM metadata block stays all-zero -- Format.from_shm() returns None, so the
    provisional flag must be left set for a retry on the next NEW_FRAME."""
    imp = make_connected_importer(write_idx=1, last_write_idx=0)
    imp._format_provisional = True

    result, outcome = imp._try_acquire()

    assert outcome is ImportOutcome.NEW_FRAME
    assert result is not None
    assert imp._format_provisional is True


def test_try_acquire_resolves_provisional_format_once_metadata_appears() -> None:
    imp = make_connected_importer(shape=(4, 4, 4), dtype="uint8", write_idx=1, last_write_idx=0)
    imp._format_provisional = True
    conn = imp._conn
    kind, bits, flags = DtypeCodec.encode("uint8")
    md = Metadata(
        width=4,
        height=4,
        num_comps=4,
        format_kind=kind,
        bits_per_comp=bits,
        flags=flags,
        data_size=4 * 4 * 4 * DtypeCodec.itemsize("uint8"),
    )
    md.pack_into(conn.shm_handle.buf, conn.layout)

    result, outcome = imp._try_acquire()

    assert outcome is ImportOutcome.NEW_FRAME
    assert result is not None
    assert imp._format_provisional is False
    assert imp._format.shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# wait_for_doorbell()
# ---------------------------------------------------------------------------


def test_wait_for_doorbell_delegates_to_doorbell_wait_when_no_frame_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.importer as _imp_mod

    imp = make_connected_importer(write_idx=1, last_write_idx=1)  # cur == last_write_idx -- no early return
    imp._conn.doorbell_handle = 0xFEED
    calls: list = []
    monkeypatch.setattr(_imp_mod._doorbell, "wait", lambda h, ms: calls.append((h, ms)) or True)

    result = imp.wait_for_doorbell(250.0)

    assert result is True
    assert calls == [(0xFEED, 250)]


# ---------------------------------------------------------------------------
# get_frame() / get_frame_numpy() / get_frame_cupy() dispatch
# ---------------------------------------------------------------------------


def test_get_frame_raises_when_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.importer as _imp_mod

    imp = make_connected_importer()
    monkeypatch.setattr(_imp_mod, "TORCH_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="torch is required"):
        imp.get_frame()


def test_get_frame_builds_backend_lazily_then_reuses_and_updates_stream() -> None:
    imp = make_connected_importer(with_events=False, dev_ptr_style="mock")
    imp._torch = SimpleNamespace(tensors=[object()])
    assert imp._torch_backend is None

    result = imp.get_frame()

    assert imp._torch_backend is not None
    assert result.outcome is ImportOutcome.NEW_FRAME
    backend = imp._torch_backend

    imp.get_frame(stream=999)  # second call: reuse cached backend, update its stream

    assert imp._torch_backend is backend
    assert backend._stream == 999


def test_get_frame_numpy_raises_when_numpy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.importer as _imp_mod

    imp = make_connected_importer()
    monkeypatch.setattr(_imp_mod, "NUMPY_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="numpy is required"):
        imp.get_frame_numpy()


def test_get_frame_numpy_builds_backend_lazily() -> None:
    imp = make_connected_importer(dev_ptr_style="c_void_p")
    imp._conn.cuda = FakeCUDAAdapter()
    assert imp._numpy_backend is None

    result = imp.get_frame_numpy()

    assert imp._numpy_backend is not None
    assert result.outcome is ImportOutcome.NEW_FRAME
    if imp._numpy is not None:
        imp._numpy.close()


def test_get_frame_cupy_raises_when_cupy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import cuda_link.importer as _imp_mod

    imp = make_connected_importer()
    monkeypatch.setattr(_imp_mod, "CUPY_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="cupy is required"):
        imp.get_frame_cupy()


def test_get_frame_cupy_builds_backend_lazily_then_reuses_and_updates_stream() -> None:
    import cuda_link.importer as _imp_mod

    imp = make_connected_importer(with_events=False, dev_ptr_style="mock")
    imp._cupy = SimpleNamespace(arrays=[object()])
    mock_cp = MagicMock()

    with patch.object(_imp_mod, "cp", mock_cp), patch.object(_imp_mod, "CUPY_AVAILABLE", True):
        assert imp._cupy_backend is None

        result = imp.get_frame_cupy()

        assert imp._cupy_backend is not None
        assert result.outcome is ImportOutcome.NEW_FRAME
        backend = imp._cupy_backend

        imp.get_frame_cupy(stream=555)  # second call: reuse cached backend, update its stream

        assert imp._cupy_backend is backend
        assert backend._stream == 555


# ---------------------------------------------------------------------------
# _apply_format_change()
# ---------------------------------------------------------------------------


def test_apply_format_change_skips_numpy_close_when_numpy_already_none() -> None:
    imp = make_connected_importer(shape=(4, 4, 4), dtype="uint8", with_numpy=False)
    assert imp._numpy is None
    new_fmt = Format.from_overrides((8, 8, 4), "uint8")

    imp._apply_format_change(new_fmt)

    assert imp._format is new_fmt
    assert imp._numpy is None


def test_apply_format_change_rebuilds_torch_buffers_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    imp = make_connected_importer(shape=(4, 4, 4), dtype="uint8", dev_ptr_style="c_void_p")
    mock_tensor = MagicMock()
    monkeypatch.setattr(torch, "as_tensor", lambda wrapper, device: mock_tensor)
    imp._torch = TorchBuffers.build(imp._conn, imp._format)
    new_fmt = Format.from_overrides((4, 4, 4), "int16")  # dtype differs -> layout_differs_from() True

    imp._apply_format_change(new_fmt)

    assert imp._format is new_fmt
    assert imp._torch is not None


# ---------------------------------------------------------------------------
# _reinitialize()
# ---------------------------------------------------------------------------


def test_reinitialize_falls_back_to_partial_cleanup_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    imp = make_connected_importer()
    monkeypatch.setattr(imp, "_open_ipc_slots", MagicMock(side_effect=RuntimeError("boom")))
    spy = MagicMock()
    monkeypatch.setattr(imp, "_partial_cleanup_for_reconnect", spy)

    imp._reinitialize()  # must not raise

    spy.assert_called_once()
