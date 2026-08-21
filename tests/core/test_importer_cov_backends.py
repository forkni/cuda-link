"""Coverage tests for _event_to_int() and the three _FrameBackend implementations:
_TorchBackend, _NumpyBackend, _CupyBackend.

All GPU-availability-gated branches (torch.cuda.is_available(), torch.cuda.current_stream())
are deterministically monkeypatched -- never relying on this dev machine's real GPU, per the
project's no-GPU CI requirement. CuPy-touching code uses unittest.mock.patch("cuda_link.importer.cp", ...)
matching the established pattern in tests/core/test_importer.py.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from fakes import make_connected_importer

from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.importer import NumpyBuffers, _CupyBackend, _event_to_int, _NumpyBackend, _TorchBackend

# ---------------------------------------------------------------------------
# _event_to_int()
# ---------------------------------------------------------------------------


def test_event_to_int_from_ctypes_value_attr() -> None:
    evt = ctypes.c_uint64(0xDEADBEEF)
    assert _event_to_int(evt) == 0xDEADBEEF


def test_event_to_int_from_bytes() -> None:
    evt = (0x1234).to_bytes(8, "little")
    assert _event_to_int(evt) == 0x1234


def test_event_to_int_from_bytearray() -> None:
    evt = bytearray((0xABCD).to_bytes(8, "little"))
    assert _event_to_int(evt) == 0xABCD


def test_event_to_int_from_plain_int() -> None:
    assert _event_to_int(42) == 42


def test_event_to_int_from_object_requiring_int_fallback() -> None:
    class _Weird:
        def __int__(self) -> int:
            return 777

    assert _event_to_int(_Weird()) == 777


# ---------------------------------------------------------------------------
# _TorchBackend.wait() / materialize()
# ---------------------------------------------------------------------------


def test_torch_backend_wait_explicit_stream_with_event_calls_stream_wait_event() -> None:
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    conn = imp._conn
    backend = _TorchBackend(imp, stream=123)

    waited = backend.wait(conn, 0)

    assert waited == 0.0
    conn.cuda.stream_wait_event.assert_called_once_with(123, conn.ipc_events[0], 0)


def test_torch_backend_wait_explicit_stream_without_event_falls_back_to_wait_for_slot() -> None:
    imp = make_connected_importer(with_events=False, dev_ptr_style="mock")
    conn = imp._conn
    backend = _TorchBackend(imp, stream=123)
    spy = MagicMock(return_value=99.0)
    imp._wait_for_slot = spy

    waited = backend.wait(conn, 0)

    assert waited == 0.0
    spy.assert_called_once_with(0)


def test_torch_backend_wait_no_stream_no_gpu_wait_falls_back_to_cpu_wait() -> None:
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    conn = imp._conn
    backend = _TorchBackend(imp, stream=None)
    assert imp._policy.torch_gpu_wait is False
    assert imp._gpu_wait_active is False

    waited = backend.wait(conn, 0)

    # Falls through to _imp._wait_for_slot() -- conn.cuda.query_event returns True
    # (pre-wired by make_fake_ipc_connection), so this resolves immediately.
    assert isinstance(waited, float)
    conn.cuda.stream_wait_event.assert_not_called()


def test_torch_backend_wait_gpu_wait_active_with_event_uses_current_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    imp._gpu_wait_active = True
    conn = imp._conn
    backend = _TorchBackend(imp, stream=None)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake_stream = SimpleNamespace(cuda_stream=0xC0FFEE)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: fake_stream)

    waited = backend.wait(conn, 0)

    assert waited == 0.0
    assert backend._gpu_wait_cs == 0xC0FFEE
    conn.cuda.stream_wait_event.assert_called_once_with(0xC0FFEE, conn.ipc_events[0], 0)


def test_torch_backend_wait_gpu_wait_active_caches_stream_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock", num_slots=2)
    imp._gpu_wait_active = True
    conn = imp._conn
    backend = _TorchBackend(imp, stream=None)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    current_stream_calls = []

    def _current_stream() -> SimpleNamespace:
        current_stream_calls.append(1)
        return SimpleNamespace(cuda_stream=0xC0FFEE)

    monkeypatch.setattr(torch.cuda, "current_stream", _current_stream)

    backend.wait(conn, 0)
    backend.wait(conn, 1)

    # torch.cuda.current_stream() is only queried once -- the raw handle is cached.
    assert len(current_stream_calls) == 1


def test_torch_backend_wait_gpu_wait_active_without_event_falls_through_to_cpu_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imp = make_connected_importer(with_events=False, dev_ptr_style="mock")
    imp._gpu_wait_active = True
    conn = imp._conn
    backend = _TorchBackend(imp, stream=None)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0x1))

    waited = backend.wait(conn, 0)

    # No IPC event for the slot -- must fall through to the CPU-side wait
    # (conn.cuda.synchronize(), since ipc_events[0] is None/falsy).
    assert isinstance(waited, float)


def test_torch_backend_wait_gpu_wait_active_but_cuda_unavailable_falls_through_to_cpu_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch.cuda.is_available() False -- even with gpu_wait_active and an IPC
    event present, the GPU-side stream_wait_event() path must be skipped
    entirely and control must fall through to the CPU-side wait."""
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    imp._gpu_wait_active = True
    conn = imp._conn
    backend = _TorchBackend(imp, stream=None)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    waited = backend.wait(conn, 0)

    assert isinstance(waited, float)
    conn.cuda.stream_wait_event.assert_not_called()


def test_torch_backend_materialize_returns_cached_tensor() -> None:
    imp = make_connected_importer()
    sentinel_tensor = object()
    imp._torch = SimpleNamespace(tensors=[sentinel_tensor])
    backend = _TorchBackend(imp, stream=None)

    assert backend.materialize(imp._conn, 0) is sentinel_tensor


# ---------------------------------------------------------------------------
# _NumpyBackend.prepare()
# ---------------------------------------------------------------------------


def test_numpy_backend_prepare_builds_when_numpy_is_none() -> None:
    imp = make_connected_importer(with_numpy=False)
    imp._conn.cuda = FakeCUDAAdapter()
    assert imp._numpy is None
    backend = _NumpyBackend(imp)

    backend.prepare(imp)

    assert imp._numpy is not None
    assert imp._numpy.buffer.shape == imp._format.shape


def test_numpy_backend_prepare_closes_and_rebuilds_on_format_change() -> None:
    from cuda_link.importer import Format

    imp = make_connected_importer(shape=(4, 4, 4), with_numpy=False)
    imp._conn.cuda = FakeCUDAAdapter()
    backend = _NumpyBackend(imp)
    backend.prepare(imp)
    old_numpy = imp._numpy
    close_spy = MagicMock(wraps=old_numpy.close)
    old_numpy.close = close_spy

    # Change format to force needs_rebuild() -> True.
    imp._format = Format.from_overrides((8, 8, 4), "uint8")
    backend.prepare(imp)

    close_spy.assert_called_once_with()
    assert imp._numpy is not old_numpy
    assert imp._numpy.buffer.shape == (8, 8, 4)


def test_numpy_backend_prepare_no_rebuild_when_format_unchanged() -> None:
    imp = make_connected_importer(with_numpy=False)
    imp._conn.cuda = FakeCUDAAdapter()
    backend = _NumpyBackend(imp)
    backend.prepare(imp)
    old_numpy = imp._numpy

    backend.prepare(imp)  # same format -- needs_rebuild() False

    assert imp._numpy is old_numpy


# ---------------------------------------------------------------------------
# _NumpyBackend.wait() / materialize()
# ---------------------------------------------------------------------------


def test_numpy_backend_wait_delegates_to_wait_for_slot() -> None:
    imp = make_connected_importer()
    backend = _NumpyBackend(imp)
    spy = MagicMock(return_value=12.5)
    imp._wait_for_slot = spy

    assert backend.wait(imp._conn, 0) == 12.5
    spy.assert_called_once_with(0)


def test_numpy_backend_materialize_single_stream() -> None:
    imp = make_connected_importer(dev_ptr_style="c_void_p")
    fake_cuda = FakeCUDAAdapter()
    wrapped = MagicMock(wraps=fake_cuda)
    imp._conn.cuda = wrapped
    imp._numpy = NumpyBuffers.build(imp._conn, imp._format, num_streams=1)
    backend = _NumpyBackend(imp)
    conn = imp._conn

    result = backend.materialize(conn, 0)

    assert result is imp._numpy.buffer
    # materialize() unconditionally returns nb.buffer at the end regardless of
    # whether the D2H copy actually ran -- pin the actual memcpy_async/
    # stream_synchronize calls too, so a regression that drops the copy
    # itself (not just the return value) is caught.
    wrapped.memcpy_async.assert_called_once_with(
        dst=imp._numpy.buffer_ptr,
        src=conn.dev_ptrs[0],
        count=imp._format.frame_nbytes,
        kind=2,
        stream=imp._numpy.primary_stream,
    )
    wrapped.stream_synchronize.assert_called_once_with(imp._numpy.primary_stream)
    imp._numpy.close()


def test_numpy_backend_materialize_multi_stream_uses_chunk_plan() -> None:
    imp = make_connected_importer(shape=(16, 16, 4), dev_ptr_style="c_void_p")
    fake_cuda = FakeCUDAAdapter()
    wrapped = MagicMock(wraps=fake_cuda)
    imp._conn.cuda = wrapped
    imp._numpy = NumpyBuffers.build(imp._conn, imp._format, num_streams=3)
    assert len(imp._numpy.chunk_plan) == 3
    backend = _NumpyBackend(imp)

    result = backend.materialize(imp._conn, 0)

    assert result is imp._numpy.buffer
    assert wrapped.memcpy_async.call_count == 3
    assert wrapped.stream_synchronize.call_count == 3
    imp._numpy.close()


def test_numpy_backend_materialize_pipelined_priming_returns_none() -> None:
    imp = make_connected_importer(dev_ptr_style="c_void_p")
    imp._conn.cuda = FakeCUDAAdapter()
    imp._numpy = NumpyBuffers.build(imp._conn, imp._format, num_streams=1, pipelined=True)
    assert imp._numpy.priming is True
    backend = _NumpyBackend(imp)

    result = backend.materialize(imp._conn, 0)

    assert result is None  # priming call -- surfaces as NO_FRAME to the caller
    assert imp._numpy.priming is False
    imp._numpy.close()


def test_numpy_backend_materialize_pipelined_steady_state_returns_front_buffer() -> None:
    imp = make_connected_importer(dev_ptr_style="c_void_p")
    imp._conn.cuda = FakeCUDAAdapter()
    imp._numpy = NumpyBuffers.build(imp._conn, imp._format, num_streams=1, pipelined=True)
    backend = _NumpyBackend(imp)
    backend.materialize(imp._conn, 0)  # priming call

    result = backend.materialize(imp._conn, 0)  # steady-state call

    assert result is not None
    imp._numpy.close()


# ---------------------------------------------------------------------------
# _CupyBackend.wait() / materialize()
# ---------------------------------------------------------------------------


def test_cupy_backend_wait_no_event_falls_back_to_wait_for_slot() -> None:
    imp = make_connected_importer(with_events=False, dev_ptr_style="mock")
    backend = _CupyBackend(imp, stream=None)
    spy = MagicMock(return_value=5.0)
    imp._wait_for_slot = spy

    assert backend.wait(imp._conn, 0) == 5.0
    spy.assert_called_once_with(0)


def test_cupy_backend_wait_default_stream_uses_get_current_stream() -> None:
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    conn = imp._conn
    backend = _CupyBackend(imp, stream=None)

    mock_cp = MagicMock()
    mock_stream = MagicMock(ptr=0xAAAA)
    mock_cp.cuda.get_current_stream.return_value = mock_stream
    with patch("cuda_link.importer.cp", mock_cp), patch("cuda_link.importer.CUPY_AVAILABLE", True):
        waited = backend.wait(conn, 0)

    assert waited == 0.0
    mock_cp.cuda.get_current_stream.assert_called_once_with()
    # Pin the resolved stream's .ptr (0xAAAA), not just "called once" -- a bug
    # that threads the wrong pointer (e.g. 0, or self._stream instead of the
    # resolved current stream) would still satisfy a bare assert_called_once().
    mock_cp.cuda.runtime.streamWaitEvent.assert_called_once_with(0xAAAA, _event_to_int(conn.ipc_events[0]), 0)


def test_cupy_backend_wait_explicit_non_cupy_stream_resolves_via_external_stream() -> None:
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    conn = imp._conn
    backend = _CupyBackend(imp, stream=0xBEEF)

    mock_cp = MagicMock()

    class _NotCupyStream:
        pass

    mock_cp.cuda.Stream = _NotCupyStream
    mock_external = MagicMock(ptr=0xBEEF)
    mock_cp.cuda.ExternalStream.return_value = mock_external
    with patch("cuda_link.importer.cp", mock_cp), patch("cuda_link.importer.CUPY_AVAILABLE", True):
        waited = backend.wait(conn, 0)

    assert waited == 0.0
    mock_cp.cuda.ExternalStream.assert_called_once_with(0xBEEF)
    # Pin the resolved ExternalStream's .ptr (0xBEEF), not just "called once" --
    # a bug that passes the wrong pointer through to streamWaitEvent would
    # still satisfy a bare assert_called_once().
    mock_cp.cuda.runtime.streamWaitEvent.assert_called_once_with(0xBEEF, _event_to_int(conn.ipc_events[0]), 0)


def test_cupy_backend_wait_stream_already_cupy_stream_instance_skips_external_stream() -> None:
    """When the caller-supplied stream is already a cp.cuda.Stream instance,
    the ExternalStream() wrapping/resolution step must be skipped entirely."""
    imp = make_connected_importer(with_events=True, dev_ptr_style="mock")
    conn = imp._conn

    mock_cp = MagicMock()

    class _CupyStream:
        ptr = 0xC0DE

    mock_cp.cuda.Stream = _CupyStream
    already_cupy_stream = _CupyStream()
    backend = _CupyBackend(imp, stream=already_cupy_stream)

    with patch("cuda_link.importer.cp", mock_cp), patch("cuda_link.importer.CUPY_AVAILABLE", True):
        waited = backend.wait(conn, 0)

    assert waited == 0.0
    mock_cp.cuda.ExternalStream.assert_not_called()
    mock_cp.cuda.runtime.streamWaitEvent.assert_called_once_with(0xC0DE, _event_to_int(conn.ipc_events[0]), 0)


def test_cupy_backend_materialize_returns_cached_array() -> None:
    imp = make_connected_importer()
    sentinel_array = object()
    imp._cupy = SimpleNamespace(arrays=[sentinel_array])
    backend = _CupyBackend(imp, stream=None)

    assert backend.materialize(imp._conn, 0) is sentinel_array
