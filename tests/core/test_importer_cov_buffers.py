"""Coverage tests for TorchBuffers.build(), CupyBuffers.build(), and NumpyBuffers.

Builds IPCConnection scaffolds directly via fakes.make_fake_ipc_connection() and
calls the buffer factories/close() paths directly -- no real GPU or SHM required.
CuPy-touching code is exercised via unittest.mock.patch("cuda_link.importer.cp", ...)
matching the established pattern in tests/core/test_importer.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fakes import make_fake_ipc_connection

from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.importer import CupyBuffers, Format, NumpyBuffers, TorchBuffers

# ---------------------------------------------------------------------------
# TorchBuffers.build()
# ---------------------------------------------------------------------------


def test_torch_buffers_build_raises_on_unsupported_dtype() -> None:
    conn, _, _ = make_fake_ipc_connection(num_slots=1, dev_ptr_style="c_void_p")
    # Bypass Format.from_overrides (which itself KeyErrors on a truly unknown
    # dtype via _numpy_dtype_for) -- construct the Format dataclass directly
    # so only DtypeCodec.typestr() inside TorchBuffers.build() sees the
    # unsupported string.
    fmt = Format(
        width=4,
        height=2,
        num_comps=2,
        kind=0,
        bits=0,
        flags=0,
        dtype_str="not_a_real_dtype",
        shape=(2, 2, 4),
        numpy_dtype=None,
        frame_nbytes=2 * 2 * 4 * 4,
    )

    with pytest.raises(ValueError, match="Unsupported dtype for torch"):
        TorchBuffers.build(conn, fmt)


def test_torch_buffers_build_raises_when_dev_ptr_not_initialized() -> None:
    conn, _, _ = make_fake_ipc_connection(num_slots=1, dev_ptr_style="none")
    fmt = Format.from_overrides((2, 2, 4), "uint8")

    with pytest.raises(RuntimeError, match="Device pointer for slot 0 not initialized"):
        TorchBuffers.build(conn, fmt)


def test_torch_buffers_build_bfloat16_uses_uint16_view_and_reinterprets(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.as_tensor() is mocked so the fake c_void_p pointer never touches real CUDA --
    this must stay deterministic on both this GPU-equipped dev machine and a genuinely
    GPU-less CI runner.
    """
    conn, _, _ = make_fake_ipc_connection(num_slots=1, dev_ptr_style="c_void_p")
    fmt = Format.from_overrides((2, 2, 4), "bfloat16")

    mock_tensor = MagicMock()
    mock_viewed = MagicMock()
    mock_tensor.view.return_value = mock_viewed
    monkeypatch.setattr(torch, "as_tensor", lambda wrapper, device: mock_tensor)  # noqa: ARG005

    buffers = TorchBuffers.build(conn, fmt)

    assert buffers.tensors == [mock_viewed]
    mock_tensor.view.assert_called_once_with(torch.bfloat16)


# ---------------------------------------------------------------------------
# CupyBuffers.build()
# ---------------------------------------------------------------------------


def test_cupy_buffers_build_raises_for_bfloat16() -> None:
    conn, _, _ = make_fake_ipc_connection(num_slots=1, dev_ptr_style="c_void_p")
    fmt = Format.from_overrides((2, 2, 4), "bfloat16")

    with (
        patch("cuda_link.importer.cp", MagicMock()),
        patch("cuda_link.importer.CUPY_AVAILABLE", True),
        pytest.raises(ValueError, match="not supported by the CuPy consumer"),
    ):
        CupyBuffers.build(conn, fmt)


def test_cupy_buffers_build_raises_on_unsupported_cp_dtype() -> None:
    conn, _, _ = make_fake_ipc_connection(num_slots=1, dev_ptr_style="c_void_p")
    fmt = Format.from_overrides((2, 2, 4), "uint8")

    mock_cp = MagicMock()
    mock_cp.dtype.side_effect = TypeError("bad dtype")
    with (
        patch("cuda_link.importer.cp", mock_cp),
        patch("cuda_link.importer.CUPY_AVAILABLE", True),
        pytest.raises(ValueError, match="Unsupported dtype for CuPy"),
    ):
        CupyBuffers.build(conn, fmt)


def test_cupy_buffers_build_raises_when_dev_ptr_not_initialized() -> None:
    conn, _, _ = make_fake_ipc_connection(num_slots=1, dev_ptr_style="none")
    fmt = Format.from_overrides((2, 2, 4), "uint8")

    mock_cp = MagicMock()
    with (
        patch("cuda_link.importer.cp", mock_cp),
        patch("cuda_link.importer.CUPY_AVAILABLE", True),
        pytest.raises(RuntimeError, match="Device pointer for slot 0 not initialized"),
    ):
        CupyBuffers.build(conn, fmt)


def test_cupy_buffers_build_success_builds_one_array_per_slot() -> None:
    conn, _, _ = make_fake_ipc_connection(num_slots=2, dev_ptr_style="c_void_p")
    fmt = Format.from_overrides((2, 2, 4), "uint8")

    mock_cp = MagicMock()
    mock_cp.dtype.return_value = "uint8"
    with patch("cuda_link.importer.cp", mock_cp), patch("cuda_link.importer.CUPY_AVAILABLE", True):
        buffers = CupyBuffers.build(conn, fmt)

    assert len(buffers.arrays) == 2
    assert mock_cp.cuda.UnownedMemory.call_count == 2
    assert mock_cp.ndarray.call_count == 2
    # Each slot's own dev_ptr must be threaded through individually -- not
    # slot 0's pointer reused for every slot. dev_ptr_style="c_void_p" gives
    # slot 0 -> 0x1000, slot 1 -> 0x2000 (see fakes.make_fake_ipc_connection).
    mock_cp.cuda.UnownedMemory.assert_any_call(0x1000, fmt.frame_nbytes, owner=conn)
    mock_cp.cuda.UnownedMemory.assert_any_call(0x2000, fmt.frame_nbytes, owner=conn)
    # Every ndarray view must use the resolved shape/dtype, regardless of slot.
    for args, kwargs in mock_cp.ndarray.call_args_list:
        assert args == (fmt.shape,)
        assert kwargs["dtype"] == "uint8"


# ---------------------------------------------------------------------------
# NumpyBuffers.build() -- allocation ladder
# ---------------------------------------------------------------------------


def _fmt(shape=(2, 2, 4), dtype="uint8") -> Format:
    return Format.from_overrides(shape, dtype)


def test_numpy_buffers_build_raises_for_dtype_with_no_numpy_representation() -> None:
    """bfloat16 without ml_dtypes -> numpy_dtype is None -> explicit ValueError."""
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    fmt = _fmt(dtype="bfloat16")
    assert fmt.numpy_dtype is None

    with pytest.raises(ValueError, match="NumPy consumer cannot be used"):
        NumpyBuffers.build(conn, fmt, num_streams=1)


def test_numpy_buffers_build_pinned_alloc_success() -> None:
    cuda = FakeCUDAAdapter()
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    nb = NumpyBuffers.build(conn, fmt, num_streams=1)

    assert nb.pinned_memory_available is True
    assert nb.pinned_ptr is not None
    assert nb.buffer.shape == fmt.shape
    nb.close()


def test_numpy_buffers_build_falls_back_to_host_register_when_malloc_host_fails() -> None:
    cuda = FakeCUDAAdapter()
    cuda.malloc_host_alloc = MagicMock(side_effect=RuntimeError("no pinned pages"))
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    nb = NumpyBuffers.build(conn, fmt, num_streams=1, allow_pageable=True)

    assert nb.pinned_ptr is None
    assert nb.host_registered_arr is not None
    assert nb.pinned_memory_available is True
    nb.close()


def test_numpy_buffers_build_falls_back_to_pageable_when_both_fail() -> None:
    cuda = FakeCUDAAdapter()
    cuda.malloc_host_alloc = MagicMock(side_effect=RuntimeError("no pinned pages"))
    cuda.host_register = MagicMock(side_effect=OSError("register failed"))
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    nb = NumpyBuffers.build(conn, fmt, num_streams=1, allow_pageable=True)

    assert nb.pinned_ptr is None
    assert nb.host_registered_arr is None
    assert nb.pinned_memory_available is False
    assert isinstance(nb.buffer, np.ndarray)
    nb.close()


def test_numpy_buffers_build_raises_when_pinned_fails_and_pageable_disallowed() -> None:
    cuda = FakeCUDAAdapter()
    cuda.malloc_host_alloc = MagicMock(side_effect=RuntimeError("no pinned pages"))
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    with pytest.raises(RuntimeError, match="Pinned-memory allocation failed"):
        NumpyBuffers.build(conn, fmt, num_streams=1, allow_pageable=False)


def test_numpy_buffers_build_multi_stream_creates_chunk_plan() -> None:
    cuda = FakeCUDAAdapter()
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt(shape=(4, 4, 4))

    nb = NumpyBuffers.build(conn, fmt, num_streams=2)

    assert nb.num_streams == 2
    assert len(nb.chunk_plan) == 2
    assert len(nb.d2h_streams) == 2
    nb.close()


def test_numpy_buffers_build_high_priority_streams() -> None:
    fake_cuda = FakeCUDAAdapter()
    cuda = MagicMock(wraps=fake_cuda)
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    nb = NumpyBuffers.build(conn, fmt, num_streams=1, high_priority=True)

    assert nb.primary_stream is not None
    # high_priority=True must route through create_stream_with_priority, not
    # the plain create_stream() path -- FakeCUDAAdapter returns a non-None
    # handle from both, so "primary_stream is not None" alone can't tell them
    # apart. Pin the actual call to catch a regression that silently ignores
    # the flag.
    cuda.create_stream_with_priority.assert_called_once_with(flags=0x01)
    cuda.create_stream.assert_not_called()
    nb.close()


# ---------------------------------------------------------------------------
# NumpyBuffers.build() -- pipelined back-buffer ladder
# ---------------------------------------------------------------------------


def test_numpy_buffers_build_pipelined_back_buffer_from_pinned() -> None:
    cuda = FakeCUDAAdapter()
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    nb = NumpyBuffers.build(conn, fmt, num_streams=1, pipelined=True)

    assert nb.pipelined is True
    assert nb.priming is True
    assert nb.back_pinned_ptr is not None
    assert nb.back_buffer is not None
    nb.close()


def test_numpy_buffers_build_pipelined_back_buffer_from_host_register() -> None:
    cuda = FakeCUDAAdapter()
    cuda.malloc_host_alloc = MagicMock(side_effect=RuntimeError("no pinned pages"))
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    nb = NumpyBuffers.build(conn, fmt, num_streams=1, pipelined=True, allow_pageable=True)

    assert nb.pinned_ptr is None
    assert nb.host_registered_arr is not None
    assert nb.back_host_registered_arr is not None
    assert nb.back_buffer is not None
    nb.close()


def test_numpy_buffers_build_pipelined_back_buffer_pageable_fallback() -> None:
    cuda = FakeCUDAAdapter()
    cuda.malloc_host_alloc = MagicMock(side_effect=RuntimeError("no pinned pages"))
    cuda.host_register = MagicMock(side_effect=OSError("register failed"))
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()

    nb = NumpyBuffers.build(conn, fmt, num_streams=1, pipelined=True, allow_pageable=True)

    assert nb.pinned_ptr is None
    assert nb.host_registered_arr is None
    assert isinstance(nb.back_buffer, np.ndarray)
    nb.close()


# ---------------------------------------------------------------------------
# NumpyBuffers.close() -- error-swallowing branches
# ---------------------------------------------------------------------------


def _built_numpy_buffers(cuda: FakeCUDAAdapter, **build_kwargs: object) -> NumpyBuffers:
    conn, _, _ = make_fake_ipc_connection(num_slots=1)
    conn.cuda = cuda
    fmt = _fmt()
    return NumpyBuffers.build(conn, fmt, num_streams=build_kwargs.pop("num_streams", 1), **build_kwargs)


def test_numpy_buffers_close_swallows_free_host_error() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda)
    cuda.free_host = MagicMock(side_effect=RuntimeError("free failed"))

    nb.close()  # must not raise

    assert nb.pinned_ptr is None
    assert nb.buffer is None


def test_numpy_buffers_close_swallows_host_unregister_error() -> None:
    cuda = FakeCUDAAdapter()
    cuda.malloc_host_alloc = MagicMock(side_effect=RuntimeError("no pinned pages"))
    nb = _built_numpy_buffers(cuda, allow_pageable=True)
    assert nb.host_registered_arr is not None
    cuda.host_unregister = MagicMock(side_effect=OSError("unregister failed"))

    nb.close()  # must not raise

    assert nb.host_registered_arr is None


def test_numpy_buffers_close_swallows_destroy_stream_error() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda)
    cuda.destroy_stream = MagicMock(side_effect=RuntimeError("destroy failed"))

    nb.close()  # must not raise

    assert nb.primary_stream is None


def test_numpy_buffers_close_swallows_secondary_stream_destroy_error() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda, num_streams=2)
    assert len(nb.d2h_streams) == 2
    cuda.destroy_stream = MagicMock(side_effect=OSError("destroy failed"))

    nb.close()  # must not raise -- secondary-stream loop swallows too

    assert nb.primary_stream is None


def test_numpy_buffers_close_skips_none_secondary_stream() -> None:
    """A None entry beyond index 0 in d2h_streams (already torn down, or never
    created) must be skipped rather than passed to destroy_stream()."""
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda, num_streams=2)
    nb.d2h_streams[1] = None
    primary = nb.primary_stream
    cuda.destroy_stream = MagicMock()

    nb.close()  # must not raise

    # Only the primary stream (index 0) is destroyed -- the None secondary
    # entry is skipped by the "if stream is not None" guard in the loop.
    cuda.destroy_stream.assert_called_once_with(primary)
    assert nb.primary_stream is None


def test_numpy_buffers_close_swallows_back_buffer_free_host_error() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda, pipelined=True)
    assert nb.back_pinned_ptr is not None
    cuda.free_host = MagicMock(side_effect=RuntimeError("free failed"))

    nb.close()  # must not raise

    assert nb.back_pinned_ptr is None
    assert nb.back_buffer is None


def test_numpy_buffers_close_swallows_back_host_unregister_error() -> None:
    cuda = FakeCUDAAdapter()
    cuda.malloc_host_alloc = MagicMock(side_effect=RuntimeError("no pinned pages"))
    nb = _built_numpy_buffers(cuda, pipelined=True, allow_pageable=True)
    assert nb.back_host_registered_arr is not None
    cuda.host_unregister = MagicMock(side_effect=OSError("unregister failed"))

    nb.close()  # must not raise

    assert nb.back_host_registered_arr is None


def test_numpy_buffers_close_is_idempotent() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda)

    nb.close()
    nb.close()  # second call must be a no-op, not raise

    assert nb.pinned_ptr is None
    assert nb.primary_stream is None


# ---------------------------------------------------------------------------
# NumpyBuffers.needs_rebuild() / rotate()
# ---------------------------------------------------------------------------


def test_numpy_buffers_needs_rebuild_true_on_shape_change() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda)
    other_fmt = _fmt(shape=(8, 8, 4))

    assert nb.needs_rebuild(other_fmt) is True
    nb.close()


def test_numpy_buffers_needs_rebuild_false_when_unchanged() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda)

    assert nb.needs_rebuild(nb.fmt) is False
    nb.close()


def test_numpy_buffers_rotate_swaps_front_and_back() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda, pipelined=True)
    front, back = nb.buffer, nb.back_buffer
    front_ptr, back_ptr = nb.buffer_ptr, nb.back_buffer_ptr

    nb.rotate()

    assert nb.buffer is back
    assert nb.back_buffer is front
    assert nb.buffer_ptr is back_ptr
    assert nb.back_buffer_ptr is front_ptr
    nb.close()


def test_numpy_buffers_drain_swallows_stream_synchronize_error() -> None:
    cuda = FakeCUDAAdapter()
    nb = _built_numpy_buffers(cuda)
    cuda.stream_synchronize = MagicMock(side_effect=RuntimeError("sync failed"))

    nb.drain()  # must not raise
    nb.close()
