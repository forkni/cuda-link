"""
Tests for the ctypes class-identity guard in CUDARuntimeAPI.ipc_open_mem_handle
and CUDARuntimeAPI.ipc_open_event_handle.

Background
----------
ctypes validates Structure arguments against argtypes by class *identity*, not
structural equivalence.  In TouchDesigner's bare-name import namespace two
separate imports of cuda_link.cuda_runtime_types can produce two distinct class
objects that share the name ``cudaIpcMemHandle_t``.  A handle built with class A
is rejected by argtypes bound to class B:

    ArgumentError: argument 2: TypeError: expected cudaIpcMemHandle_t
                   instance instead of cudaIpcMemHandle_t

The guard (added in the class-identity-mismatch fix) normalises the incoming
handle via ``from_buffer_copy(bytes(handle))`` whenever ``isinstance`` fails,
making the IPC-open calls immune to the duplicate-class problem.

These tests are intentionally GPU-free — they bypass CUDARuntimeAPI.__init__
(which loads the CUDA runtime) by using ``object.__new__`` and inject a
MagicMock cudart.  No ``@pytest.mark.requires_cuda`` marker needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_api() -> object:
    """Return a CUDARuntimeAPI instance with a mock cudart, no GPU required."""
    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    api = CUDARuntimeAPI.__new__(CUDARuntimeAPI)
    api.cudart = MagicMock()
    # ipc_open_mem_handle reads/writes a c_void_p; make cudaIpcOpenMemHandle a no-op.
    api.cudart.cudaIpcOpenMemHandle.return_value = None
    api.cudart.cudaIpcOpenEventHandle.return_value = None
    return api


class _ForeignMemHandle:
    """Simulates a cudaIpcMemHandle_t built from a *different* class object."""

    _raw: bytes = bytes(64)  # 64-byte POD, same layout

    def __bytes__(self) -> bytes:
        return self._raw


class _ForeignEventHandle:
    """Simulates a cudaIpcEventHandle_t built from a *different* class object."""

    _raw: bytes = bytes(64)

    def __bytes__(self) -> bytes:
        return self._raw


# ---------------------------------------------------------------------------
# ipc_open_mem_handle
# ---------------------------------------------------------------------------


def test_ipc_open_mem_handle_passthrough_real_handle() -> None:
    """A real cudaIpcMemHandle_t instance passes through unchanged (no copy)."""
    # Import the class from the wrapper module itself — this is the class object
    # the wrapper actually bound into argtypes (which may be CUDARuntimeTypes or
    # cuda_link.cuda_runtime_types depending on sys.path, e.g. TD bare-name vs
    # pytest env).  Importing from the wrapper's own namespace guarantees the test
    # always exercises the correct isinstance branch regardless of environment.
    import cuda_link.cuda_ipc_wrapper as _w

    cudaIpcMemHandle_t = _w.cudaIpcMemHandle_t

    api = _make_api()
    handle = cudaIpcMemHandle_t()
    api.ipc_open_mem_handle(handle)

    # The handle forwarded to cudart must be the exact same object — isinstance
    # True → guard is a no-op.
    forwarded = api.cudart.cudaIpcOpenMemHandle.call_args[0][1]
    assert forwarded is handle, "Guard must NOT copy a handle that already is an instance of the correct class"


def test_ipc_open_mem_handle_rebuilds_foreign_handle() -> None:
    """A handle from a foreign class is rebuilt via from_buffer_copy."""
    import cuda_link.cuda_ipc_wrapper as _w

    cudaIpcMemHandle_t = _w.cudaIpcMemHandle_t

    api = _make_api()
    foreign = _ForeignMemHandle()
    api.ipc_open_mem_handle(foreign)  # type: ignore[arg-type]

    forwarded = api.cudart.cudaIpcOpenMemHandle.call_args[0][1]
    assert isinstance(forwarded, cudaIpcMemHandle_t), (
        "Guard must rebuild a foreign handle into the wrapper's cudaIpcMemHandle_t class"
    )
    assert bytes(forwarded) == bytes(64), "Rebuilt handle must contain the original bytes"


def test_ipc_open_mem_handle_foreign_preserves_bytes() -> None:
    """Content of foreign handle is preserved faithfully after rebuild."""
    import cuda_link.cuda_ipc_wrapper as _w

    cudaIpcMemHandle_t = _w.cudaIpcMemHandle_t

    # Create a foreign handle with non-zero content.
    raw = bytes(range(64))

    class _NonZeroForeign:
        def __bytes__(self) -> bytes:
            return raw

    api = _make_api()
    api.ipc_open_mem_handle(_NonZeroForeign())  # type: ignore[arg-type]

    forwarded = api.cudart.cudaIpcOpenMemHandle.call_args[0][1]
    assert isinstance(forwarded, cudaIpcMemHandle_t)
    assert bytes(forwarded) == raw, "Byte content must survive the from_buffer_copy rebuild"


# ---------------------------------------------------------------------------
# ipc_open_event_handle
# ---------------------------------------------------------------------------


def test_ipc_open_event_handle_passthrough_real_handle() -> None:
    """A real cudaIpcEventHandle_t instance passes through unchanged (no copy)."""
    import cuda_link.cuda_ipc_wrapper as _w

    cudaIpcEventHandle_t = _w.cudaIpcEventHandle_t

    api = _make_api()
    handle = cudaIpcEventHandle_t()
    api.ipc_open_event_handle(handle)

    forwarded = api.cudart.cudaIpcOpenEventHandle.call_args[0][1]
    assert forwarded is handle, "Guard must NOT copy a handle that already is an instance of the correct class"


def test_ipc_open_event_handle_rebuilds_foreign_handle() -> None:
    """A foreign cudaIpcEventHandle_t is rebuilt via from_buffer_copy."""
    import cuda_link.cuda_ipc_wrapper as _w

    cudaIpcEventHandle_t = _w.cudaIpcEventHandle_t

    api = _make_api()
    foreign = _ForeignEventHandle()
    api.ipc_open_event_handle(foreign)  # type: ignore[arg-type]

    forwarded = api.cudart.cudaIpcOpenEventHandle.call_args[0][1]
    assert isinstance(forwarded, cudaIpcEventHandle_t), (
        "Guard must rebuild a foreign handle into the wrapper's cudaIpcEventHandle_t class"
    )
    assert bytes(forwarded) == bytes(64)


def test_ipc_open_event_handle_foreign_preserves_bytes() -> None:
    """Content of foreign event handle is preserved after rebuild."""
    import cuda_link.cuda_ipc_wrapper as _w

    cudaIpcEventHandle_t = _w.cudaIpcEventHandle_t

    raw = bytes(range(64))

    class _NonZeroForeign:
        def __bytes__(self) -> bytes:
            return raw

    api = _make_api()
    api.ipc_open_event_handle(_NonZeroForeign())  # type: ignore[arg-type]

    forwarded = api.cudart.cudaIpcOpenEventHandle.call_args[0][1]
    assert isinstance(forwarded, cudaIpcEventHandle_t)
    assert bytes(forwarded) == raw
