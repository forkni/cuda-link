"""
Test fixture factories for building fake IPCConnection objects and connected Importers.

Shared across test_importer.py, test_wait_for_slot_busywait.py, and
test_cuda_ipc_importer.py — each of which builds a different Importer
but uses the same IPCConnection scaffold (bytearray SHM + MagicMock cuda).
"""

from __future__ import annotations

from ctypes import c_void_p
from unittest.mock import MagicMock

from cuda_link.importer import IPCConnection
from cuda_link.shm_protocol import SHMLayout


def make_fake_ipc_connection(
    *,
    num_slots: int = 1,
    ipc_version: int = 1,
    write_idx: int = 1,
    with_events: bool = False,
    with_shm_handle: bool = True,
    dev_ptr_style: str = "c_void_p",
) -> tuple[IPCConnection, MagicMock, MagicMock | None]:
    """Build a fake IPCConnection and companion mocks for Importer unit tests.

    Args:
        num_slots: Number of ring-buffer slots.
        ipc_version: SHM protocol version field.
        write_idx: SHM write_idx value (0 = no frames written yet).
        with_events: True → ipc_events=[MagicMock()]*N (GPU event path).
                     False → ipc_events=[None]*N (CPU poll / event-less path).
        with_shm_handle: True → mock_shm.buf = bytearray with packed header.
                         False → shm_handle=None (busywait tests that skip SHM reads).
        dev_ptr_style: "c_void_p" → c_void_p(0x1000*(i+1)) per slot.
                       "mock"     → MagicMock() per slot.
                       "none"     → None per slot.

    Returns:
        (conn, mock_cuda, mock_shm)
        - conn: IPCConnection with all fields populated.
        - mock_cuda: MagicMock with query_event.return_value=True pre-wired.
        - mock_shm: MagicMock with .buf set, or None if with_shm_handle=False.
    """
    layout = SHMLayout(num_slots)
    buf = layout.build_buffer(version=ipc_version, write_idx=write_idx)

    mock_cuda = MagicMock()
    mock_cuda.query_event.return_value = True

    if with_shm_handle:
        mock_shm: MagicMock | None = MagicMock()
        mock_shm.buf = buf
    else:
        mock_shm = None

    if dev_ptr_style == "c_void_p":
        dev_ptrs = [c_void_p(0x1000 * (i + 1)) for i in range(num_slots)]
    elif dev_ptr_style == "mock":
        dev_ptrs = [MagicMock() for _ in range(num_slots)]
    elif dev_ptr_style == "none":
        dev_ptrs = [None] * num_slots
    else:
        raise ValueError(f"Unknown dev_ptr_style: {dev_ptr_style!r}")

    ipc_events = [MagicMock() for _ in range(num_slots)] if with_events else [None] * num_slots

    conn = IPCConnection(
        cuda=mock_cuda,
        shm_handle=mock_shm,
        ipc_version=ipc_version,
        num_slots=num_slots,
        ipc_handles=[None] * num_slots,
        dev_ptrs=dev_ptrs,
        ipc_events=ipc_events,
        layout=layout,
        shutdown_offset=layout.shutdown_offset,
        timestamp_offset=layout.timestamp_offset,
    )
    return conn, mock_cuda, mock_shm


def make_connected_importer(
    shape: tuple = (8, 8, 4),
    dtype: str = "uint8",
    *,
    num_slots: int = 1,
    write_idx: int = 1,
    ipc_version: int = 1,
    timeout_ms: float = 5000.0,
    spin_us: int = 0,
    allow_pageable_fallback: bool = True,
    debug: bool = False,
    dev_ptr_style: str = "c_void_p",
    with_shm_handle: bool = True,
    with_events: bool = False,
    with_numpy: bool = True,
    last_write_idx: int = 0,
    cupy: object | None = None,
    numpy: object | None = None,
) -> object:
    """Build a connected Importer via ``Importer.from_connection()`` — no GPU or SHM required.

    This is the canonical factory for obtaining a connected Importer in unit tests.
    It replaces the scattered ``_make_connected_importer`` / ``_make_importer_with_mock_state``
    helpers that hand-populated private attributes directly.

    Args:
        shape:                 Frame geometry (H, W, C).
        dtype:                 dtype string (e.g. "uint8", "float32").
        num_slots:             Ring-buffer slot count.
        write_idx:             Initial SHM write_idx (1 = one frame available).
        ipc_version:           SHM protocol version.
        timeout_ms:            ImportSpec timeout.
        spin_us:               ImportPolicy spin budget.
        allow_pageable_fallback: ImportPolicy flag for numpy D2H fallback.
        debug:                 ImportPolicy debug flag; enables per-frame stats logging.
        dev_ptr_style:         Passed to ``make_fake_ipc_connection`` — "c_void_p" for
                               most tests, "mock" for tests that patch cuda methods
                               on the connection (e.g. stream_wait_event).
        with_shm_handle:       False to omit the SHM handle (busywait tests).
        with_events:           True to populate ipc_events with MagicMocks.
        with_numpy:            True (default) to auto-build a NumpyBuffers from ``shape``
                               and ``dtype``. Ignored when ``numpy`` is given explicitly.
        last_write_idx:        Forwarded to ``Importer.from_connection`` — lets tests
                               simulate an Importer that has already consumed some frames
                               without touching ``imp._last_write_idx`` directly.
        cupy:                  Explicit CupyBuffers (or a MagicMock) to pass to
                               ``from_connection`` — skips the auto-build and avoids
                               ``imp._cupy = …`` injection in tests.
        numpy:                 Explicit NumpyBuffers (or a MagicMock) to pass to
                               ``from_connection`` — overrides ``with_numpy``.

    Returns:
        A connected ``Importer`` instance ready for ``get_frame*()`` calls.
    """
    from unittest.mock import MagicMock

    import numpy as np

    from cuda_link._cuda_adapters import FakeCUDAAdapter
    from cuda_link._importer_port import ImportPolicy, ImportSpec
    from cuda_link.importer import Format, Importer, NumpyBuffers

    conn, mock_cuda, _ = make_fake_ipc_connection(
        num_slots=num_slots,
        ipc_version=ipc_version,
        write_idx=write_idx,
        dev_ptr_style=dev_ptr_style,
        with_shm_handle=with_shm_handle,
        with_events=with_events,
    )
    fmt = Format.from_overrides(shape, dtype)

    # Resolve the numpy backend: explicit arg wins, then with_numpy auto-build.
    if numpy is None and with_numpy:
        mock_stream = MagicMock()
        numpy = NumpyBuffers(
            cuda=mock_cuda,
            fmt=fmt,
            buffer=np.zeros(shape, dtype=np.dtype(dtype)),
            pinned_ptr=None,
            host_registered_arr=None,
            pinned_memory_available=False,
            primary_stream=mock_stream,
            d2h_streams=[mock_stream],
            num_streams=1,
            chunk_plan=[],
        )

    spec = ImportSpec(shm_name="fake", device=0, timeout_ms=timeout_ms, shape=shape, dtype=dtype)
    policy = ImportPolicy(wait_spin_us=spin_us, allow_pageable_fallback=allow_pageable_fallback, debug=debug)
    # Pass FakeCUDAAdapter explicitly so that _reinitialize (which calls _open_ipc_slots via
    # self._cuda) uses a proper fake rather than the MagicMock stored on conn.cuda.
    return Importer.from_connection(
        spec,
        policy,
        conn,
        fmt,
        cuda=FakeCUDAAdapter(),
        numpy=numpy,
        cupy=cupy,
        last_write_idx=last_write_idx,
    )


def make_bare_runtime_api(*, install_errcheck: bool = False, cudart: object | None = None) -> object:
    """Build a CUDARuntimeAPI without running __init__ (no GPU / DLL load).

    Allocates via __new__, injects a MagicMock cudart (or the supplied one), and
    disables sticky-error checking.  With install_errcheck=True it also runs
    _setup_function_signatures() + _install_errcheck() for errcheck-path tests.

    Args:
        install_errcheck: If True, run _setup_function_signatures() and
                          _install_errcheck() so that errcheck-coverage tests can
                          exercise the full binding path without a real GPU.
        cudart:           An explicit cudart mock to inject.  None (default) →
                          a fresh MagicMock() is created.

    Returns:
        A CUDARuntimeAPI instance whose __init__ was never called.
    """
    from unittest.mock import MagicMock

    from cuda_link.cuda_ipc_wrapper import CUDARuntimeAPI

    api = CUDARuntimeAPI.__new__(CUDARuntimeAPI)
    api.cudart = cudart if cudart is not None else MagicMock()
    api._sticky_check_enabled = False
    if install_errcheck:
        api._setup_function_signatures()
        api._install_errcheck()
    return api
