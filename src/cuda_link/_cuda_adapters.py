"""
CUDA adapters — concrete implementations of CudaPort and ImporterCudaPort.

CTypesCudaAdapter  — production adapter; thin delegation to CUDARuntimeAPI.
FakeCudaAdapter    — in-memory adapter; no GPU required; for unit tests.

Both classes satisfy the CudaPort Protocol (exporter side) and the
ImporterCudaPort Protocol (importer side) structurally — the same two classes
serve both sides without any code duplication.
"""

from __future__ import annotations

import ctypes
from ctypes import c_void_p
from typing import Any

from .cuda_ipc_wrapper import CUDARuntimeAPI
from .cuda_runtime_types import (
    CUDAEvent_t,
    CUDAGraph_t,
    CUDAGraphExec_t,
    CUDAGraphNode_t,
    CUDAStream_t,
    cudaIpcEventHandle_t,
    cudaIpcMemHandle_t,
)

# ---------------------------------------------------------------------------
# Production adapter
# ---------------------------------------------------------------------------


class CTypesCudaAdapter:
    """CudaPort / ImporterCudaPort backed by a real CUDARuntimeAPI instance.

    This adapter is a one-to-one delegation layer — every method calls the
    identically-named method on the underlying CUDARuntimeAPI. Its value is that
    it satisfies the CudaPort and ImporterCudaPort Protocols without exposing the
    full CUDARuntimeAPI surface (which includes methods neither side needs).

    Construction:
        adapter = CTypesCudaAdapter.for_device(device=0)
        # or, when a singleton is already loaded:
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
        adapter = CTypesCudaAdapter(get_cuda_runtime(device=0))
    """

    def __init__(self, api: CUDARuntimeAPI) -> None:
        self._api = api

    @classmethod
    def for_device(cls, device: int = 0) -> CTypesCudaAdapter:
        """Construct a production adapter bound to the given CUDA device."""
        from .cuda_ipc_wrapper import get_cuda_runtime

        return cls(get_cuda_runtime(device=device))

    # --- Device ------------------------------------------------------------

    def get_device(self) -> int:
        return self._api.get_device()

    def set_device(self, device: int) -> int:
        return self._api.set_device(device)

    def restore_context(self, token: int) -> None:
        self._api.restore_context(token)

    def peek_last_error(self) -> int:
        return self._api.peek_at_last_error()

    # --- Memory (device) ---------------------------------------------------

    def malloc(self, size: int) -> c_void_p:
        return self._api.malloc(size)

    def free(self, dev_ptr: c_void_p) -> None:
        self._api.free(dev_ptr)

    def memcpy_async(
        self,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
        stream: CUDAStream_t,
    ) -> None:
        self._api.memcpy_async(dst, src, count, kind, stream)

    # --- Memory (host / pinned) --------------------------------------------

    def malloc_host_alloc(self, size: int, flags: int = 0x01) -> c_void_p:
        return self._api.malloc_host_alloc(size, flags)

    def free_host(self, ptr: c_void_p) -> None:
        self._api.free_host(ptr)

    def host_register(self, ptr: int, size: int, flags: int = 0) -> None:
        self._api.host_register(ptr, size, flags)

    def host_unregister(self, ptr: int) -> None:
        self._api.host_unregister(ptr)

    # --- Streams -----------------------------------------------------------

    def create_stream(self, flags: int = 0x01) -> CUDAStream_t:
        return self._api.create_stream(flags)

    def create_stream_with_priority(self, flags: int = 0x01, priority: int | None = None) -> CUDAStream_t:
        return self._api.create_stream_with_priority(flags, priority)

    def destroy_stream(self, stream: CUDAStream_t) -> None:
        self._api.destroy_stream(stream)

    def stream_wait_event(self, stream: CUDAStream_t, event: CUDAEvent_t, flags: int = 0) -> None:
        self._api.stream_wait_event(stream, event, flags)

    def stream_synchronize(self, stream: CUDAStream_t) -> None:
        self._api.stream_synchronize(stream)

    def stream_query(self, stream: CUDAStream_t) -> bool:
        return self._api.stream_query(stream)

    def synchronize(self) -> None:
        self._api.synchronize()

    # --- Events ------------------------------------------------------------

    def create_ipc_event(self) -> CUDAEvent_t:
        return self._api.create_ipc_event()

    def create_sync_event(self) -> CUDAEvent_t:
        return self._api.create_sync_event()

    def ipc_get_event_handle(self, event: CUDAEvent_t) -> cudaIpcEventHandle_t:
        return self._api.ipc_get_event_handle(event)

    def record_event(self, event: CUDAEvent_t, stream: CUDAStream_t | None = None) -> None:
        self._api.record_event(event, stream)

    def destroy_event(self, event: CUDAEvent_t) -> None:
        self._api.destroy_event(event)

    def query_event(self, event: CUDAEvent_t) -> bool:
        return self._api.query_event(event)

    # --- IPC memory --------------------------------------------------------

    def ipc_get_mem_handle(self, dev_ptr: c_void_p) -> cudaIpcMemHandle_t:
        return self._api.ipc_get_mem_handle(dev_ptr)

    def ipc_open_mem_handle(self, handle: cudaIpcMemHandle_t, flags: int = 1) -> c_void_p:
        return self._api.ipc_open_mem_handle(handle, flags)

    def ipc_close_mem_handle(self, dev_ptr: c_void_p) -> None:
        self._api.ipc_close_mem_handle(dev_ptr)

    def ipc_open_event_handle(self, handle: cudaIpcEventHandle_t) -> CUDAEvent_t:
        return self._api.ipc_open_event_handle(handle)

    # --- Pointer attributes ------------------------------------------------

    def pointer_get_attributes(self, ptr: int) -> Any:
        return self._api.pointer_get_attributes(ptr)

    # --- Error checking ----------------------------------------------------

    def check_sticky_error(self, context: str) -> None:
        self._api.check_sticky_error(context)

    # --- CUDA Graphs -------------------------------------------------------

    def get_runtime_version(self) -> int:
        return self._api.get_runtime_version()

    def stream_begin_capture(self, stream: CUDAStream_t, mode: int = 0) -> None:
        self._api.stream_begin_capture(stream, mode)

    def stream_end_capture(self, stream: CUDAStream_t) -> CUDAGraph_t:
        return self._api.stream_end_capture(stream)

    def graph_instantiate(self, graph: CUDAGraph_t, flags: int = 0) -> CUDAGraphExec_t:
        return self._api.graph_instantiate(graph, flags)

    def graph_launch(self, graph_exec: CUDAGraphExec_t, stream: CUDAStream_t) -> None:
        self._api.graph_launch(graph_exec, stream)

    def graph_destroy(self, graph: CUDAGraph_t) -> None:
        self._api.graph_destroy(graph)

    def graph_exec_destroy(self, graph_exec: CUDAGraphExec_t) -> None:
        self._api.graph_exec_destroy(graph_exec)

    def graph_get_nodes(self, graph: CUDAGraph_t) -> list[CUDAGraphNode_t]:
        return self._api.graph_get_nodes(graph)

    def graph_exec_memcpy_node_set_params_1d(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
    ) -> None:
        self._api.graph_exec_memcpy_node_set_params_1d(graph_exec, node, dst, src, count, kind)


# ---------------------------------------------------------------------------
# Test adapter — helpers
# ---------------------------------------------------------------------------


class _FakePointerAttributes:
    """Minimal stand-in for cudaPointerAttributes in the test fake."""

    def __init__(self, type: int = 2, device: int = 0) -> None:
        self.type = type  # 2 = cudaMemoryTypeDevice
        self.device = device


class _FakeHandle:
    """Opaque handle sentinel used by FakeCudaAdapter for events, streams, etc."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def __repr__(self) -> str:
        return f"<FakeHandle {self._tag}>"


class _FakeIpcHandle:
    """64-byte IPC handle sentinel (cudaIpcMemHandle_t / cudaIpcEventHandle_t shape).

    Both IPC handle types are 64 bytes but use different field names:
      cudaIpcMemHandle_t   → .internal  (c_byte * 64)
      cudaIpcEventHandle_t → .reserved  (c_byte * 64)
    This fake satisfies both so a single class covers both call sites.
    """

    internal: bytes = bytes(64)
    reserved: bytes = bytes(64)

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def __repr__(self) -> str:
        return f"<FakeIpcHandle {self._tag}>"


# ---------------------------------------------------------------------------
# Test adapter
# ---------------------------------------------------------------------------


class FakeCudaAdapter:
    """In-memory CudaPort / ImporterCudaPort for unit tests — no GPU, no ctypes DLL.

    Satisfies both CudaPort (exporter) and ImporterCudaPort (importer) structurally.

    Allocation tracking:
      - malloc() records allocations in self.allocations (ptr_int → size).
      - free() removes the entry and appends to self.freed.
      - After the test, assert len(adapter.allocations) == 0 to detect leaks.

    Importer-side IPC fakes:
      - ipc_open_mem_handle() records the handle in self.opened_mem_handles and
        returns a fake c_void_p pointer.
      - ipc_close_mem_handle() removes the entry from self.opened_mem_handles.
      - ipc_open_event_handle() returns a _FakeHandle sentinel.
      - query_event() always returns True (event immediately ready).
      - synchronize() is a no-op.
      - malloc_host_alloc() allocates a real ctypes buffer (no real DMA pinning).
      - free_host() frees it.
      - host_register() / host_unregister() are no-ops.

    Failure injection:
      - Set fail_on_malloc_count = N to make the Nth malloc() call raise.
      - Set fail_on_stream_create = True to make create_stream / _with_priority raise.
      - Set fail_on_event_create = True to make create_ipc_event raise.

    Graph simulation:
      - graph_instantiate / graph_launch are no-ops (returns _FakeHandle).
      - graph_exec_memcpy_node_set_params_1d is a no-op.

    All handle objects are _FakeHandle instances — they satisfy isinstance checks
    but carry no CUDA state.
    """

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self.allocations: dict[int, int] = {}  # ptr int → size
        self.freed: list[int] = []
        self._next_ptr = 0x1000_0000

        # Failure injection knobs
        self.fail_on_malloc_count: int | None = None
        self.fail_on_stream_create: bool = False
        self.fail_on_event_create: bool = False
        self._malloc_call_count = 0

        # Sticky-error simulation
        self._sticky_error: int = 0

        # Recorded events: list of (event_tag, stream_tag) tuples
        self.recorded_events: list[tuple[str, str]] = []

        # IPC tracking (importer-side)
        self.opened_mem_handles: dict[int, Any] = {}  # fake_ptr_int → handle

        # Host pinned memory tracking (importer-side)
        self._host_allocs: dict[int, ctypes.Array] = {}  # ptr_int → ctypes buf

    def _alloc_ptr(self, size: int) -> int:
        ptr_int = self._next_ptr
        self._next_ptr += max(size, 4096)
        return ptr_int

    # --- Device ------------------------------------------------------------

    def get_device(self) -> int:
        return self.device

    def set_device(self, device: int) -> int:
        return 0  # no-op in fake; real impl saves/restores the driver-API context

    def restore_context(self, token: int) -> None:
        pass  # no-op in fake

    def peek_last_error(self) -> int:
        return self._sticky_error

    # --- Memory (device) ---------------------------------------------------

    def malloc(self, size: int) -> c_void_p:
        self._malloc_call_count += 1
        if self.fail_on_malloc_count is not None and self._malloc_call_count >= self.fail_on_malloc_count:
            raise RuntimeError(f"FakeCudaAdapter: injected malloc failure on call {self._malloc_call_count}")
        ptr_int = self._alloc_ptr(size)
        self.allocations[ptr_int] = size
        return c_void_p(ptr_int)

    def free(self, dev_ptr: c_void_p) -> None:
        ptr_int = dev_ptr.value if isinstance(dev_ptr, c_void_p) else int(dev_ptr)
        self.allocations.pop(ptr_int, None)
        self.freed.append(ptr_int)

    def memcpy_async(
        self,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
        stream: Any,
    ) -> None:
        pass  # no-op in tests

    # --- Memory (host / pinned) --------------------------------------------

    def malloc_host_alloc(self, size: int, flags: int = 0x01) -> c_void_p:
        buf = (ctypes.c_ubyte * size)()
        ptr_int = ctypes.addressof(buf)
        self._host_allocs[ptr_int] = buf
        return c_void_p(ptr_int)

    def free_host(self, ptr: c_void_p) -> None:
        ptr_int = ptr.value if isinstance(ptr, c_void_p) else int(ptr)
        self._host_allocs.pop(ptr_int, None)

    def host_register(self, ptr: int, size: int, flags: int = 0) -> None:
        pass  # no-op in tests

    def host_unregister(self, ptr: int) -> None:
        pass  # no-op in tests

    # --- Streams -----------------------------------------------------------

    def create_stream(self, flags: int = 0x01) -> Any:
        if self.fail_on_stream_create:
            raise RuntimeError("FakeCudaAdapter: injected stream creation failure")
        return _FakeHandle(f"stream:flags={flags:#x}")

    def create_stream_with_priority(self, flags: int = 0x01, priority: int | None = None) -> Any:
        if self.fail_on_stream_create:
            raise RuntimeError("FakeCudaAdapter: injected stream creation failure")
        return _FakeHandle(f"stream:prio={priority}:flags={flags:#x}")

    def destroy_stream(self, stream: Any) -> None:
        pass

    def stream_wait_event(self, stream: Any, event: Any, flags: int = 0) -> None:
        pass

    def stream_synchronize(self, stream: Any) -> None:
        pass

    def stream_query(self, stream: Any) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    # --- Events ------------------------------------------------------------

    def create_ipc_event(self) -> Any:
        if self.fail_on_event_create:
            raise RuntimeError("FakeCudaAdapter: injected event creation failure")
        return _FakeHandle("ipc_event")

    def create_sync_event(self) -> Any:
        return _FakeHandle("sync_event")

    def ipc_get_event_handle(self, event: Any) -> Any:
        return _FakeIpcHandle("ipc_event_handle")

    def record_event(self, event: Any, stream: Any = None) -> None:
        event_tag = getattr(event, "_tag", str(event))
        stream_tag = getattr(stream, "_tag", str(stream)) if stream is not None else "default"
        self.recorded_events.append((event_tag, stream_tag))

    def destroy_event(self, event: Any) -> None:
        pass

    def query_event(self, event: Any) -> bool:
        return True  # events are always immediately ready in tests

    # --- IPC memory --------------------------------------------------------

    def ipc_get_mem_handle(self, dev_ptr: Any) -> Any:
        return _FakeIpcHandle("ipc_mem_handle")

    def ipc_open_mem_handle(self, handle: Any, flags: int = 1) -> c_void_p:
        ptr_int = self._alloc_ptr(4096)
        self.opened_mem_handles[ptr_int] = handle
        return c_void_p(ptr_int)

    def ipc_close_mem_handle(self, dev_ptr: c_void_p) -> None:
        ptr_int = dev_ptr.value if isinstance(dev_ptr, c_void_p) else int(dev_ptr)
        self.opened_mem_handles.pop(ptr_int, None)

    def ipc_open_event_handle(self, handle: Any) -> Any:
        return _FakeHandle("ipc_event_from_handle")

    # --- Pointer attributes ------------------------------------------------

    def pointer_get_attributes(self, ptr: int) -> _FakePointerAttributes:
        return _FakePointerAttributes(type=2, device=self.device)

    # --- Error checking ----------------------------------------------------

    def check_sticky_error(self, context: str) -> None:
        if self._sticky_error != 0:
            raise RuntimeError(f"FakeCudaAdapter: sticky error {self._sticky_error} after {context}")

    # --- CUDA Graphs -------------------------------------------------------

    def get_runtime_version(self) -> int:
        return 12080  # simulate CUDA 12.8

    def stream_begin_capture(self, stream: Any, mode: int = 0) -> None:
        pass

    def stream_end_capture(self, stream: Any) -> Any:
        return _FakeHandle("graph_template")

    def graph_instantiate(self, graph: Any, flags: int = 0) -> Any:
        return _FakeHandle("graph_exec")

    def graph_launch(self, graph_exec: Any, stream: Any) -> None:
        pass

    def graph_destroy(self, graph: Any) -> None:
        pass

    def graph_exec_destroy(self, graph_exec: Any) -> None:
        pass

    def graph_get_nodes(self, graph: Any) -> list:
        return [_FakeHandle("memcpy_node")]

    def graph_exec_memcpy_node_set_params_1d(
        self,
        graph_exec: Any,
        node: Any,
        dst: Any,
        src: Any,
        count: int,
        kind: int,
    ) -> None:
        pass
