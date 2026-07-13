"""
GPU-free coverage tests for CUDAGraphsMixin (cuda_graphs.py) — the CUDA Graphs
capture/instantiate/launch/node-update API mixed into CUDARuntimeAPI.

Background
----------
CUDAGraphsMixin methods only touch self.cudart and self.check_error, so a bare
CUDARuntimeAPI built via tests.fakes.make_bare_runtime_api() (object.__new__ +
MagicMock cudart, __init__ never run) exercises them fully without a GPU.

make_memcpy3d_params (the priority region for this module, source lines 198-217)
is a @staticmethod with no cudart dependency at all — it's tested directly against
the exact field layout documented in cuda_runtime_types.cudaMemcpy3DParms.

No @pytest.mark.requires_cuda / requires_native marker needed.
"""

from __future__ import annotations

from ctypes import c_void_p

from cuda_link.cuda_runtime_types import MemcpyKind
from tests.fakes import make_bare_runtime_api

# ---------------------------------------------------------------------------
# make_memcpy3d_params — PRIORITY region (source lines 198-217)
# ---------------------------------------------------------------------------


def test_make_memcpy3d_params_populates_all_fields() -> None:
    """Every field of the returned cudaMemcpy3DParms matches the documented layout."""
    import cuda_link.cuda_ipc_wrapper as _w

    src = c_void_p(0x2000)
    dst = c_void_p(0x3000)
    count = 256
    kind = MemcpyKind.DEVICE_TO_DEVICE

    params = _w.CUDARuntimeAPI.make_memcpy3d_params(dst, src, count, kind)

    assert params.srcArray is None
    assert (params.srcPos.x, params.srcPos.y, params.srcPos.z) == (0, 0, 0)
    assert params.srcPtr.ptr == 0x2000
    assert params.srcPtr.pitch == count
    assert params.srcPtr.xsize == count
    assert params.srcPtr.ysize == 1

    assert params.dstArray is None
    assert (params.dstPos.x, params.dstPos.y, params.dstPos.z) == (0, 0, 0)
    assert params.dstPtr.ptr == 0x3000
    assert params.dstPtr.pitch == count
    assert params.dstPtr.xsize == count
    assert params.dstPtr.ysize == 1

    assert (params.extent.width, params.extent.height, params.extent.depth) == (count, 1, 1)
    assert params.kind == kind


def test_make_memcpy3d_params_is_callable_without_an_instance() -> None:
    """make_memcpy3d_params is a staticmethod — callable via the class, no self needed."""
    from cuda_link.cuda_graphs import CUDAGraphsMixin

    params = CUDAGraphsMixin.make_memcpy3d_params(c_void_p(1), c_void_p(2), 8, MemcpyKind.HOST_TO_DEVICE)
    assert params.kind == MemcpyKind.HOST_TO_DEVICE


# ---------------------------------------------------------------------------
# Graph lifecycle: capture / instantiate / launch / destroy
# ---------------------------------------------------------------------------


def test_stream_begin_capture_forwards_stream_and_mode() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    stream = _w.CUDAStream_t(1)
    api.stream_begin_capture(stream, mode=2)
    args = api.cudart.cudaStreamBeginCapture.call_args[0]
    assert args[0] is stream
    assert args[1].value == 2


def test_stream_end_capture_returns_graph_handle() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    stream = _w.CUDAStream_t(1)
    result = api.stream_end_capture(stream)
    api.cudart.cudaStreamEndCapture.assert_called_once()
    assert isinstance(result, _w.CUDAGraph_t)


def test_graph_instantiate_returns_graph_exec_handle() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph = _w.CUDAGraph_t(1)
    result = api.graph_instantiate(graph, flags=2)
    args = api.cudart.cudaGraphInstantiateWithFlags.call_args[0]
    assert args[1] is graph
    assert args[2].value == 2
    assert isinstance(result, _w.CUDAGraphExec_t)


def test_graph_launch_forwards_exec_and_stream() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph_exec, stream = _w.CUDAGraphExec_t(1), _w.CUDAStream_t(2)
    api.graph_launch(graph_exec, stream)
    api.cudart.cudaGraphLaunch.assert_called_once_with(graph_exec, stream)


def test_graph_get_nodes_two_call_pattern_returns_list() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph = _w.CUDAGraph_t(1)

    result = api.graph_get_nodes(graph)

    assert api.cudart.cudaGraphGetNodes.call_count == 2
    # MagicMock never writes through the byref(count) out-param, so count stays 0
    # and the resulting node array (and returned list) is empty.
    assert result == []


def test_graph_destroy_forwards_graph() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph = _w.CUDAGraph_t(3)
    api.graph_destroy(graph)
    api.cudart.cudaGraphDestroy.assert_called_once_with(graph)


def test_graph_exec_destroy_forwards_graph_exec() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph_exec = _w.CUDAGraphExec_t(4)
    api.graph_exec_destroy(graph_exec)
    api.cudart.cudaGraphExecDestroy.assert_called_once_with(graph_exec)


# ---------------------------------------------------------------------------
# Per-frame node-param setters
# ---------------------------------------------------------------------------


def test_graph_exec_memcpy_node_set_params_builds_params_and_forwards() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph_exec, node = _w.CUDAGraphExec_t(1), _w.CUDAGraphNode_t(2)
    dst, src = c_void_p(0x1000), c_void_p(0x2000)

    api.graph_exec_memcpy_node_set_params(graph_exec, node, dst, src, 128, MemcpyKind.DEVICE_TO_DEVICE)

    api.cudart.cudaGraphExecMemcpyNodeSetParams.assert_called_once()
    call_args = api.cudart.cudaGraphExecMemcpyNodeSetParams.call_args[0]
    assert call_args[0] is graph_exec
    assert call_args[1] is node


def test_graph_exec_memcpy_node_set_params_1d_converts_pointers_to_raw_ints() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph_exec, node = _w.CUDAGraphExec_t(1), _w.CUDAGraphNode_t(2)
    dst, src = c_void_p(0x4000), c_void_p(0x5000)

    api.graph_exec_memcpy_node_set_params_1d(graph_exec, node, dst, src, 64, MemcpyKind.HOST_TO_DEVICE)

    args = api.cudart.cudaGraphExecMemcpyNodeSetParams1D.call_args[0]
    assert args[0] is graph_exec
    assert args[1] is node
    assert args[2].value == 0x4000
    assert args[3].value == 0x5000
    assert args[4].value == 64
    assert args[5].value == MemcpyKind.HOST_TO_DEVICE


def test_graph_exec_memcpy_node_set_params_1d_handles_null_pointers() -> None:
    """A NULL c_void_p's .value is None; the `dst.value or 0` fallback avoids a TypeError,
    but re-wrapping the resulting 0 in c_void_p(0) still reports .value as None (ctypes
    NULL pointers always read back as None, never as the int 0) — so the observable
    effect is simply that no exception is raised for NULL src/dst pointers.
    """
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph_exec, node = _w.CUDAGraphExec_t(1), _w.CUDAGraphNode_t(2)
    null_dst, null_src = c_void_p(0), c_void_p(0)

    api.graph_exec_memcpy_node_set_params_1d(graph_exec, node, null_dst, null_src, 1, MemcpyKind.HOST_TO_HOST)

    args = api.cudart.cudaGraphExecMemcpyNodeSetParams1D.call_args[0]
    assert args[2].value is None
    assert args[3].value is None


def test_graph_exec_event_record_node_set_event_forwards_args() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph_exec, node, event = _w.CUDAGraphExec_t(1), _w.CUDAGraphNode_t(2), _w.CUDAEvent_t(3)
    api.graph_exec_event_record_node_set_event(graph_exec, node, event)
    api.cudart.cudaGraphExecEventRecordNodeSetEvent.assert_called_once_with(graph_exec, node, event)


def test_graph_exec_event_wait_node_set_event_forwards_args() -> None:
    import cuda_link.cuda_ipc_wrapper as _w

    api = make_bare_runtime_api()
    graph_exec, node, event = _w.CUDAGraphExec_t(1), _w.CUDAGraphNode_t(2), _w.CUDAEvent_t(3)
    api.graph_exec_event_wait_node_set_event(graph_exec, node, event)
    api.cudart.cudaGraphExecEventWaitNodeSetEvent.assert_called_once_with(graph_exec, node, event)


# ---------------------------------------------------------------------------
# get_runtime_version
# ---------------------------------------------------------------------------


def test_get_runtime_version_returns_int() -> None:
    api = make_bare_runtime_api()
    result = api.get_runtime_version()
    api.cudart.cudaRuntimeGetVersion.assert_called_once()
    assert isinstance(result, int)
