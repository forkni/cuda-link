"""
CUDA Graphs Mixin — CUDA Graph capture, instantiation, launch, and node-update methods.

Provides CUDAGraphsMixin, mixed into CUDARuntimeAPI to contribute the graph-lifecycle
API. All methods rely on self.cudart (the cudart DLL handle) and self.check_error from
the host class.

Shared between the pip package (cuda_link) and TouchDesigner textDATs.
Compatible with both Python package and TD COMP namespace imports.
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_int, c_size_t, c_void_p

try:
    from cuda_link.cuda_runtime_types import (  # noqa: E402
        CUDAEvent_t,
        CUDAGraph_t,
        CUDAGraphExec_t,
        CUDAGraphNode_t,
        CUDAStream_t,
        cudaExtent,
        cudaMemcpy3DParms,
        cudaPitchedPtr,
        cudaPos,
    )
except ImportError:
    from CUDARuntimeTypes import (  # type: ignore[no-redef]  # noqa: E402
        CUDAEvent_t,
        CUDAGraph_t,
        CUDAGraphExec_t,
        CUDAGraphNode_t,
        CUDAStream_t,
        cudaExtent,
        cudaMemcpy3DParms,
        cudaPitchedPtr,
        cudaPos,
    )


class CUDAGraphsMixin:
    """Mixin contributing CUDA Graph lifecycle methods to CUDARuntimeAPI.

    Requires self.cudart (cudart DLL handle) and self.check_error from the host class.
    """

    # --- Phase 2: CUDA Graph API wrappers ---

    def stream_begin_capture(self, stream: CUDAStream_t, mode: int = 0) -> None:
        """Begin capturing a stream into a CUDA graph.

        After this call, operations enqueued on *stream* are recorded into a
        graph rather than executed immediately. End with stream_end_capture().

        Args:
            stream: Stream to capture.
            mode:   cudaStreamCaptureMode — 0=global (safest), 1=thread_local,
                    2=relaxed. Use 0 unless you know what you're doing.

        Raises:
            RuntimeError: If capture start fails (e.g., stream already capturing).
        """
        result = self.cudart.cudaStreamBeginCapture(stream, c_int(mode))
        self.check_error(result, "cudaStreamBeginCapture")

    def stream_end_capture(self, stream: CUDAStream_t) -> CUDAGraph_t:
        """End stream capture and return the captured graph.

        After this call the stream resumes normal execution mode. The returned
        graph must be instantiated with graph_instantiate() before use, and
        destroyed with graph_destroy() when done.

        Args:
            stream: Stream that was passed to stream_begin_capture().

        Returns:
            CUDAGraph_t handle to the captured graph template.

        Raises:
            RuntimeError: If capture end fails.
        """
        graph = CUDAGraph_t()
        result = self.cudart.cudaStreamEndCapture(stream, byref(graph))
        self.check_error(result, "cudaStreamEndCapture")
        return graph

    def graph_instantiate(self, graph: CUDAGraph_t, flags: int = 0) -> CUDAGraphExec_t:
        """Instantiate a graph template into an executable graph.

        The executable graph (CUDAGraphExec_t) can be launched repeatedly via
        graph_launch(). The template graph can be destroyed after instantiation.

        Args:
            graph:  CUDAGraph_t template returned by stream_end_capture().
            flags:  cudaGraphInstantiateFlagDeviceLaunch (0x02) for device-side
                    launch; 0 for normal host-side launch.

        Returns:
            CUDAGraphExec_t executable graph handle.

        Raises:
            RuntimeError: If instantiation fails.
        """
        from ctypes import c_uint64

        graph_exec = CUDAGraphExec_t()
        result = self.cudart.cudaGraphInstantiateWithFlags(byref(graph_exec), graph, c_uint64(flags))
        self.check_error(result, "cudaGraphInstantiateWithFlags")
        return graph_exec

    def graph_launch(self, graph_exec: CUDAGraphExec_t, stream: CUDAStream_t) -> None:
        """Launch an executable graph on a stream (single WDDM submission).

        This replaces N individual API calls (stream_wait_event, memcpy_async,
        record_event) with one batched WDDM submission, reducing kernel-mode
        transition overhead from N×~15µs to ~15µs on Windows WDDM.

        Args:
            graph_exec: Executable graph from graph_instantiate().
            stream:     Stream on which to launch the graph.

        Raises:
            RuntimeError: If launch fails.
        """
        result = self.cudart.cudaGraphLaunch(graph_exec, stream)
        self.check_error(result, "cudaGraphLaunch")

    def graph_get_nodes(self, graph: CUDAGraph_t) -> list[CUDAGraphNode_t]:
        """Return all nodes in a graph in topological (capture) order.

        Useful for discovering node handles after stream capture, before the
        template graph is destroyed.

        Args:
            graph: CUDAGraph_t template (must NOT yet be destroyed).

        Returns:
            List of CUDAGraphNode_t handles in capture order:
            [EventWaitNode (if present), MemcpyNode, EventRecordNode].

        Raises:
            RuntimeError: If query fails.
        """
        count = c_size_t(0)
        result = self.cudart.cudaGraphGetNodes(graph, None, byref(count))
        self.check_error(result, "cudaGraphGetNodes (count)")
        node_array = (CUDAGraphNode_t * count.value)()
        result = self.cudart.cudaGraphGetNodes(graph, node_array, byref(count))
        self.check_error(result, "cudaGraphGetNodes (fill)")
        return list(node_array)

    def graph_destroy(self, graph: CUDAGraph_t) -> None:
        """Destroy a graph template (not the executable — use graph_exec_destroy for that).

        Args:
            graph: Template graph to destroy.

        Raises:
            RuntimeError: If destruction fails.
        """
        result = self.cudart.cudaGraphDestroy(graph)
        self.check_error(result, "cudaGraphDestroy")

    def graph_exec_destroy(self, graph_exec: CUDAGraphExec_t) -> None:
        """Destroy an executable graph and free its resources.

        Args:
            graph_exec: Executable graph to destroy.

        Raises:
            RuntimeError: If destruction fails.
        """
        result = self.cudart.cudaGraphExecDestroy(graph_exec)
        self.check_error(result, "cudaGraphExecDestroy")

    @staticmethod
    def make_memcpy3d_params(dst: c_void_p, src: c_void_p, count: int, kind: int) -> cudaMemcpy3DParms:
        """Build a cudaMemcpy3DParms struct for a flat 1D memory copy.

        Represents the copy as a single-row 2D memcpy (height=1, depth=1) so
        that 'count' bytes are transferred from src to dst. This is the required
        form for cudaGraphExecMemcpyNodeSetParams even when the original copy was
        issued as cudaMemcpyAsync (1D form).

        Args:
            dst:   Destination pointer.
            src:   Source pointer.
            count: Number of bytes to copy.
            kind:  cudaMemcpyKind (3 = DeviceToDevice).

        Returns:
            Populated cudaMemcpy3DParms instance.
        """
        params = cudaMemcpy3DParms()
        params.srcArray = None
        params.srcPos = cudaPos(0, 0, 0)
        params.srcPtr = cudaPitchedPtr(
            ptr=ctypes.cast(src, c_void_p),
            pitch=count,
            xsize=count,
            ysize=1,
        )
        params.dstArray = None
        params.dstPos = cudaPos(0, 0, 0)
        params.dstPtr = cudaPitchedPtr(
            ptr=ctypes.cast(dst, c_void_p),
            pitch=count,
            xsize=count,
            ysize=1,
        )
        params.extent = cudaExtent(width=count, height=1, depth=1)
        params.kind = kind
        return params

    def graph_exec_memcpy_node_set_params(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
    ) -> None:
        """Update src/dst/count/kind of a memcpy node in an executable graph.

        This is a CPU-only operation (no WDDM submission). Changes take effect
        on the next graph_launch() call. The extent (count) must match the
        extent used when the graph was captured — only pointer reassignment
        within the same buffer size is supported.

        Args:
            graph_exec: Executable graph containing the node.
            node:       MemcpyNode handle from graph_get_nodes().
            dst:        New destination pointer.
            src:        New source pointer.
            count:      Copy size in bytes (must match captured size).
            kind:       cudaMemcpyKind (must match captured kind).

        Raises:
            RuntimeError: If parameter update fails.
        """
        params = self.make_memcpy3d_params(dst, src, count, kind)
        result = self.cudart.cudaGraphExecMemcpyNodeSetParams(graph_exec, node, byref(params))
        self.check_error(result, "cudaGraphExecMemcpyNodeSetParams")

    def graph_exec_memcpy_node_set_params_1d(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
    ) -> None:
        """Update src/dst/count/kind of a 1D memcpy node in an executable graph.

        Use this for nodes captured from cudaMemcpyAsync (1D form). The 3D variant
        (graph_exec_memcpy_node_set_params) returns INVALID_VALUE on 1D nodes.
        Requires CUDA 11.3+.
        """
        dst_int = dst.value if isinstance(dst, c_void_p) else int(dst)
        src_int = src.value if isinstance(src, c_void_p) else int(src)
        result = self.cudart.cudaGraphExecMemcpyNodeSetParams1D(
            graph_exec,
            node,
            c_void_p(dst_int),
            c_void_p(src_int),
            c_size_t(count),
            c_int(kind),
        )
        self.check_error(result, "cudaGraphExecMemcpyNodeSetParams1D")

    def graph_exec_event_record_node_set_event(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        event: CUDAEvent_t,
    ) -> None:
        """Update the event recorded by an event-record node. CUDA 11.4+.

        CPU-only — takes effect on next graph_launch(). Use this to update the
        per-ring-slot IPC event when the ring slot changes between launches.

        Args:
            graph_exec: Executable graph containing the node.
            node:       EventRecordNode handle from graph_get_nodes().
            event:      New CUDAEvent_t to record.

        Raises:
            RuntimeError: If update fails.
        """
        result = self.cudart.cudaGraphExecEventRecordNodeSetEvent(graph_exec, node, event)
        self.check_error(result, "cudaGraphExecEventRecordNodeSetEvent")

    def graph_exec_event_wait_node_set_event(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        event: CUDAEvent_t,
    ) -> None:
        """Update the event waited on by an event-wait node. CUDA 11.4+.

        Args:
            graph_exec: Executable graph containing the node.
            node:       EventWaitNode handle from graph_get_nodes().
            event:      New CUDAEvent_t to wait on.

        Raises:
            RuntimeError: If update fails.
        """
        result = self.cudart.cudaGraphExecEventWaitNodeSetEvent(graph_exec, node, event)
        self.check_error(result, "cudaGraphExecEventWaitNodeSetEvent")

    def get_runtime_version(self) -> int:
        """Return the CUDA runtime version as an int.

        Examples: 11030 = CUDA 11.3, 11040 = CUDA 11.4, 12080 = CUDA 12.8.
        Used to gate optional API calls when the loaded cudart DLL may be from
        an older patch level (e.g., TouchDesigner ships ``cudart64_110.dll``).
        """
        version = c_int(0)
        result = self.cudart.cudaRuntimeGetVersion(byref(version))
        self.check_error(result, "cudaRuntimeGetVersion")
        return int(version.value)
