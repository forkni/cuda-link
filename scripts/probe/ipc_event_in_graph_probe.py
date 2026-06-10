"""
ipc_event_in_graph_probe.py - Validate IPC-event-in-CUDA-graph on this hardware.

Tests whether cudaEventRecord on a cudaEventInterprocess event is capturable
inside a CUDA graph, and whether the resulting event-record node fires the IPC
event correctly when the graph is launched.

This probe MUST pass on the target machine before implementing P3 (graph event
consolidation) in exporter.py, because IPC-event-in-captured-graph is not
officially blessed by the CUDA spec.

Usage:
    python scripts/probe/ipc_event_in_graph_probe.py [--device N]

Exit codes:
    0 - probe passed; P3 is safe to implement on this hardware.
    1 - probe failed; keep record_event outside the graph (current behaviour).
    2 - CUDA not available; skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import sys

# cudaEventRecordExternal — flag for cudaEventRecordWithFlags that records the
# event as an external node during stream capture (CUDA Runtime API).
CUDA_EVENT_RECORD_EXTERNAL = 0x01
# cudaEventWaitExternal — flag for cudaStreamWaitEvent that captures a wait on an
# event recorded OUTSIDE the capture (e.g. the producer-stream source-sync event)
# as an external event-wait node, rather than failing the capture.
CUDA_EVENT_WAIT_EXTERNAL = 0x01


def run_probe(device: int = 0) -> bool:
    """Return True iff IPC-event-in-graph is supported on *device*."""
    try:
        from cuda_link.cuda_ipc_wrapper import get_cuda_runtime
    except ImportError:
        print("[SKIP] cuda_link not importable - install the package first.", file=sys.stderr)
        sys.exit(2)

    cuda = get_cuda_runtime(device=device)
    rt_ver = cuda.get_runtime_version()
    print(f"CUDA runtime version : {rt_ver // 1000}.{(rt_ver % 1000) // 10}")
    print(f"Device               : {device}")

    stream = cuda.create_stream()
    ipc_event = cuda.create_ipc_event()
    # Acquire the handle now - before capture - to confirm the event is IPC-capable.
    ipc_handle = cuda.ipc_get_event_handle(ipc_event)
    print(f"IPC event handle     : {len(ipc_handle.reserved)} bytes (OK)")

    # P3 records the IPC event INSIDE the captured graph using
    # cudaEventRecordWithFlags(..., cudaEventRecordExternal).  Plain cudaEventRecord
    # during capture is rejected with cudaErrorStreamCaptureUnsupported (900): the
    # External flag is precisely what turns the in-capture record into a legal
    # external event-record node.  The wrapper does not bind this entry point yet
    # (it would be the first thing P3 adds), so bind it locally on the cudart handle
    # for the probe, reusing cudaEventRecord's (event, stream) argtypes.
    cudart = cuda.cudart
    _record_wf = cudart.cudaEventRecordWithFlags
    _record_wf.argtypes = [*cudart.cudaEventRecord.argtypes, ctypes.c_uint]
    _record_wf.restype = ctypes.c_int

    def _record_external(event: object, strm: object) -> None:
        rc = _record_wf(event, strm, CUDA_EVENT_RECORD_EXTERNAL)
        if rc != 0:
            raise RuntimeError(f"cudaEventRecordWithFlags(External) failed: code {rc}")

    # --- Attempt capture with IPC event-record node ---------------------------
    print(
        "\n[1/4] Capturing graph: stream_begin_capture -> "
        "cudaEventRecordWithFlags(ipc_event, External) -> stream_end_capture ..."
    )
    capture_started = False
    graph = None
    try:
        cuda.stream_begin_capture(stream, mode=2)  # Relaxed mode
        capture_started = True
        _record_external(ipc_event, stream)
        graph = cuda.stream_end_capture(stream)
        capture_started = False
        print("       Capture: OK")
    except (RuntimeError, OSError) as exc:
        if capture_started:
            try:
                abandoned = cuda.stream_end_capture(stream)
                cuda.graph_destroy(abandoned)
            except (RuntimeError, OSError):
                pass
        print(f"       Capture: FAIL - {exc}")
        cuda.destroy_event(ipc_event)
        cuda.destroy_stream(stream)
        return False

    nodes = cuda.graph_get_nodes(graph)
    print(f"       Graph nodes  : {len(nodes)} (expect 1 EventRecordNode)")

    # --- Instantiate and launch -----------------------------------------------
    print("\n[2/4] Instantiating and launching the graph ...")
    try:
        graph_exec = cuda.graph_instantiate(graph)
        cuda.graph_launch(graph_exec, stream)
        cuda.stream_synchronize(stream)
        print("       Launch+sync  : OK")
    except (RuntimeError, OSError) as exc:
        print(f"       Launch       : FAIL - {exc}")
        cuda.graph_destroy(graph)
        cuda.destroy_event(ipc_event)
        cuda.destroy_stream(stream)
        return False

    event_ready = cuda.query_event(ipc_event)
    print(f"       query_event  : {event_ready} (expect True)")

    # --- Verify graph_exec_event_record_node_set_event works ------------------
    print("\n[3/4] Verify graph_exec_event_record_node_set_event (CPU-only update) ...")
    sync_event = cuda.create_sync_event()
    try:
        cuda.graph_exec_event_record_node_set_event(graph_exec, nodes[0], sync_event)
        cuda.graph_launch(graph_exec, stream)
        cuda.stream_synchronize(stream)
        set_event_ok = cuda.query_event(sync_event)
        print(f"       node_set_event: {set_event_ok} (expect True)")
    except (RuntimeError, OSError) as exc:
        print(f"       node_set_event: FAIL - {exc}")
        set_event_ok = False
    finally:
        cuda.destroy_event(sync_event)

    cuda.graph_exec_destroy(graph_exec)
    cuda.graph_destroy(graph)

    # --- [4/4] The FULL P3 graph shape ----------------------------------------
    # external-wait(source_sync) -> memcpy(D2D) -> external-record(ipc_event).
    # This is what _build_export_graphs will actually capture.  The wait node is
    # the unproven piece: source_sync is recorded on a SEPARATE (producer) stream
    # outside the capture, so the wait must use cudaEventWaitExternal or capture
    # fails.  Without this stage passing, only the record node is safe to fold in.
    print("\n[4/4] Full P3 graph: external-wait(source_sync) -> memcpy -> external-record(ipc_event) ...")
    full_graph_ok = False
    size = 256 * 1024  # 256 KiB scratch D2D copy
    src_buf = dst_buf = None
    producer_stream = None
    source_sync = cuda.create_sync_event()
    capture_started = False
    try:
        src_buf = cuda.malloc(size)
        dst_buf = cuda.malloc(size)
        producer_stream = cuda.create_stream()
        # Record source_sync on the producer stream — OUTSIDE any capture.
        cuda.record_event(source_sync, stream=producer_stream)
        cuda.stream_synchronize(producer_stream)

        cuda.stream_begin_capture(stream, mode=2)
        capture_started = True
        cuda.stream_wait_event(stream, source_sync, CUDA_EVENT_WAIT_EXTERNAL)
        cuda.memcpy_async(dst=dst_buf, src=src_buf, count=size, kind=3, stream=stream)
        _record_external(ipc_event, stream)
        full_graph = cuda.stream_end_capture(stream)
        capture_started = False

        full_nodes = cuda.graph_get_nodes(full_graph)
        print(f"       Graph nodes  : {len(full_nodes)} (expect 3: Wait, Memcpy, EventRecord)")
        full_exec = cuda.graph_instantiate(full_graph)
        cuda.graph_launch(full_exec, stream)
        cuda.stream_synchronize(stream)
        full_event_ready = cuda.query_event(ipc_event)
        print(f"       query_event  : {full_event_ready} (expect True)")
        full_graph_ok = full_event_ready and len(full_nodes) == 3
        cuda.graph_exec_destroy(full_exec)
        cuda.graph_destroy(full_graph)
    except (RuntimeError, OSError) as exc:
        if capture_started:
            with contextlib.suppress(RuntimeError, OSError):
                cuda.graph_destroy(cuda.stream_end_capture(stream))
        print(f"       Full graph   : FAIL - {exc}")
        full_graph_ok = False
    finally:
        cuda.destroy_event(source_sync)
        if producer_stream is not None:
            with contextlib.suppress(RuntimeError, OSError):
                cuda.destroy_stream(producer_stream)
        for _buf in (src_buf, dst_buf):
            if _buf is not None:
                with contextlib.suppress(RuntimeError, OSError):
                    cuda.free(_buf)

    # --- Cleanup --------------------------------------------------------------
    cuda.destroy_event(ipc_event)
    cuda.destroy_stream(stream)

    passed = event_ready and len(nodes) == 1 and set_event_ok and full_graph_ok
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0, metavar="N", help="CUDA device index (default: 0)")
    args = ap.parse_args()

    ok = run_probe(device=args.device)

    print()
    if ok:
        print("[PASS] IPC-event-in-graph is supported on this hardware.")
        print("       Safe to proceed with P3 graph-event consolidation in exporter.py.")
    else:
        print("[FAIL] IPC-event-in-graph is NOT reliably supported on this hardware.")
        print("       Keep record_event outside the graph (existing exporter.py behaviour).")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
