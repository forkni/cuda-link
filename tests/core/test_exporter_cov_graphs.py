"""
Coverage tests for CUDA-Graphs export capture/build/launch branches:
_build_export_graphs(), _destroy_export_graphs(), and export()'s
graph-launch dispatch — none of which are exercised by the existing
use_graphs=False-only test suite.

Key lever: FakeCUDAAdapter.graph_get_nodes() always returns a single-element
list, so with the default (truthy) ipc_event the 3-node expectation is never
met — this naturally drives the node-count-mismatch fallback path without any
patching. Patching create_ipc_event() to return None makes ipc_event falsy,
so the 1-node expectation matches and the build succeeds instead.

All tests use FakeCUDAAdapter — no GPU required.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

from cuda_link import ExportPolicy, FrameOutcome, FrameSpec, GpuFrame
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.exporter import Exporter

_H = _W = _C = 4
_DATA_SIZE = _H * _W * _C


def _spec(*, num_slots: int = 1) -> FrameSpec:
    return FrameSpec(
        shm_name=f"test_cov_graphs_{uuid.uuid4().hex[:8]}",
        height=_H,
        width=_W,
        channels=_C,
        dtype="uint8",
        num_slots=num_slots,
        device=0,
    )


def _graphs_policy(*, export_profile: bool = False, flush_probe: bool = False) -> ExportPolicy:
    return ExportPolicy(
        export_sync=False,
        use_graphs=True,
        flush_probe=flush_probe,
        strict_device=False,
        require_source_sync=False,
        barrier_enabled=False,
        high_priority_stream=False,
        export_profile=export_profile,
    )


# ---------------------------------------------------------------------------
# _initialize() — runtime-version-check exception disables graphs (338-353)
# ---------------------------------------------------------------------------


def test_runtime_version_check_exception_disables_graphs() -> None:
    fake = FakeCUDAAdapter(device=0)
    with (
        patch.object(fake, "get_runtime_version", side_effect=RuntimeError("simulated version query failure")),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp = Exporter.open(_spec(), policy=_graphs_policy(), cuda=fake)
    try:
        assert exp._graphs_disabled is True
        assert exp._graph_execs == [None]  # _build_export_graphs() never called
        messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
        assert any("cudaRuntimeGetVersion failed" in m for m in messages)
        assert any("ignored: cudart" in m for m in messages)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# _build_export_graphs() — default node-count mismatch fallback (399-437, 448, 453-462)
# ---------------------------------------------------------------------------


def test_build_export_graphs_default_node_count_mismatch_disables_graphs() -> None:
    fake = FakeCUDAAdapter(device=0)
    with patch("cuda_link.exporter.logger") as mock_logger:
        exp = Exporter.open(_spec(num_slots=2), policy=_graphs_policy(), cuda=fake)
    try:
        assert exp._graphs_disabled is True
        assert exp._graph_execs == [None, None]
        messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
        assert any("CUDA Graph build failed" in m for m in messages)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# _build_export_graphs() — success path (438-447, 463) + _destroy_export_graphs() (466-477)
# ---------------------------------------------------------------------------


def test_build_export_graphs_success_path_builds_and_destroys_cleanly() -> None:
    """A falsy ipc_event makes the fake's 1-node graph match the expected count."""
    fake = FakeCUDAAdapter(device=0)
    with (
        patch.object(fake, "create_ipc_event", return_value=None),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp = Exporter.open(_spec(num_slots=2), policy=_graphs_policy(), cuda=fake)

    assert exp._graphs_disabled is False
    assert all(e is not None for e in exp._graph_execs)
    assert all(t is not None for t in exp._graph_templates)
    messages = [str(c.args[0]) for c in mock_logger.info.call_args_list if c.args]
    assert any("CUDA export graphs built" in m for m in messages)

    exp.close()  # exercises _destroy_export_graphs() real-destroy branches + _do_cleanup STEP1c

    assert exp._graph_execs == [None, None]
    assert exp._graph_templates == [None, None]


# ---------------------------------------------------------------------------
# _build_export_graphs() — 3-node success path with a truthy ipc_event (445)
# ---------------------------------------------------------------------------


def test_build_export_graphs_three_node_success_path_with_truthy_ipc_event() -> None:
    """A truthy ipc_event (default) needs a 3-node graph; patch graph_get_nodes()
    to return 3 nodes so the build succeeds down the Wait+Memcpy+EventRecord branch."""
    fake = FakeCUDAAdapter(device=0)
    three_nodes = [fake.graph_get_nodes(None)[0] for _ in range(3)]
    with (
        patch.object(fake, "graph_get_nodes", return_value=three_nodes),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp = Exporter.open(_spec(), policy=_graphs_policy(), cuda=fake)
    try:
        assert exp._graphs_disabled is False
        assert exp._graph_execs[0] is not None
        messages = [str(c.args[0]) for c in mock_logger.debug.call_args_list if c.args]
        assert any("3-node: Wait+Memcpy+EventRecord" in m for m in messages)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# _build_export_graphs() — abandons a mid-capture on exception (449-452)
# ---------------------------------------------------------------------------


def test_build_export_graphs_abandons_capture_on_mid_capture_exception() -> None:
    """An exception raised while capture_started=True must abandon+destroy the partial capture."""
    fake = FakeCUDAAdapter(device=0)
    with (
        patch.object(fake, "memcpy_async", side_effect=RuntimeError("simulated capture failure")),
        patch("cuda_link.exporter.logger") as mock_logger,
    ):
        exp = Exporter.open(_spec(), policy=_graphs_policy(), cuda=fake)
    try:
        assert exp._graphs_disabled is True
        messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
        assert any("CUDA Graph build failed" in m for m in messages)
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# export() — graph-launch success dispatch (630-657, 661->686 False direction)
# ---------------------------------------------------------------------------


def test_export_graph_launch_success_profiled() -> None:
    """Successful per-frame graph_launch (goto_legacy=False) with profiling on —
    also exercises the falsy-ipc_events export_sync/flush_probe timing branches
    since ipc_events[slot] is None (required for the graph build to succeed)."""
    fake = FakeCUDAAdapter(device=0)
    policy = _graphs_policy(export_profile=True, flush_probe=True)
    with patch.object(fake, "create_ipc_event", return_value=None):
        exp = Exporter.open(_spec(), policy=policy, cuda=fake)
    try:
        assert exp._graphs_disabled is False
        outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert outcome == FrameOutcome.PUBLISHED
        assert exp._graph_last_src[0] == 0xDEAD0000
        # Same pointer again -> skips the node-params update (bonus branch coverage).
        outcome2 = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert outcome2 == FrameOutcome.PUBLISHED
    finally:
        exp.close()


# ---------------------------------------------------------------------------
# export() — graph-launch failure falls back to legacy path (650-653)
# ---------------------------------------------------------------------------


def test_export_graph_launch_failure_falls_back_to_legacy() -> None:
    fake = FakeCUDAAdapter(device=0)
    with patch.object(fake, "create_ipc_event", return_value=None):
        exp = Exporter.open(_spec(), policy=_graphs_policy(), cuda=fake)
    try:
        assert exp._graphs_disabled is False
        with (
            patch.object(fake, "graph_launch", side_effect=RuntimeError("simulated launch failure")),
            patch("cuda_link.exporter.logger") as mock_logger,
        ):
            outcome = exp.export(GpuFrame(ptr=0xDEAD0000, size=_DATA_SIZE))
        assert outcome == FrameOutcome.PUBLISHED  # falls back to legacy within the same call
        assert exp._graphs_disabled is True
        messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list if c.args]
        assert any("Graph launch failed" in m for m in messages)
    finally:
        exp.close()
