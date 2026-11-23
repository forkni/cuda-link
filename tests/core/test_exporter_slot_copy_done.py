# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""Exporter.slot_copy_done() and Exporter.drain(): the two calls an async producer uses to keep
its source buffer alive until the queued copy has read it.

GPU-free: a real Exporter is opened on a FakeCUDAAdapter subclass that reports a settable set of
events as not done and records each stream wait.
"""

from __future__ import annotations

import uuid
from typing import Any

from cuda_link import ExportPolicy, FrameSpec
from cuda_link._cuda_adapters import FakeCUDAAdapter
from cuda_link.exporter import Exporter

_NUM_SLOTS = 3


class _Cuda(FakeCUDAAdapter):
    def __init__(self) -> None:
        super().__init__(device=0)
        self.pending: list[Any] = []  # events whose copy has not completed
        self.queried: list[Any] = []
        self.synced: list[Any] = []

    def query_event(self, event: Any) -> bool:
        self.queried.append(event)
        return not any(event is p for p in self.pending)

    def stream_synchronize(self, stream: Any) -> None:
        self.synced.append(stream)


def _open() -> tuple[Exporter, _Cuda]:
    cuda = _Cuda()
    spec = FrameSpec(
        shm_name=f"test_copy_done_{uuid.uuid4().hex[:8]}",
        height=4,
        width=4,
        channels=4,
        dtype="uint8",
        num_slots=_NUM_SLOTS,
        device=0,
    )
    return Exporter.open(spec, policy=ExportPolicy.for_testing(), cuda=cuda), cuda


def test_slot_copy_done_reads_the_event_of_that_slot() -> None:
    exp, cuda = _open()
    try:
        assert all(exp.ipc_events), "The fake adapter gives each slot an IPC event."
        cuda.pending = [exp.ipc_events[1]]
        cuda.queried.clear()

        assert exp.slot_copy_done(0) is True
        assert exp.slot_copy_done(1) is False
        assert exp.slot_copy_done(2) is True
        assert [any(q is e for e in exp.ipc_events) for q in cuda.queried] == [True, True, True]
        assert cuda.queried[1] is exp.ipc_events[1]
    finally:
        exp.close()


def test_slot_without_an_event_is_done() -> None:
    """export() blocks on the stream for a slot with no IPC event, so nothing is queued."""
    exp, cuda = _open()
    try:
        exp.ipc_events[0] = None
        cuda.queried.clear()

        assert exp.slot_copy_done(0) is True
        assert exp.slot_copy_done(_NUM_SLOTS + 2) is True  # not a slot of this ring
        assert cuda.queried == []
    finally:
        exp.close()


def test_drain_waits_for_the_ipc_stream() -> None:
    exp, cuda = _open()
    try:
        cuda.synced.clear()

        exp.drain()

        assert len(cuda.synced) == 1
        assert cuda.synced[0] is exp.ipc_stream
    finally:
        exp.close()


def test_closed_exporter_reports_done_and_does_not_touch_the_stream() -> None:
    exp, cuda = _open()
    exp.close()
    cuda.synced.clear()
    cuda.queried.clear()

    exp.drain()

    assert exp.slot_copy_done(0) is True
    assert (cuda.synced, cuda.queried) == ([], [])


def test_drain_after_a_close_with_a_sticky_error_does_not_touch_the_stream() -> None:
    """close() with a CUDA context that is not valid leaves ipc_stream set. drain() must not
    use that stream again."""
    exp, cuda = _open()
    cuda._sticky_error = 719
    exp.close()
    assert exp.ipc_stream, "Precondition: the faulted close() did not clear the stream."
    cuda.synced.clear()

    exp.drain()

    assert cuda.synced == []
