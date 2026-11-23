# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""Exporter.slot_copy_done() and Exporter.drain(): the two calls an async producer uses to keep
its source buffer alive until the queued copy has read it. GPU-free: the methods are called on
a stand-in that has only the attributes they read."""

from __future__ import annotations

import types

from cuda_link.exporter import Exporter


class _Cuda:
    def __init__(self, done: dict[object, bool]) -> None:
        self.done = done
        self.queried: list[object] = []
        self.synced: list[object] = []

    def query_event(self, event: object) -> bool:
        self.queried.append(event)
        return self.done[event]

    def stream_synchronize(self, stream: object) -> None:
        self.synced.append(stream)


def _exporter(events: list, cuda: _Cuda, stream: object = "stream") -> types.SimpleNamespace:
    return types.SimpleNamespace(ipc_events=events, _cuda=cuda, ipc_stream=stream)


def test_slot_copy_done_reads_the_event_of_that_slot() -> None:
    cuda = _Cuda({"ev0": True, "ev1": False})
    exp = _exporter(["ev0", "ev1"], cuda)

    assert Exporter.slot_copy_done(exp, 0) is True
    assert Exporter.slot_copy_done(exp, 1) is False
    assert cuda.queried == ["ev0", "ev1"]


def test_slot_without_an_event_is_done() -> None:
    """export() blocks on the stream for a slot with no IPC event, so nothing is queued."""
    cuda = _Cuda({})
    exp = _exporter([None], cuda)

    assert Exporter.slot_copy_done(exp, 0) is True
    assert Exporter.slot_copy_done(exp, 5) is True  # out of range: closed or resized ring
    assert cuda.queried == []


def test_drain_waits_for_the_ipc_stream() -> None:
    cuda = _Cuda({})

    Exporter.drain(_exporter([], cuda))
    Exporter.drain(_exporter([], cuda, stream=None))  # closed: no stream, no call

    assert cuda.synced == ["stream"]
