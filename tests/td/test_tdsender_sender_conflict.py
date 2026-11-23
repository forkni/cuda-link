# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""A Sender COMP whose name another sender holds: one log line, no traceback, and a retry.

export_frame() calls initialize() again on each cook while the engine is not open. A
SenderConflictError on that path is a setup mistake that can last for minutes, so the
log gets one line for each different message. Other failures keep their traceback.
"""

from __future__ import annotations

import types
import uuid

import pytest
from Exporter import Exporter, FrameOutcome, SenderConflictError
from fakes import FakeTDHost, FakeTOPHandle
from SHMProtocol import DtypeCodec
from TDConfig import TDSenderConfig
from TDSender import TDSenderEngine

_IN_PROCESS = "SharedMemory 'x' already has a sender in this process"
_OTHER_PROCESS = "SharedMemory 'x' already has a running sender (process 4242)"


class _FakeExporter:
    """Minimal Exporter double -- see test_tdsender_grow_safety.py for rationale."""

    def __init__(self, spec) -> None:  # noqa: ANN001
        self.data_size = spec.width * spec.height * spec.channels * DtypeCodec.itemsize(spec.dtype)
        self.write_idx = 0
        self.frame_count = 0
        self.ipc_stream = types.SimpleNamespace(value=0)
        self.shm_handle = None

    def record_source_sync(self, stream_id: int) -> None:
        pass

    def is_ready(self) -> bool:
        return True

    def export(self, frame) -> FrameOutcome:  # noqa: ANN001
        self.frame_count += 1
        self.write_idx += 1
        return FrameOutcome.PUBLISHED

    def close(self) -> None:
        pass


class _Name:
    """What Exporter.open does for the name: raise *error*, or open when it is None."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, error: Exception | None) -> None:
        self.error = error
        self.opens = 0

        def _open(spec, *, policy=None, cuda=None, barrier=None):  # noqa: ANN001, ARG001
            self.opens += 1
            if self.error is not None:
                raise self.error
            return _FakeExporter(spec)

        monkeypatch.setattr(Exporter, "open", _open)


def _engine(log: list[str]) -> tuple[TDSenderEngine, FakeTDHost]:
    top = FakeTOPHandle(
        pixel_format="32-bit float (RGBA)",
        pixel_format_name="rgba32float",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x9000,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = TDSenderEngine(
        host=host,
        config=TDSenderConfig(activation_barrier=False),
        cuda=None,
        log_fn=log.append,
        num_slots=1,
        device=0,
        shm_name=f"test_conflict_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )
    return engine, host


def _failure_lines(log: list[str]) -> list[str]:
    return [line for line in log if "Initialization failed" in line]


def test_a_held_name_gives_one_log_line_for_many_cooks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    name = _Name(monkeypatch, SenderConflictError(_IN_PROCESS))
    log: list[str] = []
    engine, host = _engine(log)

    for _ in range(5):
        assert engine.export_frame() is False

    assert name.opens == 5, "The engine must try again on each cook."
    assert _failure_lines(log) == [f"Initialization failed: {_IN_PROCESS}"]
    assert len([line for line in log if "Auto-init" in line]) == 1
    assert "Traceback" not in capsys.readouterr().err
    assert host.status_calls[-1] == ("error", f"Initialization failed: {_IN_PROCESS}")


def test_a_different_conflict_message_gives_a_new_log_line(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _Name(monkeypatch, SenderConflictError(_IN_PROCESS))
    log: list[str] = []
    engine, host = _engine(log)

    engine.export_frame()
    engine.export_frame()
    name.error = SenderConflictError(_OTHER_PROCESS)
    engine.export_frame()
    engine.export_frame()

    assert _failure_lines(log) == [
        f"Initialization failed: {_IN_PROCESS}",
        f"Initialization failed: {_OTHER_PROCESS}",
    ]
    assert host.status_calls[-1] == ("error", f"Initialization failed: {_OTHER_PROCESS}")


def test_the_engine_opens_when_the_name_becomes_free(monkeypatch: pytest.MonkeyPatch) -> None:
    name = _Name(monkeypatch, SenderConflictError(_IN_PROCESS))
    log: list[str] = []
    engine, host = _engine(log)
    engine.export_frame()
    engine.export_frame()

    name.error = None
    assert engine.export_frame() is False, "The cook that opens the Exporter publishes nothing."
    assert engine.export_frame() is True

    kinds = [kind for kind, _ in host.status_calls]
    assert "clear" in kinds[len(kinds) - kinds[::-1].index("error") :], "The error status must be cleared."
    assert any("Initialization complete" in line for line in log)


def test_the_same_conflict_after_cleanup_is_logged_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _Name(monkeypatch, SenderConflictError(_IN_PROCESS))
    log: list[str] = []
    engine, _host = _engine(log)
    engine.initialize(64, 64, 4, 64 * 64 * 16)

    engine.cleanup()
    assert engine.initialize(64, 64, 4, 64 * 64 * 16) is False

    assert len(_failure_lines(log)) == 2


def test_another_failure_keeps_its_traceback_on_each_cook(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _Name(monkeypatch, OSError("simulated CUDA IPC failure"))
    log: list[str] = []
    engine, _host = _engine(log)

    engine.export_frame()
    engine.export_frame()

    assert len(_failure_lines(log)) == 2
    assert capsys.readouterr().err.count("Traceback") == 2
