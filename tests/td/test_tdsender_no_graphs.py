"""The .tox Sender always opens its Exporter with the plain stream path (use_graphs=False).

CUDALINK_TD_USE_GRAPHS was removed in 2.0.0: graphs on vs off measured +0.1 % at 1920x1080
on TD 2025.32820 (docs/BENCHMARKS.md, 2026-11-21). The variable must have no effect.

GPU-free: Exporter.open is patched to record the policy and return a fake exporter.
"""

from __future__ import annotations

import dataclasses
import types
import uuid

import pytest
from Exporter import Exporter, FrameOutcome
from fakes import FakeTDHost, FakeTOPHandle
from TDConfig import TDSenderConfig
from TDSender import TDSenderEngine


class _FakeExporter:
    """Minimal Exporter double: only the surface TDSender.py touches."""

    def __init__(self, spec) -> None:  # noqa: ANN001
        self.spec = spec
        self.data_size = spec.width * spec.height * spec.channels
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


@pytest.mark.parametrize("env_value", [None, "0", "1"])
def test_sender_opens_exporter_without_graphs(monkeypatch: pytest.MonkeyPatch, env_value: str | None) -> None:
    if env_value is None:
        monkeypatch.delenv("CUDALINK_TD_USE_GRAPHS", raising=False)
    else:
        monkeypatch.setenv("CUDALINK_TD_USE_GRAPHS", env_value)
    policies: list = []

    def _recording_open(spec, *, policy=None, cuda=None, barrier=None):  # noqa: ANN001, ARG001
        policies.append(policy)
        return _FakeExporter(spec)

    monkeypatch.setattr(Exporter, "open", _recording_open)

    config = dataclasses.replace(TDSenderConfig.from_env(), activation_barrier=False)
    assert not hasattr(config, "use_graphs")
    top = FakeTOPHandle(
        pixel_format="8-bit fixed",
        pixel_format_name="rgba8fixed",
        width=64,
        height=64,
        channels=4,
        gpu_ptr=0x1000,
        cuda_size=64 * 64 * 4,
    )
    host = FakeTDHost(params={"Active": True, "Debug": False}, tops={"ExportBuffer": top})
    engine = TDSenderEngine(
        host=host,
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=f"test_no_graphs_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )

    engine.export_frame()  # lazy-init probe opens the Exporter
    engine.export_frame()

    assert policies, "the Sender never opened an Exporter"
    assert all(p is not None and p.use_graphs is False for p in policies)
