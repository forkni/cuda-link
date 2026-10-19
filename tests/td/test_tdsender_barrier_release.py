"""
TDSender must not leave the activation barrier held after a failed initialize().

A failed init never reaches arm_settle_countdown(), so without an explicit release
the hold only cleared through the producer's 5 s stale timeout -- and each retried
cook used to stack another increment on top. GPU-free: Exporter.open is patched to
raise, and the HolderBarrier's SHM port is swapped for the in-process FakeShmAdapter.
"""

from __future__ import annotations

import uuid

import pytest
from Exporter import Exporter
from fakes import FakeShmAdapter, FakeTDHost
from TDConfig import TDSenderConfig
from TDSender import TDSenderEngine


def _make_engine(host: FakeTDHost) -> TDSenderEngine:
    config = TDSenderConfig(activation_barrier=True, use_graphs=False)
    engine = TDSenderEngine(
        host=host,
        config=config,
        cuda=None,
        log_fn=lambda *a, **k: None,
        num_slots=1,
        device=0,
        shm_name=f"test_barrier_{uuid.uuid4().hex[:8]}",
        verbose=False,
    )
    engine._barrier.port = FakeShmAdapter()
    return engine


def test_failed_initialize_releases_barrier_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_open(*args, **kwargs):
        raise OSError("simulated SHM create failure")

    monkeypatch.setattr(Exporter, "open", staticmethod(_failing_open))
    host = FakeTDHost(params={"Active": True, "Debug": False})
    engine = _make_engine(host)
    port = engine._barrier.port

    for _ in range(3):
        assert engine.initialize(64, 64, 4) is False
        assert port.active_count == 0
        assert engine._barrier.held is False
