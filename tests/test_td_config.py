"""
Unit tests for TDConfig — frozen configuration dataclasses.
"""

from __future__ import annotations

import os

import pytest


def test_sender_config_defaults() -> None:
    """TDSenderConfig() has correct production defaults."""
    from TDConfig import TDSenderConfig

    cfg = TDSenderConfig()
    assert cfg.export_sync is True
    assert cfg.export_profile is False
    assert cfg.export_flush_probe is True
    assert cfg.use_graphs is False
    assert cfg.graphs_deferred is False
    assert cfg.stream_high_prio is False
    assert cfg.init_pace is False
    assert cfg.persist_stream is True
    assert cfg.activation_barrier is True
    assert cfg.barrier_settle_frames == 30
    assert cfg.nvml is False


def test_sender_config_is_frozen() -> None:
    """TDSenderConfig is immutable after construction."""
    from TDConfig import TDSenderConfig

    cfg = TDSenderConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.export_sync = False  # type: ignore[misc]


def test_sender_config_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_env() with no env vars produces defaults."""
    from TDConfig import TDSenderConfig

    # Remove all CUDALINK_* vars so we get pure defaults
    for key in list(os.environ):
        if key.startswith("CUDALINK_"):
            monkeypatch.delenv(key, raising=False)

    cfg = TDSenderConfig.from_env()
    assert cfg == TDSenderConfig()


def test_sender_config_from_env_export_sync_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_EXPORT_SYNC=0 disables export_sync."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_EXPORT_SYNC", "0")
    cfg = TDSenderConfig.from_env()
    assert cfg.export_sync is False


def test_sender_config_from_env_export_profile_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_EXPORT_PROFILE=1 enables export_profile."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_EXPORT_PROFILE", "1")
    cfg = TDSenderConfig.from_env()
    assert cfg.export_profile is True


def test_sender_config_from_env_export_flush_probe_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_EXPORT_FLUSH_PROBE=0 disables export_flush_probe."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_EXPORT_FLUSH_PROBE", "0")
    cfg = TDSenderConfig.from_env()
    assert cfg.export_flush_probe is False


def test_sender_config_from_env_use_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_TD_USE_GRAPHS=1 enables use_graphs."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_TD_USE_GRAPHS", "1")
    cfg = TDSenderConfig.from_env()
    assert cfg.use_graphs is True


def test_sender_config_from_env_stream_high_prio(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_TD_STREAM_PRIO=high enables stream_high_prio."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_TD_STREAM_PRIO", "high")
    cfg = TDSenderConfig.from_env()
    assert cfg.stream_high_prio is True


def test_sender_config_from_env_stream_prio_normal(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_TD_STREAM_PRIO=normal keeps stream_high_prio False."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_TD_STREAM_PRIO", "normal")
    cfg = TDSenderConfig.from_env()
    assert cfg.stream_high_prio is False


def test_sender_config_from_env_barrier_settle_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_TD_BARRIER_SETTLE_FRAMES overrides the integer default."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_TD_BARRIER_SETTLE_FRAMES", "60")
    cfg = TDSenderConfig.from_env()
    assert cfg.barrier_settle_frames == 60


def test_sender_config_from_env_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_NVML=1 enables nvml."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_NVML", "1")
    cfg = TDSenderConfig.from_env()
    assert cfg.nvml is True


def test_sender_config_from_env_persist_stream_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_TD_PERSIST_STREAM=0 disables persist_stream."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_TD_PERSIST_STREAM", "0")
    cfg = TDSenderConfig.from_env()
    assert cfg.persist_stream is False


def test_sender_config_from_env_activation_barrier_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_TD_ACTIVATION_BARRIER=0 disables activation_barrier."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_TD_ACTIVATION_BARRIER", "0")
    cfg = TDSenderConfig.from_env()
    assert cfg.activation_barrier is False


def test_sender_config_post_init_negative_settle_frames() -> None:
    """barrier_settle_frames < 0 raises ValueError."""
    from TDConfig import TDSenderConfig

    with pytest.raises(ValueError, match="barrier_settle_frames"):
        TDSenderConfig(barrier_settle_frames=-1)


def test_sender_config_equality() -> None:
    """Two TDSenderConfig instances with same values compare equal."""
    from TDConfig import TDSenderConfig

    a = TDSenderConfig(export_sync=False, use_graphs=True, barrier_settle_frames=10)
    b = TDSenderConfig(export_sync=False, use_graphs=True, barrier_settle_frames=10)
    assert a == b


def test_receiver_config_instantiates() -> None:
    """TDReceiverConfig() can be constructed (placeholder)."""
    from TDConfig import TDReceiverConfig

    cfg = TDReceiverConfig()
    assert cfg is not None
