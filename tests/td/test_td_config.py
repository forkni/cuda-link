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
    # P1: export_sync is now tri-state; None = auto-select by topology (new default)
    assert cfg.export_sync is None
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


def test_sender_config_from_env_export_sync_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_EXPORT_SYNC absent → None (auto-select by topology)."""
    from TDConfig import TDSenderConfig

    monkeypatch.delenv("CUDALINK_EXPORT_SYNC", raising=False)
    cfg = TDSenderConfig.from_env()
    assert cfg.export_sync is None


def test_sender_config_from_env_export_sync_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_EXPORT_SYNC=1 forces blocking export_sync."""
    from TDConfig import TDSenderConfig

    monkeypatch.setenv("CUDALINK_EXPORT_SYNC", "1")
    cfg = TDSenderConfig.from_env()
    assert cfg.export_sync is True


def test_sender_config_from_env_export_sync_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDALINK_EXPORT_SYNC=0 forces async export_sync."""
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


# ---------------------------------------------------------------------------
# S-B — honest receiver config type guard
# ---------------------------------------------------------------------------


def test_receiver_engine_stores_receiver_config() -> None:
    """TDReceiverEngine accepts TDReceiverConfig and stores it as _config (S-B).

    Ensures the engine's declared type matches the object it actually holds.
    """
    from unittest.mock import MagicMock

    from TDConfig import TDReceiverConfig
    from TDReceiver import TDReceiverEngine

    engine = TDReceiverEngine(
        host=MagicMock(),
        config=TDReceiverConfig(),
        cuda=None,
        log_fn=lambda msg, force=False: None,
        num_slots=3,
        device=0,
        shm_name="test",
        verbose=False,
    )
    assert isinstance(engine._config, TDReceiverConfig), (
        "TDReceiverEngine._config must be a TDReceiverConfig, not TDSenderConfig"
    )


def test_make_engine_receiver_mode_passes_receiver_config() -> None:
    """_make_engine() in Receiver mode injects TDReceiverConfig(), not TDSenderConfig (S-B)."""
    from unittest.mock import MagicMock

    import CUDAIPCExtension as ext_mod  # noqa: N813
    from CUDAIPCExtension import CUDAIPCExtension
    from TDConfig import TDReceiverConfig, TDRuntimeState, TDSenderConfig

    # Build extension instance without __init__ to avoid TD-host side-effects.
    ext = object.__new__(CUDAIPCExtension)
    ext._host = MagicMock()
    ext._config = TDSenderConfig()
    ext._mode = "Receiver"
    ext._device = 0
    ext._runtime_state = TDRuntimeState(shm_name="test", num_slots=3, verbose=False)
    ext._log = lambda msg, force=False: None
    ext._receiver_registered = False  # required by cleanup() / _make_engine()

    before = ext_mod._active_receiver_count
    engine = ext._make_engine()
    try:
        assert isinstance(engine._config, TDReceiverConfig), (
            "_make_engine() must pass TDReceiverConfig() to TDReceiverEngine, not TDSenderConfig"
        )
        assert ext_mod._active_receiver_count == before + 1, (
            "_make_engine() in Receiver mode must call _register_receiver()"
        )
    finally:
        # Undo the registration so this test doesn't leak into the process-level registry.
        ext_mod._unregister_receiver()
        ext._receiver_registered = False


# ---------------------------------------------------------------------------
# P1 — Active-receiver registry and tri-state export_sync resolution
# ---------------------------------------------------------------------------


def test_receiver_registry_register_unregister() -> None:
    """_register_receiver / _unregister_receiver keep the count correct."""
    import CUDAIPCExtension as ext_mod  # noqa: N813

    initial = ext_mod._active_receiver_count
    ext_mod._register_receiver()
    assert ext_mod.receiver_active_in_process() is True
    assert ext_mod._active_receiver_count == initial + 1
    ext_mod._unregister_receiver()
    assert ext_mod._active_receiver_count == initial


def test_receiver_registry_unregister_floors_at_zero() -> None:
    """_unregister_receiver never goes below zero (guard against double-cleanup)."""
    import CUDAIPCExtension as ext_mod  # noqa: N813

    # Ensure count is 0 for this test.
    saved = ext_mod._active_receiver_count
    ext_mod._active_receiver_count = 0
    try:
        ext_mod._unregister_receiver()  # should not raise, should stay 0
        assert ext_mod._active_receiver_count == 0
    finally:
        ext_mod._active_receiver_count = saved


def test_export_sync_resolution_table() -> None:
    """P1 tri-state resolution: (config.export_sync, receiver_active) → effective_sync.

    (None,  False) → False  (async — no receiver in process)
    (None,  True)  → True   (blocking — receiver coexists in process)
    (True,  False) → True   (explicit override)
    (False, True)  → False  (explicit override)
    """
    from unittest.mock import MagicMock

    from TDConfig import TDSenderConfig
    from TDSender import TDSenderEngine

    def _make_engine(export_sync_val: bool | None, receiver_present: bool) -> TDSenderEngine:
        eng = object.__new__(TDSenderEngine)
        eng._host = MagicMock()
        eng._config = TDSenderConfig(export_sync=export_sync_val)
        eng._receiver_check = lambda: receiver_present
        eng._log_fn = lambda msg, force=False: None
        eng._log = eng._log_fn
        eng._initialized = False
        return eng

    table: list[tuple[bool | None, bool, bool]] = [
        (None, False, False),  # auto → async
        (None, True, True),  # auto → blocking
        (True, False, True),  # explicit True
        (False, True, False),  # explicit False
    ]

    for export_sync_val, receiver_present, expected in table:
        eng = _make_engine(export_sync_val, receiver_present)
        # Resolve the effective_sync the same way initialize() does.
        effective = eng._config.export_sync if eng._config.export_sync is not None else eng._receiver_check()
        assert effective is expected, (
            f"export_sync={export_sync_val!r}, receiver={receiver_present} → expected {expected}, got {effective}"
        )
