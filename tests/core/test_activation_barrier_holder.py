"""Tests for HolderBarrier + HolderShmPort via FakeShmAdapter (no real SHM)."""

from __future__ import annotations

from fakes import FakeShmAdapter

from cuda_link.activation_barrier import HolderBarrier, HolderShmPort, RealShmAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PID = 12345


def _noop_log(msg: str, *, force: bool = False) -> None:
    pass


def _collecting_log(log_list: list[str]):
    def _fn(msg: str, *, force: bool = False) -> None:
        log_list.append(msg)

    return _fn


def _make_holder(*, enabled: bool = True, settle_frames: int = 3, port=None) -> HolderBarrier:
    port = port or FakeShmAdapter()
    return HolderBarrier(enabled=enabled, settle_frames=settle_frames, port=port)


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


class TestHolderShmPortStructural:
    def test_fake_shm_adapter_satisfies_holder_port(self):
        assert isinstance(FakeShmAdapter(), HolderShmPort)

    def test_real_shm_adapter_satisfies_holder_port(self):
        assert isinstance(RealShmAdapter(), HolderShmPort)


# ---------------------------------------------------------------------------
# acquire()
# ---------------------------------------------------------------------------


class TestAcquire:
    def test_disabled_acquire_is_noop(self):
        fake = FakeShmAdapter()
        hb = _make_holder(enabled=False, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        assert not hb.held
        assert fake.active_count == 0

    def test_enabled_acquire_increments_and_sets_held(self):
        fake = FakeShmAdapter()
        hb = _make_holder(enabled=True, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        assert hb.held
        assert fake.active_count == 1

    def test_acquire_logs_message_on_success(self):
        logs: list[str] = []
        fake = FakeShmAdapter()
        hb = _make_holder(enabled=True, port=fake)
        hb.acquire(_PID, log_fn=_collecting_log(logs))
        assert any("ACTIVATION_BARRIER" in m and "held +1" in m for m in logs)

    def test_acquire_failure_swallowed_held_stays_false(self):
        class _FailPort:
            def open_and_increment(self, pid: int) -> int:
                raise OSError("simulated")

            def decrement(self, pid: int) -> int:
                return 0

            def close(self) -> None:
                pass

        hb = _make_holder(enabled=True, port=_FailPort())
        hb.acquire(_PID, log_fn=_noop_log)
        assert not hb.held

    def test_acquire_failure_logs_error(self):
        class _FailPort:
            def open_and_increment(self, pid: int) -> int:
                raise OSError("boom")

            def decrement(self, pid: int) -> int:
                return 0

            def close(self) -> None:
                pass

        logs: list[str] = []
        hb = _make_holder(enabled=True, port=_FailPort())
        hb.acquire(_PID, log_fn=_collecting_log(logs))
        assert any("increment failed" in m for m in logs)


# ---------------------------------------------------------------------------
# arm_settle_countdown()
# ---------------------------------------------------------------------------


class TestArmSettle:
    def test_arm_settle_when_held_sets_remaining(self):
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=5, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        assert hb.settle_remaining == 5

    def test_arm_settle_when_not_held_is_noop(self):
        hb = _make_holder(settle_frames=5)
        hb.arm_settle_countdown()
        assert hb.settle_remaining == 0


# ---------------------------------------------------------------------------
# tick_and_maybe_release()
# ---------------------------------------------------------------------------


class TestTickAndMaybeRelease:
    def test_tick_before_arm_returns_false(self):
        fake = FakeShmAdapter()
        hb = _make_holder(port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        assert hb.tick_and_maybe_release(_PID, log_fn=_noop_log) is False

    def test_tick_decrements_remaining(self):
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=3, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        hb.tick_and_maybe_release(_PID, log_fn=_noop_log)
        assert hb.settle_remaining == 2

    def test_tick_does_not_release_before_zero(self):
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=3, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        # first two ticks should not release
        hb.tick_and_maybe_release(_PID, log_fn=_noop_log)
        hb.tick_and_maybe_release(_PID, log_fn=_noop_log)
        assert hb.held
        assert fake.active_count == 1

    def test_tick_releases_at_zero_and_returns_true(self):
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=1, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        result = hb.tick_and_maybe_release(_PID, log_fn=_noop_log)
        assert result is True
        assert fake.active_count == 0

    def test_tick_release_clears_held(self):
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=1, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        hb.tick_and_maybe_release(_PID, log_fn=_noop_log)
        assert not hb.held

    def test_tick_release_logs_success(self):
        logs: list[str] = []
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=1, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        hb.tick_and_maybe_release(_PID, log_fn=_collecting_log(logs))
        assert any("released after" in m for m in logs)

    def test_tick_decrement_failure_swallowed_and_held_cleared(self):
        class _PartialPort:
            def open_and_increment(self, pid: int) -> int:
                return 1

            def decrement(self, pid: int) -> int:
                raise RuntimeError("simulated")

            def close(self) -> None:
                pass

        hb = _make_holder(settle_frames=1, port=_PartialPort())
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        result = hb.tick_and_maybe_release(_PID, log_fn=_noop_log)
        assert result is False
        assert not hb.held


# ---------------------------------------------------------------------------
# force_release()
# ---------------------------------------------------------------------------


class TestForceRelease:
    def test_force_release_when_not_held_is_noop(self):
        fake = FakeShmAdapter()
        hb = _make_holder(port=fake)
        hb.force_release(_PID, log_fn=_noop_log)
        assert fake.active_count == 0

    def test_force_release_decrements_and_clears_held(self):
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=5, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.arm_settle_countdown()
        hb.force_release(_PID, log_fn=_noop_log)
        assert not hb.held
        assert fake.active_count == 0

    def test_force_release_logs_count(self):
        logs: list[str] = []
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=5, port=fake)
        hb.acquire(_PID, log_fn=_noop_log)
        hb.force_release(_PID, log_fn=_collecting_log(logs))
        assert any("released on cleanup" in m for m in logs)

    def test_force_release_decrement_failure_swallowed_held_cleared(self):
        class _FailDecPort:
            def open_and_increment(self, pid: int) -> int:
                return 1

            def decrement(self, pid: int) -> int:
                raise OSError("simulated")

            def close(self) -> None:
                pass

        hb = _make_holder(port=_FailDecPort())
        hb.acquire(_PID, log_fn=_noop_log)
        hb.force_release(_PID, log_fn=_noop_log)
        assert not hb.held


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_delegates_to_port(self):
        closed: list[bool] = []

        class _TrackingPort:
            def open_and_increment(self, pid: int) -> int:
                return 1

            def decrement(self, pid: int) -> int:
                return 0

            def close(self) -> None:
                closed.append(True)

        hb = _make_holder(port=_TrackingPort())
        hb.acquire(_PID, log_fn=_noop_log)
        hb.close()
        assert closed == [True]

    def test_close_before_acquire_is_idempotent(self):
        fake = FakeShmAdapter()
        hb = _make_holder(port=fake)
        hb.close()  # must not raise


# ---------------------------------------------------------------------------
# End-to-end lifecycle
# ---------------------------------------------------------------------------


class TestEndToEndLifecycle:
    def test_full_lifecycle_acquire_arm_tick_release(self):
        """acquire → arm → tick × N-1 → tick (releases) → force_release (noop)."""
        logs: list[str] = []
        log_fn = _collecting_log(logs)
        fake = FakeShmAdapter()
        hb = _make_holder(settle_frames=3, port=fake)

        hb.acquire(_PID, log_fn=log_fn)
        assert hb.held and fake.active_count == 1

        hb.arm_settle_countdown()
        assert hb.settle_remaining == 3

        # ticks 1 and 2 — no release
        assert hb.tick_and_maybe_release(_PID, log_fn=log_fn) is False
        assert hb.tick_and_maybe_release(_PID, log_fn=log_fn) is False
        assert hb.held

        # tick 3 — release fires
        assert hb.tick_and_maybe_release(_PID, log_fn=log_fn) is True
        assert not hb.held
        assert fake.active_count == 0

        # force_release is now noop
        hb.force_release(_PID, log_fn=log_fn)
        assert fake.active_count == 0
