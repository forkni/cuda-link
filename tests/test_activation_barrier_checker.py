"""Tests for CheckerBarrier + CheckerOutcome via FakeShmAdapter (no real SHM)."""

from __future__ import annotations

import time

import pytest

from cuda_link.activation_barrier import BarrierShmPort, CheckerBarrier, CheckerOutcome
from conftest import FakeShmAdapter

# ---------------------------------------------------------------------------
# Structural check: FakeShmAdapter satisfies BarrierShmPort
# ---------------------------------------------------------------------------


def test_fake_shm_adapter_satisfies_port():
    assert isinstance(FakeShmAdapter(), BarrierShmPort)


# ---------------------------------------------------------------------------
# Every CheckerOutcome value
# ---------------------------------------------------------------------------


class TestCheckerOutcomes:
    def test_disabled_returns_disabled(self):
        cb = CheckerBarrier(enabled=False, stale_ns=5_000_000_000, shm=FakeShmAdapter())
        assert cb.evaluate() is CheckerOutcome.DISABLED
        assert not cb.evaluate().should_skip

    def test_shm_absent_when_attach_raises_filenotfound(self):
        fake = FakeShmAdapter(raise_on_attach=FileNotFoundError)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert cb.evaluate() is CheckerOutcome.SHM_ABSENT
        assert not cb.evaluate().should_skip

    def test_shm_absent_when_attach_raises_oserror(self):
        fake = FakeShmAdapter(raise_on_attach=OSError)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert cb.evaluate() is CheckerOutcome.SHM_ABSENT

    def test_shm_absent_when_read_state_raises(self):
        fake = FakeShmAdapter(active_count=1, raise_on_read=OSError)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert cb.evaluate() is CheckerOutcome.SHM_ABSENT

    def test_no_skip_when_active_count_zero(self):
        fake = FakeShmAdapter(active_count=0)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert cb.evaluate() is CheckerOutcome.NO_SKIP
        assert not cb.evaluate().should_skip

    def test_skip_active_when_count_positive_and_fresh(self):
        fake = FakeShmAdapter(active_count=1)  # last_change_ns set to now on attach
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert cb.evaluate() is CheckerOutcome.SKIP_ACTIVE
        assert cb.evaluate().should_skip

    def test_skip_stale_when_count_positive_but_past_stale_ns(self):
        # last_change_ns set to 0 (far past stale_ns of 1 second)
        fake = FakeShmAdapter(active_count=2, last_change_ns=0)
        cb = CheckerBarrier(enabled=True, stale_ns=1_000_000_000, shm=fake)
        assert cb.evaluate() is CheckerOutcome.SKIP_STALE
        assert not cb.evaluate().should_skip

    def test_skip_active_increments_barrier_skips(self):
        fake = FakeShmAdapter(active_count=1)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        cb.evaluate()
        assert fake.barrier_skips == 1
        cb.evaluate()
        assert fake.barrier_skips == 2

    def test_bump_skip_error_does_not_change_outcome(self):
        fake = FakeShmAdapter(active_count=1, raise_on_bump=OSError)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        # OSError from bump_skip is swallowed; outcome is still SKIP_ACTIVE
        assert cb.evaluate() is CheckerOutcome.SKIP_ACTIVE

    def test_stale_does_not_increment_barrier_skips(self):
        fake = FakeShmAdapter(active_count=1, last_change_ns=0)
        cb = CheckerBarrier(enabled=True, stale_ns=1_000_000_000, shm=fake)
        cb.evaluate()
        assert fake.barrier_skips == 0


# ---------------------------------------------------------------------------
# Lazy attach: first call attaches; second call does not re-attach
# ---------------------------------------------------------------------------


class TestLazyAttach:
    def test_attach_happens_on_first_evaluate(self):
        fake = FakeShmAdapter(active_count=0)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert not fake.is_attached
        cb.evaluate()
        assert fake.is_attached

    def test_second_evaluate_does_not_re_attach(self):
        attach_calls = []
        fake = FakeShmAdapter(active_count=0)
        original_attach = fake.attach

        def counting_attach(*, create: bool) -> None:
            attach_calls.append(create)
            original_attach(create=create)

        fake.attach = counting_attach  # type: ignore[method-assign]
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        cb.evaluate()
        cb.evaluate()
        assert len(attach_calls) == 1

    def test_disabled_never_attaches(self):
        fake = FakeShmAdapter(active_count=1)
        cb = CheckerBarrier(enabled=False, stale_ns=5_000_000_000, shm=fake)
        cb.evaluate()
        assert not fake.is_attached


# ---------------------------------------------------------------------------
# Bool wrapper
# ---------------------------------------------------------------------------


class TestBoolWrapper:
    def test_should_skip_publish_matches_should_skip_property_when_active(self):
        fake = FakeShmAdapter(active_count=1)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert cb.should_skip_publish() is True

    def test_should_skip_publish_false_when_no_skip(self):
        fake = FakeShmAdapter(active_count=0)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        assert cb.should_skip_publish() is False

    def test_should_skip_publish_false_when_disabled(self):
        cb = CheckerBarrier(enabled=False, stale_ns=5_000_000_000, shm=FakeShmAdapter())
        assert cb.should_skip_publish() is False


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_detaches_fake(self):
        fake = FakeShmAdapter(active_count=0)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        cb.evaluate()
        assert fake.is_attached
        cb.close()
        assert not fake.is_attached

    def test_close_before_attach_is_idempotent(self):
        fake = FakeShmAdapter()
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        cb.close()  # should not raise


# ---------------------------------------------------------------------------
# Log throttle: warnings fire at most once per second
# ---------------------------------------------------------------------------


class TestLogThrottling:
    def test_stale_log_throttled(self, caplog: pytest.LogCaptureFixture):
        fake = FakeShmAdapter(active_count=1, last_change_ns=0)
        cb = CheckerBarrier(enabled=True, stale_ns=1_000_000_000, shm=fake)
        with caplog.at_level("WARNING", logger="cuda_link.activation_barrier"):
            cb.evaluate()
            cb.evaluate()
            cb.evaluate()
        # Only one warning despite three calls (log throttle = 1 second)
        stale_warnings = [r for r in caplog.records if "stale barrier" in r.message]
        assert len(stale_warnings) == 1

    def test_skip_log_throttled(self, caplog: pytest.LogCaptureFixture):
        fake = FakeShmAdapter(active_count=1)
        cb = CheckerBarrier(enabled=True, stale_ns=5_000_000_000, shm=fake)
        with caplog.at_level("INFO", logger="cuda_link.activation_barrier"):
            cb.evaluate()
            cb.evaluate()
            cb.evaluate()
        skip_infos = [r for r in caplog.records if "skipping publish" in r.message]
        assert len(skip_infos) == 1
