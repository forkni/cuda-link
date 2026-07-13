"""Coverage-focused tests for RealShmAdapter in src/cuda_link/activation_barrier.py.

test_activation_barrier.py already covers the module-level SHM-IO functions
(open_or_create/read_state/increment/decrement/bump_skip) directly. The
Checker/Holder role-class test files exercise RealShmAdapter only via an
isinstance() structural check, never calling its methods. This file drives
RealShmAdapter's methods directly against a real (but test-local) named
SharedMemory segment — no GPU required.
"""

from __future__ import annotations

import contextlib
import logging
from multiprocessing.shared_memory import SharedMemory

import pytest
from fakes import FakeShmAdapter

from cuda_link.activation_barrier import (
    _STRUCT,
    MAGIC,
    SHM_NAME,
    SHM_SIZE,
    VERSION,
    CheckerBarrier,
    CheckerOutcome,
    HolderBarrier,
    RealShmAdapter,
    decrement,
    increment,
    open_or_create,
)

# ---------------------------------------------------------------------------
# Helpers (duplicated from test_activation_barrier.py — new test files must
# not edit existing ones)
# ---------------------------------------------------------------------------


def _cleanup(name: str) -> None:
    try:
        shm = SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True)
def cleanup_barrier():
    _cleanup(SHM_NAME)
    # On Windows a live TD session (or a not-yet-GC'd handle elsewhere in this
    # process) can keep the named SHM alive past unlink() because SHM lifetime
    # is handle-bound (not name-bound like POSIX). If the segment persists,
    # re-zero its contents so each test starts from a deterministic zero state.
    with contextlib.suppress(FileNotFoundError):
        shm = SharedMemory(name=SHM_NAME)
        try:
            _STRUCT.pack_into(shm.buf, 0, MAGIC, VERSION, 0, 0, 0, 0, 0, b"\x00" * 32)
        finally:
            shm.close()
    yield
    _cleanup(SHM_NAME)


# ---------------------------------------------------------------------------
# is_attached / attach
# ---------------------------------------------------------------------------


def test_is_attached_false_initially():
    adapter = RealShmAdapter()
    assert adapter.is_attached is False


def test_attach_sets_is_attached_true():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        assert adapter.is_attached is True
    finally:
        adapter.close()


def test_attach_is_idempotent_does_not_recreate():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        first_shm = adapter._shm
        adapter.attach(create=True)  # second call must be a no-op
        assert adapter._shm is first_shm
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# read_state
# ---------------------------------------------------------------------------


def test_read_state_before_attach_raises():
    adapter = RealShmAdapter()
    with pytest.raises(RuntimeError, match="attach\\(\\) not called"):
        adapter.read_state()


def test_read_state_after_attach_returns_tuple():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        state = adapter.read_state()
        assert state == (0, 0, 0)
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# bump_skip
# ---------------------------------------------------------------------------


def test_bump_skip_before_attach_is_noop():
    adapter = RealShmAdapter()
    adapter.bump_skip()  # must not raise even though never attached
    assert adapter._shm is None


def test_bump_skip_after_attach_increments():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        adapter.bump_skip()
        _, _, skips = adapter.read_state()
        assert skips == 1
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# open_and_increment
# ---------------------------------------------------------------------------


def test_open_and_increment_creates_when_absent():
    adapter = RealShmAdapter()
    try:
        count = adapter.open_and_increment(pid=1234)
        assert count == 1
        assert adapter.is_attached is True
    finally:
        adapter.close()


def test_open_and_increment_reuses_existing_shm():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    try:
        first_shm = adapter._shm
        adapter.open_and_increment(pid=1)
        assert adapter._shm is first_shm
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# decrement
# ---------------------------------------------------------------------------


def test_decrement_before_open_raises():
    adapter = RealShmAdapter()
    with pytest.raises(RuntimeError, match="open_and_increment\\(\\) not called"):
        adapter.decrement(pid=1234)


def test_decrement_after_open_and_increment_works():
    adapter = RealShmAdapter()
    try:
        adapter.open_and_increment(pid=1)
        count = adapter.decrement(pid=1)
        assert count == 0
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


def test_close_before_attach_is_noop():
    adapter = RealShmAdapter()
    adapter.close()  # must not raise
    assert adapter._shm is None


def test_close_is_idempotent():
    adapter = RealShmAdapter()
    adapter.attach(create=True)
    adapter.close()
    adapter.close()  # second call must be a no-op, not raise
    assert adapter._shm is None


def test_close_swallows_oserror_from_underlying_shm(monkeypatch):
    adapter = RealShmAdapter()
    adapter.attach(create=True)

    def _raise_oserror():
        raise OSError("simulated close failure")

    monkeypatch.setattr(adapter._shm, "close", _raise_oserror)
    adapter.close()  # must not raise — OSError is suppressed
    assert adapter._shm is None


# ===========================================================================
# Mutation kill-tests (cosmic-ray survivor triage; see
# survivors-activation_barrier.txt at repo root for the full mutant list).
# Each test below is named for the specific behavior it locks down and the
# mutant class it kills; see the ledger in the triage report for the mapping
# from survivor line/operator to test name.
# ===========================================================================


# ---------------------------------------------------------------------------
# Local test doubles (kept local to this file — tests/fakes/__init__.py is
# not in the exclusive-edit set for this triage pass).
# ---------------------------------------------------------------------------


class _FailingHolderPort:
    """HolderShmPort double whose I/O methods always raise, to exercise the
    except-branches of HolderBarrier.acquire/tick_and_maybe_release/force_release."""

    def open_and_increment(self, pid):
        raise RuntimeError("simulated open_and_increment failure")

    def decrement(self, pid):
        raise RuntimeError("simulated decrement failure")

    def close(self):
        pass


class _AttachSpyPort:
    """BarrierShmPort double that records the `create` kwarg passed to attach()."""

    def __init__(self):
        self.create_calls: list[bool] = []
        self.attached = False

    @property
    def is_attached(self) -> bool:
        return self.attached

    def attach(self, *, create: bool) -> None:
        self.create_calls.append(create)
        self.attached = True

    def read_state(self):
        return (0, 0, 0)

    def bump_skip(self) -> None:
        pass

    def close(self) -> None:
        self.attached = False


def _make_recording_log(calls: list) -> object:
    """Return a log_fn double that records (message, force) tuples."""

    def _log(msg, *, force=False):
        calls.append((msg, force))

    return _log


# ---------------------------------------------------------------------------
# Module-level constants (lines 36-37): MAGIC / VERSION literal values.
# Kills: NumberReplacer on MAGIC (line 36, occ2/occ3), NumberReplacer on
# VERSION (line 37, occ4/occ5). The existing test_layout_magic_version_
# roundtrip compares against the *imported* MAGIC/VERSION symbols, so it is
# self-referential and can never catch a mutation to their definitions —
# this test asserts the literal wire-contract values instead.
# ---------------------------------------------------------------------------


def test_magic_and_version_literal_values_kill_number_replacer_mutants():
    assert MAGIC == 0xCDA1BAAA
    assert VERSION == 1


# ---------------------------------------------------------------------------
# open_or_create init (line 67): pad / last_change_ns / last_writer_pid must
# start at 0. Kills: NumberReplacer occ10 (pad 0->1), occ12 (last_change_ns
# 0->1), occ16 (last_writer_pid 0->1). Uses monkeypatch to force the
# FileNotFoundError -> create-and-initialize branch deterministically —
# Windows SHM handles can otherwise outlive unlink() on this dev machine,
# making the create branch flaky to hit via a bare attach(create=True).
# ---------------------------------------------------------------------------


def test_open_or_create_initializes_full_header_to_zero(monkeypatch):
    import cuda_link.activation_barrier as ab

    real_shm_cls = ab.SharedMemory

    def _factory(*, name, create=False, size=None):
        if not create:
            raise FileNotFoundError(name)
        return real_shm_cls(name=name, create=True, size=size)

    monkeypatch.setattr(ab, "SharedMemory", _factory)
    shm = ab.open_or_create(create=True)
    try:
        fields = ab._STRUCT.unpack(bytes(shm.buf[:SHM_SIZE]))
        assert fields == (MAGIC, VERSION, 0, 0, 0, 0, 0, b"\x00" * 32)
    finally:
        shm.close()
        _cleanup(SHM_NAME)


# ---------------------------------------------------------------------------
# increment/decrement field-index integrity (lines 95, 107, 109, 111).
# ---------------------------------------------------------------------------


def test_increment_writes_pid_to_last_writer_pid_field_kills_index_mutant():
    shm = open_or_create(create=True)
    try:
        increment(shm, pid=4242)
        fields = _STRUCT.unpack(bytes(shm.buf[:SHM_SIZE]))
        assert fields[6] == 4242  # last_writer_pid
        assert fields[5] == 0  # barrier_skips must be untouched
    finally:
        shm.close()


def test_decrement_subtracts_one_not_half_kills_rshift_mutant():
    shm = open_or_create(create=True)
    try:
        increment(shm, pid=1)
        increment(shm, pid=1)
        increment(shm, pid=1)  # active_count = 3
        new_count = decrement(shm, pid=1)
        assert new_count == 2
    finally:
        shm.close()


def test_decrement_writes_pid_to_last_writer_pid_field_kills_index_mutant():
    shm = open_or_create(create=True)
    try:
        increment(shm, pid=1)
        decrement(shm, pid=9999)
        fields = _STRUCT.unpack(bytes(shm.buf[:SHM_SIZE]))
        assert fields[6] == 9999  # last_writer_pid
        assert fields[5] == 0  # barrier_skips must be untouched
    finally:
        shm.close()


def test_decrement_returns_active_count_not_pad_kills_index_mutant():
    shm = open_or_create(create=True)
    try:
        increment(shm, pid=1)
        increment(shm, pid=1)  # active_count = 2
        result = decrement(shm, pid=1)
        assert result == 1
    finally:
        shm.close()


# ---------------------------------------------------------------------------
# RealShmAdapter (lines 191, 197, 213).
# ---------------------------------------------------------------------------


def test_real_shm_adapter_repr_excludes_shm_field_kills_repr_mutant():
    adapter = RealShmAdapter()
    assert "_shm" not in repr(adapter)


def test_attach_create_is_keyword_only_kills_separator_mutant():
    adapter = RealShmAdapter()
    with pytest.raises(TypeError):
        adapter.attach(True)  # create must be keyword-only


def test_open_and_increment_calls_open_or_create_with_create_true(monkeypatch):
    import cuda_link.activation_barrier as ab

    calls: list[bool] = []
    real_open_or_create = ab.open_or_create

    def _spy(*, create):
        calls.append(create)
        return real_open_or_create(create=True)

    monkeypatch.setattr(ab, "open_or_create", _spy)
    adapter = RealShmAdapter()
    try:
        adapter.open_and_increment(pid=1)
        assert calls == [True]
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# CheckerBarrier dataclass fields (lines 246, 247).
# ---------------------------------------------------------------------------


def test_checker_barrier_repr_excludes_private_log_fields_kills_repr_mutants():
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    r = repr(checker)
    assert "_skip_log_last_ns" not in r
    assert "_stale_log_last_ns" not in r


def test_checker_barrier_private_fields_are_init_false_kills_init_mutants():
    with pytest.raises(TypeError):
        CheckerBarrier(enabled=True, stale_ns=1, _skip_log_last_ns=5)
    with pytest.raises(TypeError):
        CheckerBarrier(enabled=True, stale_ns=1, _stale_log_last_ns=5)


def test_checker_barrier_private_fields_default_to_zero_kills_default_mutants():
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    assert checker._skip_log_last_ns == 0
    assert checker._stale_log_last_ns == 0


# ---------------------------------------------------------------------------
# CheckerBarrier.evaluate() control flow (lines 255, 264).
# ---------------------------------------------------------------------------


def test_evaluate_lazy_attach_uses_create_false_kills_flag_mutant():
    port = _AttachSpyPort()
    checker = CheckerBarrier(enabled=True, stale_ns=1_000_000_000, shm=port)
    checker.evaluate()
    assert port.create_calls == [False]


def test_evaluate_negative_active_count_treated_as_no_skip_kills_boundary_mutant():
    port = FakeShmAdapter(active_count=-1, attached=True, last_change_ns=1)
    checker = CheckerBarrier(enabled=True, stale_ns=1_000_000_000, shm=port)
    assert checker.evaluate() == CheckerOutcome.NO_SKIP


# ---------------------------------------------------------------------------
# Staleness comparison inside evaluate() (line 267): now_ns - last_change_ns
# > self.stale_ns. Exercised via evaluate() itself (not a private method),
# using FakeShmAdapter to control active_count/last_change_ns, and real
# time.monotonic_ns() for now_ns — margins are chosen generously wide so
# real wall-clock jitter cannot flip the outcome.
# ---------------------------------------------------------------------------


def test_stale_check_kills_rshift_mutant():
    # last_change_ns=1: age = now_ns - 1, an enormous number on any running
    # process (monotonic_ns since boot is always >> stale_ns=1). Under the
    # RShift mutant, now_ns >> 1 collapses to roughly now_ns/2, which is
    # still compared against stale_ns=1 -- so instead we pin stale_ns high
    # enough that only the correct subtraction result exceeds it, while
    # `now_ns >> last_change_ns` (right-shift by a huge amount) collapses to 0.
    port = FakeShmAdapter(active_count=1, attached=True, last_change_ns=10_000_000_000)
    checker = CheckerBarrier(enabled=True, stale_ns=1, shm=port)
    # now_ns (real monotonic) is astronomically larger than last_change_ns's
    # role as a *shift amount* (10_000_000_000 bit positions) — real subtraction
    # is trivially > stale_ns=1, but now_ns >> 10_000_000_000 == 0, which is not.
    assert checker.evaluate() == CheckerOutcome.SKIP_STALE


def test_stale_check_kills_bitxor_mutant(monkeypatch):
    # now_ns=2**32, last_change_ns=2**32-1: real age = 1 (well under
    # stale_ns), so the correct subtraction must NOT report stale. But
    # now_ns ^ last_change_ns = 0x1FFFFFFFF = 8589934591, far past
    # stale_ns -- a XOR-corrupted comparison flips the outcome to stale.
    # (A brute-force search proved a^b >= a-b for all a>=b>=0, so no
    # "clearly stale" scenario can ever distinguish these two operators;
    # this "not-stale-but-XOR-huge" scenario is the only kind that can.)
    import cuda_link.activation_barrier as ab

    now = 2**32
    last_change = 2**32 - 1
    monkeypatch.setattr(ab.time, "monotonic_ns", lambda: now)
    port = FakeShmAdapter(active_count=1, attached=True, last_change_ns=last_change)
    checker = CheckerBarrier(enabled=True, stale_ns=1_000_000_000, shm=port)
    assert checker.evaluate() != CheckerOutcome.SKIP_STALE


def test_stale_check_boundary_kills_gtge_mutant(monkeypatch):
    # now_ns - last_change_ns == stale_ns exactly: the strict '>' must be
    # False (not stale); a '>=' mutant flips this boundary case to stale.
    import cuda_link.activation_barrier as ab

    now = 2_000_000_000
    last_change = 1_000_000_000
    monkeypatch.setattr(ab.time, "monotonic_ns", lambda: now)
    port = FakeShmAdapter(active_count=1, attached=True, last_change_ns=last_change)
    checker = CheckerBarrier(enabled=True, stale_ns=1_000_000_000, shm=port)
    assert checker.evaluate() != CheckerOutcome.SKIP_STALE


# ---------------------------------------------------------------------------
# _log_stale / _log_skip throttle gates (lines 280, 289): both compare
# (now_ns - last_ns) against the literal 1_000_000_000 (1s). Called directly
# (private methods) with fully deterministic now_ns/last_ns operands so the
# gate arithmetic is pinned exactly -- no real sleeps, no clock dependence.
# ---------------------------------------------------------------------------

_GATE_S1_NOW, _GATE_S1_LAST = 10_000_000_000, 1_000_000_000  # clearly over gate; RShift collapses to 0
_GATE_S2_NOW, _GATE_S2_LAST = 2**32, 2**32 - 1  # real diff = 1 (under gate); XOR diff >> gate
_GATE_S3_NOW, _GATE_S3_LAST = 1_000_000_000, 0  # diff == gate exactly (boundary)
_GATE_S4_NOW, _GATE_S4_LAST = 1_000_000_001, 0  # diff == gate + 1


def test_log_stale_gate_kills_rshift_mutant(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._stale_log_last_ns = _GATE_S1_LAST
    with caplog.at_level(logging.WARNING):
        checker._log_stale(_GATE_S1_NOW, active_count=1, last_change_ns=1)
    assert len(caplog.records) == 1


def test_log_stale_gate_kills_bitxor_mutant(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._stale_log_last_ns = _GATE_S2_LAST
    with caplog.at_level(logging.WARNING):
        checker._log_stale(_GATE_S2_NOW, active_count=1, last_change_ns=1)
    assert len(caplog.records) == 0


def test_log_stale_gate_boundary_kills_offbyone_mutants(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._stale_log_last_ns = _GATE_S3_LAST
    with caplog.at_level(logging.WARNING):
        checker._log_stale(_GATE_S3_NOW, active_count=1, last_change_ns=1)
    assert len(caplog.records) == 0


def test_log_stale_gate_kills_plus_one_mutant(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._stale_log_last_ns = _GATE_S4_LAST
    with caplog.at_level(logging.WARNING):
        checker._log_stale(_GATE_S4_NOW, active_count=1, last_change_ns=1)
    assert len(caplog.records) == 1


def test_log_skip_gate_kills_rshift_mutant(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._skip_log_last_ns = _GATE_S1_LAST
    with caplog.at_level(logging.INFO):
        checker._log_skip(_GATE_S1_NOW, active_count=1)
    assert len(caplog.records) == 1


def test_log_skip_gate_kills_bitxor_mutant(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._skip_log_last_ns = _GATE_S2_LAST
    with caplog.at_level(logging.INFO):
        checker._log_skip(_GATE_S2_NOW, active_count=1)
    assert len(caplog.records) == 0


def test_log_skip_gate_boundary_kills_offbyone_mutants(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._skip_log_last_ns = _GATE_S3_LAST
    with caplog.at_level(logging.INFO):
        checker._log_skip(_GATE_S3_NOW, active_count=1)
    assert len(caplog.records) == 0


def test_log_skip_gate_kills_plus_one_mutant(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    checker._skip_log_last_ns = _GATE_S4_LAST
    with caplog.at_level(logging.INFO):
        checker._log_skip(_GATE_S4_NOW, active_count=1)
    assert len(caplog.records) == 1


# ---------------------------------------------------------------------------
# _log_stale message age arithmetic (line 284): (now_ns - last_change_ns) /
# 1e9, formatted as %.1fs. Two scenarios: one with a large last_change_ns to
# cheaply diverge every bitwise/arithmetic-corruption mutant, one with a
# tiny last_change_ns (2 ns) so the Pow/LShift mutants stay computationally
# tractable (a large exponent/shift-amount would otherwise hang or exhaust
# memory when the mutation is transiently applied).
# ---------------------------------------------------------------------------

_MSG_S1_NOW, _MSG_S1_LAST = 13_300_000_000, 3_000_000_000  # age = 10.3s
_MSG_S2_NOW, _MSG_S2_LAST = 3_300_000_002, 2  # age = 3.3s; tiny LC keeps **/<< tractable


def test_log_stale_message_age_scenario1_kills_arithmetic_mutants(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    with caplog.at_level(logging.WARNING):
        checker._log_stale(_MSG_S1_NOW, active_count=1, last_change_ns=_MSG_S1_LAST)
    assert len(caplog.records) == 1
    assert "age=10.3s" in caplog.records[0].getMessage()


def test_log_stale_message_age_scenario2_kills_pow_and_shift_mutants(caplog):
    checker = CheckerBarrier(enabled=True, stale_ns=1)
    with caplog.at_level(logging.WARNING):
        checker._log_stale(_MSG_S2_NOW, active_count=1, last_change_ns=_MSG_S2_LAST)
    assert len(caplog.records) == 1
    assert "age=3.3s" in caplog.records[0].getMessage()


# ---------------------------------------------------------------------------
# HolderBarrier keyword-only separators (lines 312, 328, 350).
# ---------------------------------------------------------------------------


def test_acquire_log_fn_is_keyword_only_kills_separator_mutant():
    holder = HolderBarrier(enabled=True, settle_frames=1, port=FakeShmAdapter())
    with pytest.raises(TypeError):
        holder.acquire(1, lambda *a, **k: None)  # log_fn must be keyword-only


def test_tick_and_maybe_release_log_fn_is_keyword_only_kills_separator_mutant():
    holder = HolderBarrier(enabled=True, settle_frames=1, port=FakeShmAdapter())
    with pytest.raises(TypeError):
        holder.tick_and_maybe_release(1, lambda *a, **k: None)  # log_fn must be keyword-only


def test_force_release_log_fn_is_keyword_only_kills_separator_mutant():
    holder = HolderBarrier(enabled=True, settle_frames=1, port=FakeShmAdapter())
    with pytest.raises(TypeError):
        holder.force_release(1, lambda *a, **k: None)  # log_fn must be keyword-only


# ---------------------------------------------------------------------------
# HolderBarrier.acquire force= kwarg (lines 319, 321).
# ---------------------------------------------------------------------------


def test_acquire_success_logs_with_force_true_kills_force_mutant():
    calls: list = []
    holder = HolderBarrier(enabled=True, settle_frames=1, port=FakeShmAdapter())
    holder.acquire(pid=1, log_fn=_make_recording_log(calls))
    assert calls[-1][1] is True


def test_acquire_failure_logs_with_force_true_kills_force_mutant():
    calls: list = []
    holder = HolderBarrier(enabled=True, settle_frames=1, port=_FailingHolderPort())
    holder.acquire(pid=1, log_fn=_make_recording_log(calls))
    assert calls[-1][1] is True


# ---------------------------------------------------------------------------
# HolderBarrier.tick_and_maybe_release settle_remaining boundary (lines 333,
# 336) and force= kwarg (lines 341, 345).
# ---------------------------------------------------------------------------


def test_tick_release_early_return_at_zero_kills_boundary_mutants():
    holder = HolderBarrier(enabled=True, settle_frames=3, held=True, settle_remaining=0, port=FakeShmAdapter())
    fired = holder.tick_and_maybe_release(pid=1, log_fn=lambda *a, **k: None)
    assert fired is False
    assert holder.settle_remaining == 0  # gate must short-circuit before any decrement


def test_tick_release_negative_settle_remaining_kills_eq_mutant():
    holder = HolderBarrier(enabled=True, settle_frames=3, held=True, settle_remaining=-1, port=FakeShmAdapter())
    fired = holder.tick_and_maybe_release(pid=1, log_fn=lambda *a, **k: None)
    assert fired is False
    assert holder.settle_remaining == -1  # gate must short-circuit; must not decrement further


def test_tick_release_success_logs_with_force_true_kills_force_mutant():
    calls: list = []
    holder = HolderBarrier(enabled=True, settle_frames=1, held=True, settle_remaining=1, port=FakeShmAdapter())
    fired = holder.tick_and_maybe_release(pid=1, log_fn=_make_recording_log(calls))
    assert fired is True
    assert calls[-1][1] is True


def test_tick_release_failure_logs_with_force_true_kills_force_mutant():
    calls: list = []
    holder = HolderBarrier(enabled=True, settle_frames=1, held=True, settle_remaining=1, port=_FailingHolderPort())
    fired = holder.tick_and_maybe_release(pid=1, log_fn=_make_recording_log(calls))
    assert fired is False
    assert calls[-1][1] is True


# ---------------------------------------------------------------------------
# HolderBarrier.force_release force= kwarg (lines 358, 361).
# ---------------------------------------------------------------------------


def test_force_release_success_logs_with_force_true_kills_force_mutant():
    calls: list = []
    holder = HolderBarrier(enabled=True, settle_frames=1, held=True, port=FakeShmAdapter())
    holder.force_release(pid=1, log_fn=_make_recording_log(calls))
    assert calls[-1][1] is True


def test_force_release_failure_logs_with_force_true_kills_force_mutant():
    calls: list = []
    holder = HolderBarrier(enabled=True, settle_frames=1, held=True, port=_FailingHolderPort())
    holder.force_release(pid=1, log_fn=_make_recording_log(calls))
    assert calls[-1][1] is True
