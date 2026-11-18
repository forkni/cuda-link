"""The multi-stream sync rule example (examples/td_multistream_sync/sync_rule.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "examples" / "td_multistream_sync" / "sync_rule.py"
_spec = importlib.util.spec_from_file_location("sync_rule_example", _PATH)
assert _spec is not None and _spec.loader is not None
sync_rule = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_rule)

LIVE, CACHED, MOD = sync_rule.LIVE, sync_rule.CACHED, sync_rule.FRAME_MOD


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(10, 7, 3), (7, 10, -3), (0, MOD - 1, 1), (MOD - 1, 0, -1), (5, 5, 0)],
    ids=["ahead", "behind", "ahead_across_wrap", "behind_across_wrap", "equal"],
)
def test_frame_delta_wraps(a: int, b: int, expected: int) -> None:
    assert sync_rule.frame_delta(a, b) == expected


def test_all_equal_uses_live() -> None:
    assert sync_rule.choose_sources([100, 100, 100], [99, 99, 99]) == ([LIVE] * 3, False, 100)


def test_one_ahead_with_matching_cache_uses_cache() -> None:
    sel, mismatch, t = sync_rule.choose_sources([100, 101, 100], [99, 100, 99])
    assert (sel, mismatch, t) == ([LIVE, CACHED, LIVE], False, 100)


def test_several_ahead_use_cache() -> None:
    sel, mismatch, _ = sync_rule.choose_sources([101, 101, 100, 101], [100, 100, 99, 100])
    assert sel == [CACHED, CACHED, LIVE, CACHED]
    assert not mismatch


def test_ahead_but_cache_not_at_t_is_mismatch() -> None:
    # stream 1 jumped two frames since the last cook: its cache holds 99, not 100
    sel, mismatch, _ = sync_rule.choose_sources([100, 101], [99, 99])
    assert sel == [LIVE, LIVE]
    assert mismatch


def test_two_ahead_is_mismatch() -> None:
    _, mismatch, _ = sync_rule.choose_sources([100, 102], [99, 101])
    assert mismatch


def test_unknown_cache_is_mismatch() -> None:
    _, mismatch, _ = sync_rule.choose_sources([100, 101], [None, None])
    assert mismatch


def test_min_across_wrap() -> None:
    sel, mismatch, t = sync_rule.choose_sources([0, MOD - 1], [MOD - 1, MOD - 2])
    assert t == MOD - 1
    assert sel == [CACHED, LIVE]
    assert not mismatch


def _fresh_state() -> dict:
    sync_rule._reset_state()
    return dict(sync_rule._state)


def test_step_counts_only_cooks_with_a_new_frame() -> None:
    state = _fresh_state()
    out = sync_rule.step(state, [100, 100], [1, 1])  # first cook counts as new
    assert out["new_cooks"] == 1.0
    out = sync_rule.step(state, [100, 100], [1, 1])  # no new frame
    assert out["new_frame"] == 0.0
    assert out["new_cooks"] == 1.0
    out = sync_rule.step(state, [101, 103], [2, 2])  # 103 is two ahead: mismatch
    assert out["mismatch"] == 1.0
    assert out["mismatches"] == 1.0
    out = sync_rule.step(state, [103, 103], [3, 2])
    assert (out["sel0"], out["sel1"], out["mismatch"]) == (0.0, 0.0, 0.0)
    assert out["b_cooks"] == 4.0
    assert out["new_cooks"] == 3.0


def test_step_cache_holds_previous_distinct_frame_across_idle_cooks() -> None:
    state = _fresh_state()
    sync_rule.step(state, [100, 100], [1, 1])
    out = sync_rule.step(state, [100, 101], [1, 2])  # stream 1 one ahead, its cache holds 100
    assert (out["sel0"], out["sel1"], out["mismatch"]) == (0.0, 1.0, 0.0)
    assert (out["new0"], out["new1"]) == (0.0, 1.0)
    out = sync_rule.step(state, [100, 101], [1, 2])  # idle cook: same choice, cache still 100
    assert (out["sel0"], out["sel1"], out["mismatch"]) == (0.0, 1.0, 0.0)
    assert (out["new0"], out["new1"]) == (0.0, 0.0)
    out = sync_rule.step(state, [101, 101], [2, 2])  # stream 0 catches up
    assert (out["sel0"], out["sel1"], out["mismatch"]) == (0.0, 0.0, 0.0)


def test_step_resets_when_stream_count_changes() -> None:
    state = _fresh_state()
    sync_rule.step(state, [100, 100], [1, 1])
    out = sync_rule.step(state, [100, 100, 100], [1, 1, 1])
    assert out["new_frame"] == 1.0
    assert "sel2" in out


def test_step_detects_new_frame_when_frames_channel_is_stuck() -> None:
    """Above 2^24 the float32 `frames` channel repeats a value; producer_frame still changes."""
    state = _fresh_state()
    stuck = 1 << 24
    sync_rule.step(state, [100, 100], [stuck, stuck])
    out = sync_rule.step(state, [100, 101], [stuck, stuck])  # stream 1: new frame, same `frames`
    assert (out["new0"], out["new1"]) == (0.0, 1.0)
    assert (out["sel0"], out["sel1"], out["mismatch"]) == (0.0, 1.0, 0.0)  # its cache holds 100


class _Chan:
    def __init__(self, value: float) -> None:
        self._value = value

    def eval(self) -> float:
        return self._value


class _Chop(dict):
    """CHOP double: indexing a missing channel gives None, as in TouchDesigner."""

    def __getitem__(self, name: str) -> _Chan | None:
        return self.get(name)


def test_chan_is_zero_for_missing_chop_or_channel() -> None:
    assert sync_rule._chan(None, "producer_frame") == 0
    assert sync_rule._chan(_Chop(), "producer_frame") == 0  # plugin not loaded: no such channel
    assert sync_rule._chan(_Chop(producer_frame=_Chan(41.0)), "producer_frame") == 41


# ---------------------------------------------------------------------------
# run_logger.py (same kit): the pure counter arithmetic
# ---------------------------------------------------------------------------

_LOGGER_PATH = _PATH.with_name("run_logger.py")
_logger_spec = importlib.util.spec_from_file_location("run_logger_example", _LOGGER_PATH)
assert _logger_spec is not None and _logger_spec.loader is not None
run_logger = importlib.util.module_from_spec(_logger_spec)
_logger_spec.loader.exec_module(run_logger)


def _counters(**values: float) -> dict:
    return {**dict.fromkeys(run_logger._COUNTERS, 0.0), **values}


def test_logger_counters_are_relative_to_run_start() -> None:
    base = _counters(b_cooks=500.0, new_cooks=480.0, mismatches=30.0, gap_sum=12.0, skip_sum=2.0)
    now = _counters(b_cooks=1500.0, new_cooks=1480.0, mismatches=31.0, gap_sum=15.0, skip_sum=2.0)
    assert run_logger.since_start(now, base) == {
        "b_cooks": 1000,
        "new_cooks": 1000,
        "mismatches": 1,
        "gap_sum": 3,
        "skip_sum": 0,
    }


def test_logger_counter_reset_during_run_counts_from_zero() -> None:
    base = _counters(mismatches=30.0, new_cooks=480.0)
    now = _counters(mismatches=4.0, new_cooks=90.0)  # `Reset` pulsed on sync during the run
    out = run_logger.since_start(now, base)
    assert (out["mismatches"], out["new_cooks"]) == (4, 90)


def test_logger_session_counters_sum_streams_and_tolerate_missing_channels() -> None:
    import types

    infos = {
        "info0": _Chop(frame_gap_count=_Chan(3.0), producer_skip_count=_Chan(1.0)),
        "info1": _Chop(),  # plugin not loaded
    }
    owner = types.SimpleNamespace(op=infos.get)
    sync = _Chop(b_cooks=_Chan(10.0), new_cooks=_Chan(8.0), mismatches=_Chan(2.0))
    sync.par = types.SimpleNamespace(Streams=_Chan(3.0))  # info2 does not exist
    assert run_logger._session_counters(owner, sync) == {
        "b_cooks": 10.0,
        "new_cooks": 8.0,
        "mismatches": 2.0,
        "gap_sum": 3.0,
        "skip_sum": 1.0,
    }


def test_logger_kick_plugins_cooks_each_cplusplus_top_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sink Out TOP never loads its DLL after a project opens until something cooks it."""
    cooks: list[str] = []

    class _Top:
        def __init__(self, name: str) -> None:
            self.name = name

        def cook(self, force: bool = False) -> None:
            assert force
            cooks.append(self.name)

    marker = object()  # stands in for TD's cplusplusTOP type

    class _Comp:
        def findChildren(self, type: object, depth: int) -> list[_Top]:  # noqa: A002, N802 (TD API)
            assert type is marker and depth == 1
            return [_Top("out0"), _Top("out1")]

    monkeypatch.setattr(run_logger, "cplusplusTOP", marker, raising=False)
    assert run_logger.kick_plugins(_Comp()) == 2
    assert cooks == ["out0", "out1"]
