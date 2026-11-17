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
