# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Execute DAT callbacks: log a multi-stream run once per interval (cuda-link Phase 5 kit).

Put this in an Execute DAT (Frame End On) inside the COMP that holds the network:

- Receiver COMP (B, has a `sync` Script CHOP): logs B cooks, cooks with a new frame,
  mismatches and their rate, the summed In TOP frame_gap_count / producer_skip_count,
  latency_us (mean and max over the interval), In TOP cook_us mean, and B's own drops.
  Frame Start must be On too: it force-cooks `sync` once per B frame.
- Sender COMP (A, no `sync`): logs A frames and A's dropped frames (absTime.step > 1).
  A frame that A drops is not a B frame gap: A does not publish in it, so the In TOPs
  count it as a producer skip. B `skip_sum` / streams is therefore near A `drops`.

The receiver counters (b_cooks, new_cooks, mismatches, gap_sum, skip_sum) are relative to
the start of the logged run: the logger reads the session counters of `sync` and the In
TOPs when it starts and subtracts them from every row.

Rows go to a Table DAT `runlog` in the same COMP (cleared when a run starts) and, after
each row, to <project.folder>/cudalink_run_<role>_<start>.csv. Interval: custom par
`Interval` (seconds) on the COMP if present, else 60. Stream count for B: the `sync` CHOP's
`Streams` par.

On start the logger cooks each C++ TOP of its COMP once (`kick_plugins`). TD loads a C++
TOP's DLL only on the op's first cook, so an Out TOP that nothing pulls (a sink, as in A)
never loads and never publishes after the project opens. One cook loads it, and then its
own cookEveryFrame keeps it cooking (measured on TD 2025.33230).

The A-side drop count uses absTime.step: TD sets it above 1 when a frame is skipped to
keep real time (Realtime On). Run A and B at the same root FPS (Decisions precondition).
"""

from __future__ import annotations

import time

_run: dict = {}

_COUNTERS = ("b_cooks", "new_cooks", "mismatches", "gap_sum", "skip_sum")


def _chan(chop: object, name: str) -> float:
    """Channel value; 0.0 when the CHOP or the channel is missing (plugin not loaded yet)."""
    chan = chop[name] if chop is not None else None
    return float(chan.eval()) if chan is not None else 0.0


def _session_counters(owner: object, sync: object) -> dict:
    """Session counters of `sync` and the In TOPs, as they are now."""
    gaps = skips = 0.0
    for s in range(int(sync.par.Streams.eval())):
        info = owner.op(f"info{s}")
        gaps += _chan(info, "frame_gap_count")
        skips += _chan(info, "producer_skip_count")
    return {
        "b_cooks": _chan(sync, "b_cooks"),
        "new_cooks": _chan(sync, "new_cooks"),
        "mismatches": _chan(sync, "mismatches"),
        "gap_sum": gaps,
        "skip_sum": skips,
    }


def since_start(now: dict, base: dict) -> dict:
    """Counters relative to the run start (pure; testable).

    A counter below its base was reset during the run (the `Reset` pulse on `sync`, or an
    In TOP that was recreated); it then counts from 0, so its value is used as it is.
    """
    return {key: int(now[key] - base[key]) if now[key] >= base[key] else int(now[key]) for key in _COUNTERS}


def _reset(owner: object) -> None:
    sync = owner.op("sync")
    _run.clear()
    _run.update(
        start=time.time(),
        last=time.time(),
        role="receiver" if sync is not None else "sender",
        base=_session_counters(owner, sync) if sync is not None else None,
        frames=0,
        drops=0,
        lat_sum=0.0,
        lat_n=0,
        lat_max=0.0,
        cook_sum=0.0,
        cook_n=0,
        path=None,
    )
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(_run["start"]))
    _run["path"] = f"{project.folder}/cudalink_run_{_run['role']}_{stamp}.csv"  # noqa: F821
    table = owner.op("runlog")
    if table is not None:
        table.clear()  # rows of an earlier run must not go into this run's CSV


def _interval(owner: object) -> float:
    par = getattr(owner.par, "Interval", None)
    try:
        return float(par.eval()) if par is not None else 60.0
    except (TypeError, ValueError):
        return 60.0


def _log_row(owner: object, row: dict) -> None:
    table = owner.op("runlog")
    if table is None:
        table = owner.create(tableDAT, "runlog")  # noqa: F821
        table.clear()  # a new Table DAT starts with one empty cell
    if table.numRows == 0 or table[0, 0].val != "elapsed_s":
        table.clear()
        table.appendRow(list(row))
    header = [cell.val for cell in table.row(0)]
    table.appendRow([row.get(key, "") for key in header])
    table.save(_run["path"])  # DAT.write() would append to the DAT itself


def kick_plugins(owner: object) -> int:
    """Cook each C++ TOP that is a direct child of `owner` once; return how many."""
    tops = owner.findChildren(type=cplusplusTOP, depth=1)  # noqa: F821
    for top in tops:
        top.cook(force=True)
    return len(tops)


def onStart() -> None:
    _run.clear()
    kick_plugins(parent())  # noqa: F821


def onFrameStart(frame: int) -> None:
    # Receiver: cook the sync rule once per B frame, whether or not anything downstream pulls
    # the Switch TOPs yet. It reads every In TOP's Info CHOP back to back at this one point.
    sync = parent().op("sync")  # noqa: F821
    if sync is not None:
        sync.cook(force=True)


def onFrameEnd(frame: int) -> None:
    owner = parent()  # noqa: F821
    if not _run:
        _reset(owner)
    _run["frames"] += 1
    step = int(absTime.step)  # noqa: F821
    if step > 1:
        _run["drops"] += step - 1

    sync = owner.op("sync")
    if sync is not None:
        for s in range(int(sync.par.Streams.eval())):
            info = owner.op(f"info{s}")
            # cook_us is new on every cook (a cook with no frame can hold the Framewaitms wait).
            _run["cook_sum"] += _chan(info, "cook_us")
            _run["cook_n"] += 1
            # latency_us changes only with a new frame: without one the channel still holds
            # the previous frame's value, which must not be counted again.
            lat = _chan(info, "latency_us") if _chan(sync, f"new{s}") > 0 else 0.0
            if lat > 0:
                _run["lat_sum"] += lat
                _run["lat_n"] += 1
                _run["lat_max"] = max(_run["lat_max"], lat)

    now = time.time()
    if now - _run["last"] < _interval(owner):
        return
    _run["last"] = now
    row = {"elapsed_s": round(now - _run["start"], 1), "frames": _run["frames"], "drops": _run["drops"]}
    if sync is not None:
        counters = since_start(_session_counters(owner, sync), _run["base"])
        new_cooks = counters["new_cooks"]
        row.update(counters)
        row.update(
            mismatch_pct=round(100.0 * counters["mismatches"] / new_cooks, 4) if new_cooks else 0.0,
            latency_mean_us=round(_run["lat_sum"] / _run["lat_n"], 1) if _run["lat_n"] else 0.0,
            latency_max_us=round(_run["lat_max"], 1),
            in_cook_mean_us=round(_run["cook_sum"] / _run["cook_n"], 1) if _run["cook_n"] else 0.0,
        )
        _run.update(lat_sum=0.0, lat_n=0, lat_max=0.0, cook_sum=0.0, cook_n=0)
    _log_row(owner, row)
