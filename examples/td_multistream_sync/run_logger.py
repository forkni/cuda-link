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
  The Phase 5 gap threshold is net of these A drops.

Rows go to a Table DAT `runlog` in the same COMP and, after each row, to
<project.folder>/cudalink_run_<role>_<start>.csv. Interval: custom par `Interval` (seconds)
on the COMP if present, else 60. Stream count for B: the `sync` CHOP's `Streams` par.

The A-side drop count uses absTime.step: TD sets it above 1 when a frame is skipped to
keep real time (Realtime On). Run A and B at the same root FPS (Decisions precondition).
"""

from __future__ import annotations

import time

_run: dict = {}


def _reset(owner: object) -> None:
    _run.clear()
    _run.update(
        start=time.time(),
        last=time.time(),
        role="receiver" if owner.op("sync") is not None else "sender",
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


def onStart() -> None:
    _run.clear()


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
        n = int(sync.par.Streams.eval())
        for s in range(n):
            info = owner.op(f"info{s}")
            if info is None:
                continue
            lat = float(info["latency_us"].eval())
            if lat > 0:
                _run["lat_sum"] += lat
                _run["lat_n"] += 1
                _run["lat_max"] = max(_run["lat_max"], lat)
            _run["cook_sum"] += float(info["cook_us"].eval())
            _run["cook_n"] += 1

    now = time.time()
    if now - _run["last"] < _interval(owner):
        return
    _run["last"] = now
    row = {"elapsed_s": round(now - _run["start"], 1), "frames": _run["frames"], "drops": _run["drops"]}
    if sync is not None:
        n = int(sync.par.Streams.eval())
        new_cooks = float(sync["new_cooks"].eval())
        mismatches = float(sync["mismatches"].eval())
        gaps = skips = 0.0
        for s in range(n):
            info = owner.op(f"info{s}")
            if info is not None:
                gaps += float(info["frame_gap_count"].eval())
                skips += float(info["producer_skip_count"].eval())
        row.update(
            b_cooks=int(float(sync["b_cooks"].eval())),
            new_cooks=int(new_cooks),
            mismatches=int(mismatches),
            mismatch_pct=round(100.0 * mismatches / new_cooks, 4) if new_cooks else 0.0,
            gap_sum=int(gaps),
            skip_sum=int(skips),
            latency_mean_us=round(_run["lat_sum"] / _run["lat_n"], 1) if _run["lat_n"] else 0.0,
            latency_max_us=round(_run["lat_max"], 1),
            in_cook_mean_us=round(_run["cook_sum"] / _run["cook_n"], 1) if _run["cook_n"] else 0.0,
        )
        _run.update(lat_sum=0.0, lat_n=0, lat_max=0.0, cook_sum=0.0, cook_n=0)
    _log_row(owner, row)
