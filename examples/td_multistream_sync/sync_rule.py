# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Multi-stream sync rule for a receiving TouchDesigner instance (B), as a Script CHOP.

Instance A sends S streams (for example 3 Orbbec cameras x color/depth) through C++ Out
TOPs; every Out TOP that cooks in one A frame stamps the same producer_frame. Instance B
receives them with S In TOPs. A B cook can land between two A publishes, so some streams
can be one frame ahead of the others. This rule picks, per stream, the live In TOP output
or the frame from the previous B cook (a Cache TOP), so that all streams B uses in one
cook show the same producer_frame.

The rule, per B cook (pf[s] = producer_frame of stream s, from its In TOP Info CHOP):

    t = min(pf)
    pf[s] == t                          -> live      (sel = 0)
    pf[s] == t + 1 and prev[s] == t     -> cached    (sel = 1)
    otherwise                           -> live, and this B cook is a mismatch

prev[s] is the producer_frame of the stream's previous distinct frame: the frame its Cache
TOP outputs. The Cache TOP captures only when that stream gets a new frame (Active driven
by the `new{s}` channel below), so Output Index -1 is the previous distinct frame, not the
previous B cook. Values are compared mod 2^24: In TOP Info CHOP
channels are float32, so producer_frame there is producer_frame mod 2^24.

Network, inside one COMP, per stream s = 0..S-1 (names are fixed):

    in{s}     CudaLinkInTOP (C++ In TOP)
    info{s}   Info CHOP, Operator = in{s}, Passive Off
    cache{s}  Cache TOP, input in{s}, Cache Size 2, Output Index -1, Always Cook On,
              Active = op('sync')['new{s}']
    sw{s}     Switch TOP, input 0 = in{s}, input 1 = cache{s},
              Index = op('sync')['sel{s}']

    sync      Script CHOP with this file as its Callbacks DAT, custom Int par `Streams` = S;
              an Execute DAT force-cooks it at every frame start (run_logger.py onFrameStart)

Output channels of `sync` (one sample):

    sel0..sel{S-1}   0 = live, 1 = cached
    new0..new{S-1}   1 when stream s got a new frame this cook (drives cache{s} Active)
    t                min(pf) mod 2^24
    mismatch         1 when the rule cannot align the frames chosen in this cook
    new_frame        1 when at least one stream got a new frame this cook
    b_cooks, new_cooks, mismatches   session counters (reset with the `Reset` pulse par)

The Phase 5 threshold counts mismatches / new_cooks (B cooks with at least one new frame).
"""

from __future__ import annotations

FRAME_MOD = 1 << 24  # Info CHOP producer_frame is producer_frame mod 2^24 (float32-exact)
LIVE = 0
CACHED = 1


def frame_delta(a: int, b: int) -> int:
    """Signed difference a - b of two producer_frame values mod 2^24 (in -2^23 .. 2^23-1)."""
    d = (a - b) % FRAME_MOD
    return d - FRAME_MOD if d >= FRAME_MOD // 2 else d


def choose_sources(pf: list[int], prev: list[int | None]) -> tuple[list[int], bool, int]:
    """Apply the sync rule to one B cook.

    Args:
        pf: producer_frame (mod 2^24) of each stream's live In TOP output.
        prev: producer_frame of each stream at the previous B cook (the cached frame),
            or None when unknown (first cook, after a reset).

    Returns:
        (sel, mismatch, t): sel[s] is LIVE or CACHED, mismatch is True when the rule
        cannot make all chosen frames equal, t is min(pf) mod 2^24.
    """
    if not pf:
        return [], False, 0
    ref = pf[0]
    t = min(pf, key=lambda v: frame_delta(v, ref)) % FRAME_MOD
    sel: list[int] = []
    mismatch = False
    for live, cached in zip(pf, prev):
        d = frame_delta(live, t)
        if d == 0:
            sel.append(LIVE)
        elif d == 1 and cached is not None and frame_delta(cached, t) == 0:
            sel.append(CACHED)
        else:
            sel.append(LIVE)
            mismatch = True
    return sel, mismatch, t


# ---------------------------------------------------------------------------
# Script CHOP callbacks (TouchDesigner). State lives in this module, not in op storage,
# so nothing is saved into the .toe.
# ---------------------------------------------------------------------------

_state: dict = {}


def _reset_state() -> None:
    _state.clear()
    _state.update(frame=None, last_pf=None, prev=None, frames=None, out=None, b_cooks=0, new_cooks=0, mismatches=0)


_reset_state()


def step(state: dict, pf: list[int], frames: list[int]) -> dict:
    """Advance the rule by one B cook and return the output channels (pure; testable).

    frames[s] is the In TOP `frames` channel (copies so far) of stream s. The stream got a
    new frame in this cook when `frames` or the producer_frame changed. Both are checked:
    `frames` is float32 and stops changing by 1 above 2^24 (about 78 hours at 60 FPS),
    while producer_frame (mod 2^24) stays exact and differs for every publish.
    """
    n = len(pf)
    fresh = state["frames"] is None or len(state["frames"]) != n
    if fresh:
        state["last_pf"] = [None] * n
        state["prev"] = [None] * n
        state["frames"] = [None] * n
    new = [frames[s] != state["frames"][s] or pf[s] != state["last_pf"][s] for s in range(n)]
    for s in range(n):
        if new[s]:
            # the frame that was live is now the stream's previous distinct frame (cache{s})
            state["prev"][s] = state["last_pf"][s]
            state["last_pf"][s] = pf[s]
            state["frames"][s] = frames[s]
    new_frame = any(new)
    sel, mismatch, t = choose_sources(pf, state["prev"])
    state["b_cooks"] += 1
    if new_frame:
        state["new_cooks"] += 1
        if mismatch:
            state["mismatches"] += 1
    out = {f"sel{s}": float(v) for s, v in enumerate(sel)}
    out.update({f"new{s}": float(v) for s, v in enumerate(new)})
    out.update(
        t=float(t),
        mismatch=float(mismatch),
        new_frame=float(new_frame),
        b_cooks=float(state["b_cooks"]),
        new_cooks=float(state["new_cooks"]),
        mismatches=float(state["mismatches"]),
    )
    return out


def onSetupParameters(scriptOp: object) -> None:
    page = scriptOp.appendCustomPage("Sync")
    p = page.appendInt("Streams", label="Streams")[0]
    p.default = 6
    p.min = 1
    p.normMax = 12
    p.clampMin = True
    page.appendPulse("Reset", label="Reset Counters")


def onPulse(par: object) -> None:
    if par.name == "Reset":
        _reset_state()


def _chan(chop: object, name: str) -> int:
    """Channel value as int; 0 when the CHOP or the channel is missing.

    The Info CHOP has no producer_frame / frames channels until the In TOP plugin has
    loaded (or when its DLL path is wrong); the rule must not raise on those cooks.
    """
    chan = chop[name] if chop is not None else None
    return int(chan.eval()) if chan is not None else 0


def onCook(scriptOp: object) -> None:
    owner = scriptOp.parent()
    now = int(absTime.frame)  # noqa: F821 -- TD global; one rule step per TD frame
    if _state["frame"] != now:
        n = int(scriptOp.par.Streams.eval())
        pf: list[int] = []
        frames: list[int] = []
        for s in range(n):
            info = owner.op(f"info{s}")
            pf.append(_chan(info, "producer_frame"))
            frames.append(_chan(info, "frames"))
        _state["frame"] = now
        _state["out"] = step(_state, pf, frames)
    scriptOp.clear()
    scriptOp.numSamples = 1
    for name, value in _state["out"].items():
        scriptOp.appendChan(name)[0] = value
