# Multi-stream sync kit (cuda-link 2.0.0.dev0, Phase 5)

This kit is for two TouchDesigner instances on one machine:

- Instance **A** sends S streams through C++ `CudaLinkOutTOP`s, for example 3 Orbbec
  cameras, each sending color and depth (S = 6).
- Instance **B** receives them with C++ `CudaLinkInTOP`s.

Every Out TOP that cooks in one A frame stamps the same `producer_frame`. In each cook, B
picks for every stream either the live frame or the frame before it, so that all S
streams B uses show the same `producer_frame`. The kit also logs a run, so you can check
the Phase 5 limits.

| File | Use |
| --- | --- |
| `sync_rule.py` | Script CHOP callbacks: the sync rule (pure functions `choose_sources`, `step`; tests in `tests/core/test_sync_rule_example.py`) |
| `run_logger.py` | Execute DAT callbacks: one log row per interval, to a Table DAT `runlog` and a CSV in `project.folder` |
| `build_kit.py` | Builds the B network (`build_receiver`) and the A logger (`build_sender_logger`) |

## The sync rule

For each B cook, `pf[s]` is the `producer_frame` of stream s (In TOP Info CHOP, mod 2^24).
Let `t = min(pf)`. Then for each stream:

- If `pf[s] == t`, use the live In TOP output.
- If `pf[s] == t + 1` and the stream's previous frame was `t`, use the cached frame.
- In all other cases, use the live output and count this B cook as a **mismatch**.

Each stream has a Cache TOP `cache{s}` (Cache Size 2, Output Index -1, Always Cook On).
Its `Active` parameter is set by `sync`'s `new{s}` channel, so it captures only when that
stream gets a new frame. Output Index -1 is then the stream's previous frame.

## Build it

The DLL paths are absolute. Use the DLLs you install in the project's `Plugins/` folder.
Keep cuda-link 1.14.1 off TD's module path.

**Instance B** (Textport or a Text DAT):

```python
K = r"<repo>/examples/td_multistream_sync"
g = {
    n: globals()[n]
    for n in (
        "textDAT",
        "baseCOMP",
        "scriptCHOP",
        "cplusplusTOP",
        "infoCHOP",
        "cacheTOP",
        "switchTOP",
        "outTOP",
        "executeDAT",
        "run",
        "op",
    )
}
exec(open(K + "/build_kit.py", encoding="utf-8").read(), g)
g["build_receiver"](
    op("/project1"),
    streams=6,
    in_dll=r"<Plugins>/CudaLinkInTOP.dll",
    names=["cam0_color", "cam0_depth", "cam1_color", "cam1_depth", "cam2_color", "cam2_depth"],
    kit_dir=K,
)
```

This makes `/project1/sync_receiver_B`. For each stream s it holds these ops:

- `in{s}`: the In TOP.
- `info{s}`: an Info CHOP with Passive Off.
- `cache{s}`: the Cache TOP described above.
- `sw{s}`: a Switch TOP, with Index = `op('sync')['sel{s}']`.
- `out{s}`: an Out TOP.

The COMP also holds `sync`, `run_exec` and the custom parameters `Interval` (seconds per
log row) and `Framewaitms` (copied to every In TOP). Use `out{s}` (the aligned frames) in
your effects network.

**Instance A**:

- Each stream s needs its own Out TOP, with `Ipcmemname` = `names[s]` and `Streamid` = s.
- All six Out TOPs must cook in the same A frame.
- Add the drop logger to the COMP that holds them:

```python
g["build_sender_logger"](op("/project1/<sender COMP>"), kit_dir=K)
```

## Run and read it

Before the run:

- Run A and B at the same root FPS (60), with Realtime On and Debug Off on every TOP.
- Run for 30 minutes. Each instance writes `cudalink_run_<role>_<time>.csv` into its project folder.

The run passes when both of these hold:

- **mismatch_pct** (B CSV) is at or below 0.1 %. The denominator is the number of B cooks
  that got at least one new frame.
- **Gap growth** is at or below 0.5 % of frames, after you remove A's own drops:
  (B `gap_sum` / S - A `drops`) / A `frames`.

Phase 6 starts if `mismatch_pct` > 1 %, or if the Cache TOPs use more than 1 GB of VRAM
(2 × the size of one frame per stream).

Also record:

- `latency_mean_us`
- `in_cook_mean_us`
- `check_ipc_capability()`
- each TOP's `init_note` Info DAT row
- the driver version
- Hardware-accelerated GPU scheduling (on or off)

If `gap_sum` grows fast, set `Framewaitms` on the receiver COMP (try 2, then 4). This
makes each In TOP wait for a new frame on a cook that has none. See the dev-box results
below.

## Dev-box results (RTX 4090, Noise TOPs in place of cameras)

Setup:

- TouchDesigner 2025.33230 on one machine.
- Instance A sends 6 Noise TOP streams; instance B receives them.
- Both run at 60 FPS.
- Another GPU-heavy job ran at the same time (a separate build and test session), so these
  numbers are a worst case, not a clean baseline.
- Each trial is short (40 s to 80 s), not the 30-minute Phase 5 run.

| `Framewaitms` | B frames | mismatch_pct | frame gaps per stream | B FPS |
| --- | --- | --- | --- | --- |
| 0 (off) | 4,804 | 5.04 % | 37.7 % of frames | 60.0 |
| 2 | 1,890 | 0.79 % | 2.5 % of frames | 60.0 |
| 4 | 2,595 | 0.19 % | 0.40 % of frames | 60.0 |

Other values (4 ms trial):

- A dropped 3 frames in 4,801.
- In TOP cook time was 56 µs to 123 µs per TOP.
- `latency_us` was 61 µs to 171 µs per stream.

Results:

- Without `Framewaitms`, B often cooks before A has published, then skips one frame. The
  sync rule cannot repair these skips, so they become mismatches.
- With `Framewaitms` = 4, the mismatch rate is near the 0.1 % limit, and B stays at 60 FPS.
- Start the show-machine run with `Framewaitms` = 4. If B drops frames, use 2.
