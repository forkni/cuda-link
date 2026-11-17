# SPDX-FileCopyrightText: 2026 Alexander Korneev
# SPDX-License-Identifier: MIT
"""
Build the cuda-link multi-stream sync kit inside TouchDesigner (Phase 5).

Run it from a Text DAT (right-click > Run Script) or the Textport, in each instance:

    kit = mod('build_kit')                 # or the exec() form in README.md (once per session)
    # Instance B (receiver): S In TOPs + Info CHOPs + Cache/Switch per stream + sync CHOP
    kit.build_receiver(op('/project1'), streams=6, in_dll=r'<path>/CudaLinkInTOP.dll',
                       names=['cam0_color', 'cam0_depth', ...], kit_dir=r'<this folder>')
    # Instance A (sender): only the run logger (Out TOPs stay in your own network)
    kit.build_sender_logger(op('/project1/my_sender_comp'), kit_dir=r'<this folder>')

`names[s]` is the Ipcmemname of stream s (the matching Out TOP in A uses the same name and
Streamid = s). `kit_dir` is the folder that holds sync_rule.py and run_logger.py; the
builder reads them into Text DATs so the saved .toe does not depend on this folder.

The C++ plugin parameters appear one frame after the plugin loads, so the builder sets
Ipcmemname / Active / Framewaitms through run(..., delayFrames=...).

TD 2025.33230 keeps the code object of any new code that writes a parameter (forum report
1026657, defect A). So every parameter write here is in a function that is compiled one
time: run() gets the function `_apply_in_pars`, never a text of code, and a refresh calls
build_receiver() again instead of a new exec() of this file.

The receiver COMP gets custom pars: `Interval` (logger seconds, default 60) and
`Framewaitms` (copied to every In TOP; 0 = off).
"""

from __future__ import annotations

_PARAM_DELAY_FRAMES = 5


def _apply_in_pars(comp_path: str, streams: int, names: list[str]) -> None:
    """Set the plugin pars of the In TOPs (run() calls it after the plugin has loaded)."""
    c = op(comp_path)  # noqa: F821
    c.op("sync").par.Streams = streams
    for s, name in enumerate(names):
        t = c.op(f"in{s}")
        t.par.Ipcmemname = name
        t.par.Framewaitms.expr = "parent().par.Framewaitms"
        t.par.Active = True


def _text_dat(comp: object, name: str, path: str, x: int, y: int) -> object:
    dat = comp.op(name) or comp.create(textDAT, name)  # noqa: F821
    with open(path, encoding="utf-8") as f:
        dat.text = f.read()
    dat.nodeX, dat.nodeY = x, y
    return dat


def build_receiver(
    parent_comp: object,
    streams: int,
    in_dll: str,
    names: list[str],
    kit_dir: str,
    comp_name: str = "sync_receiver_B",
    framewait_ms: int = 0,
) -> object:
    """Create (or refresh) the receiver kit COMP and return it."""
    if len(names) != streams:
        raise ValueError(f"names has {len(names)} entries, expected {streams}")
    comp = parent_comp.op(comp_name) or parent_comp.create(baseCOMP, comp_name)  # noqa: F821
    page = comp.customPages[0] if comp.customPages else comp.appendCustomPage("Kit")
    if not hasattr(comp.par, "Interval"):
        page.appendFloat("Interval", label="Log Interval (s)")[0].default = 60.0
        comp.par.Interval = 60.0
    if not hasattr(comp.par, "Framewaitms"):
        page.appendInt("Framewaitms", label="Frame Wait (ms)")
    comp.par.Framewaitms = framewait_ms

    sync_cb = _text_dat(comp, "sync_rule", f"{kit_dir}/sync_rule.py", 1200, 200)
    sync = comp.op("sync")
    if sync is None:
        sync = comp.create(scriptCHOP, "sync")  # noqa: F821
        default_cb = comp.op("sync_callbacks")  # TD adds a template callbacks DAT; the kit has its own
        if default_cb is not None:
            default_cb.destroy()
    sync.par.callbacks = sync_cb
    sync.nodeX, sync.nodeY = 1200, 0
    sync.par.setuppars.pulse()  # runs onSetupParameters: adds the Streams / Reset pars

    for s in range(streams):
        y = -250 * s
        t = comp.op(f"in{s}") or comp.create(cplusplusTOP, f"in{s}")  # noqa: F821
        t.par.plugin = in_dll
        t.nodeX, t.nodeY = 0, y
        info = comp.op(f"info{s}") or comp.create(infoCHOP, f"info{s}")  # noqa: F821
        info.par.op = t
        info.par.passive = False
        info.nodeX, info.nodeY = 200, y - 100
        cache = comp.op(f"cache{s}") or comp.create(cacheTOP, f"cache{s}")  # noqa: F821
        cache.inputConnectors[0].connect(t)
        cache.par.cachesize = 2
        cache.par.outputindex = -1
        cache.par.alwayscook = True
        cache.par.active.expr = f"op('sync')['new{s}']"
        cache.nodeX, cache.nodeY = 200, y
        sw = comp.op(f"sw{s}") or comp.create(switchTOP, f"sw{s}")  # noqa: F821
        sw.inputConnectors[0].connect(t)
        sw.inputConnectors[1].connect(cache)
        sw.par.index.expr = f"op('sync')['sel{s}']"
        sw.nodeX, sw.nodeY = 400, y
        out = comp.op(f"out{s}") or comp.create(outTOP, f"out{s}")  # noqa: F821
        out.inputConnectors[0].connect(sw)
        out.nodeX, out.nodeY = 600, y

    run(  # noqa: F821
        _apply_in_pars,
        comp.path,
        streams,
        list(names),
        delayFrames=_PARAM_DELAY_FRAMES,
    )
    for s in range(streams):
        comp.op(f"in{s}").cook(force=True)  # load the plugin now; its pars appear next frame

    logger = _text_dat(comp, "run_logger", f"{kit_dir}/run_logger.py", 1200, -200)
    ex = comp.op("run_exec") or comp.create(executeDAT, "run_exec")  # noqa: F821
    ex.text = logger.text
    ex.par.framestart = True  # force-cooks `sync` once per B frame (reads every In TOP together)
    ex.par.frameend = True
    ex.par.start = True
    ex.nodeX, ex.nodeY = 1200, -400
    return comp


def build_sender_logger(sender_comp: object, kit_dir: str) -> object:
    """Add the A-side run logger (frames, dropped frames) to an existing COMP."""
    page = sender_comp.customPages[0] if sender_comp.customPages else sender_comp.appendCustomPage("Kit")
    if not hasattr(sender_comp.par, "Interval"):
        page.appendFloat("Interval", label="Log Interval (s)")[0].default = 60.0
        sender_comp.par.Interval = 60.0
    ex = sender_comp.op("run_exec") or sender_comp.create(executeDAT, "run_exec")  # noqa: F821
    with open(f"{kit_dir}/run_logger.py", encoding="utf-8") as f:
        ex.text = f.read()
    ex.par.framestart = False
    ex.par.frameend = True
    ex.par.start = True
    ex.nodeX, ex.nodeY = 1200, -400
    return ex
