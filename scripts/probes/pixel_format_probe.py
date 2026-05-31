"""
TD Pixel Format Probe — paste sections into TouchDesigner's Python textport.

PURPOSE
-------
Enumerate every pixelFormatName / pixelFormat string that TD exposes, and show
what cm.data_type / cm.size cuda_memory() reports for each format on ExportBuffer.

This gives us the exact pixelFormatName → (dtype, num_channels) mapping table
needed to fix the 32bit→8bit dtype-change detection bug in TDSender.

HOW TO USE
----------
1. With the sender Active and streaming, paste SECTION 1 into the TD textport and
   press Enter. Note the full menu list of par.format names.

2. Change the source texture's pixel format (e.g. float32 → uint8), then paste
   SECTION 2 and press Enter. Capture the pixelFormatName + cm.* values.

3. Repeat step 2 for each format you want to probe (uint8, uint16, float32, etc.)
   and for both directions (shrink: float32→uint8, grow: uint8→float32).

4. Paste the captured output back to Claude for the mapping table.

PATHS TO ADJUST
---------------
Adjust SCRIPT_TOP_PATH and EXPORT_BUFFER_PATH to match your project layout.
The ImportBuffer Script TOP is in the Consumer (.tox); ExportBuffer is in Producer.
"""

# ── SECTION 1: enumerate all Script TOP par.format menu names ─────────────
# Run once to get the full list of pixelFormatName values TD exposes.
# Point this at any Script TOP — ImportBuffer inside the Consumer .tox works.

SCRIPT_TOP_PATH = "/project1/CUDAIPCLink_to_Touchdesigner/ExportBuffer"

_top = op(SCRIPT_TOP_PATH)  # noqa: F821 — op() is a TouchDesigner global
if _top is None:
    print(f"[probe] Script TOP not found: {SCRIPT_TOP_PATH!r}")
    print("        Adjust SCRIPT_TOP_PATH to a Script TOP in your project.")
else:
    _fmt = _top.par.format
    _names = list(_fmt.menuNames)
    _labels = list(_fmt.menuLabels)
    print(f"[probe] Script TOP par.format menu — {len(_names)} entries:")
    print(f"  {'menuName (pixelFormatName)':<28} {'menuLabel (display)'}")
    print(f"  {'-' * 28} {'-' * 40}")
    for n, label in zip(_names, _labels):
        print(f"  {n:<28} {label}")
    print("\n[probe] Current TOP values:")
    print(f"  pixelFormatName = {_top.pixelFormatName!r}")
    print(f"  pixelFormat     = {_top.pixelFormat!r}")


# ── SECTION 2: probe ExportBuffer live cm values + pixelFormatName ─────────
# Run AFTER changing the source texture's pixel format.
# Point this at the ExportBuffer Script TOP inside the Producer (.tox).

EXPORT_BUFFER_PATH = "/project1/CUDAIPCLink_to_Touchdesigner/ExportBuffer"

_src = op(EXPORT_BUFFER_PATH)  # noqa: F821 — op() is a TouchDesigner global
if _src is None:
    print(f"[probe] ExportBuffer not found: {EXPORT_BUFFER_PATH!r}")
    print("        Adjust EXPORT_BUFFER_PATH to the ExportBuffer Script TOP.")
else:
    print(f"\n[probe] ExportBuffer: {_src.path}")
    print(f"  pixelFormatName = {_src.pixelFormatName!r}")
    print(f"  pixelFormat     = {_src.pixelFormat!r}")
    try:
        _cm = _src.cudaMemory()
        _dt_name = getattr(_cm.data_type, "name", None) if _cm.data_type is not None else None
        print(f"  cm.width={_cm.width}  cm.height={_cm.height}  cm.channels={_cm.channels}")
        print(f"  cm.size={_cm.size}  cm.data_type.name={_dt_name!r}")
        _px = _cm.width * _cm.height * _cm.channels
        if _px > 0:
            _bpp = _cm.size / _px
            print(f"  bytes/component = {_bpp:.2f}  (uint8→1.0, uint16→2.0, float32→4.0)")
            _expected = {
                "uint8": _px * 1,
                "uint16": _px * 2,
                "float32": _px * 4,
            }
            print(f"  [expected sizes: {', '.join(f'{k}={v}' for k, v in _expected.items())}]")
    except Exception as _e:
        print(f"  cudaMemory() error: {_e}")
