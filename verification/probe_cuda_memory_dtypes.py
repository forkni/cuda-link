"""
probe_cuda_memory_dtypes.py — TouchDesigner CUDA dtype acceptance probe.

PURPOSE:
  Determine which TOP Pixel Formats TD's cudaMemory() actually accepts.
  This challenges the assumption in TDSender.py that float16 is rejected.

USAGE (two ways):
  Option A — Textport (recommended):
    1. In TD: Dialogs → Textport
    2. Paste this entire script and press Enter.

  Option B — Script DAT:
    1. Create a Script DAT, paste this script.
    2. Wrap the body in `def onCook(scriptOp): ...` and call run() from onCook.
    3. Or: right-click the DAT → Run Script.

OUTPUT:
  - Printed summary table in the Textport
  - JSON file written to:  verification/results/cuda_memory_probe_<timestamp>.json
    (path auto-detected from this file's location; falls back to the hardcoded path below)

NOTE: This script is read-only with respect to the project. It creates a single
temporary Constant TOP named '_cuda_probe_top' in /project1, runs the test, then
destroys it. Worst case: a Python exception in TD, fully contained.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Output path — auto-detect from __file__, else use the hardcoded project path
# ---------------------------------------------------------------------------
try:
    _this_file = Path(__file__).resolve()
    _out_dir = _this_file.parent / "results"
except NameError:
    # Pasted into Textport — __file__ is not defined
    _out_dir = Path("F:/RD_PROJECTS/COMPONENTS/cuda-link/verification/results")

_out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CUDA stream — non-blocking preferred (default stream=0 has known issues in
# TD 2025 / CUDA 12.8: can cause implicit sync on the main thread).
# cupy ships with TD; fall back to 0 only if unavailable.
# ---------------------------------------------------------------------------
_stream_obj = None
_stream_ptr = 0
try:
    import cupy as cp

    _stream_obj = cp.cuda.Stream(non_blocking=True)
    _stream_ptr = _stream_obj.ptr
    print(f"[probe] cupy stream: 0x{_stream_ptr:016x}")
except Exception as _e:
    print(
        f"[probe] cupy unavailable ({_e!r}) — using stream=0 (default). "
        "Results for float16 rows may be less reliable (may sync)."
    )

# ---------------------------------------------------------------------------
# Pixel format table — one entry per TD Pixel Formats menu item.
#
# Columns:
#   par_val   — string passed to top.par.format  (TD parameter menu value)
#   label     — human name matching the TD menu label exactly
#   channels  — channel count as TD reports (numComps in shape)
#   bpc       — bits per channel (int) or string descriptor for packed formats
#   expected  — the numpy dtype we predict cudaMemory() returns ("?" = unknown)
#
# Source: https://derivative.ca/UserGuide/Pixel_Formats
# CUDA cross-ref: driver CUarray_format / runtime cudaChannelFormatKind
#   ✓ low risk:  UNORM_INT8X{1,2,4}, UNORM_INT16X{1,2,4}, HALF×{1,2,4}, FLOAT×{1,2,4}
#   ⚠ uncertain: HALF×{1,2,4} — TDSender.py L58 claims rejected; CUDA says fully supported
#   ⚠⚠ packed:  10:10:10:2 and 11:11:10 have no first-class linear CUDA enum
# ---------------------------------------------------------------------------
FORMATS = [
    # par_val          label                              ch  bpc                expected
    ("rgba8fixed", "8-bit fixed (RGBA)", 4, 8, "uint8"),
    ("rgba16fixed", "16-bit fixed (RGBA)", 4, 16, "uint16"),
    ("rgba16float", "16-bit float (RGBA)", 4, 16, "float16"),  # ⚠ primary probe target
    ("rgba32float", "32-bit float (RGBA)", 4, 32, "float32"),  # reference
    ("rgb10a2fixed", "10-bit RGB / 2-bit A fixed", 4, "packed_10_10_10_2", "?"),  # ⚠ gray zone
    ("rgb11float", "11-bit float (RGB)", 3, "packed_11_11_10", "?"),  # ⚠⚠ highest risk
    ("r8fixed", "8-bit fixed (R)", 1, 8, "uint8"),
    ("r16fixed", "16-bit fixed (R)", 1, 16, "uint16"),
    ("r16float", "16-bit float (R)", 1, 16, "float16"),  # ⚠
    ("r32float", "32-bit float (R)", 1, 32, "float32"),
    ("rg8fixed", "8-bit fixed (RG)", 2, 8, "uint8"),
    ("rg16fixed", "16-bit fixed (RG)", 2, 16, "uint16"),
    ("rg16float", "16-bit float (RG)", 2, 16, "float16"),  # ⚠
    ("rg32float", "32-bit float (RG)", 2, 32, "float32"),
    ("a8fixed", "8-bit fixed (A)", 1, 8, "uint8"),
    ("a16float", "16-bit float (A)", 1, 16, "float16"),  # ⚠
    ("a32float", "32-bit float (A)", 1, 32, "float32"),
]

# ---------------------------------------------------------------------------
# Create a 64×64 Constant TOP as the probe surface.
# (64×64 is small enough to be fast but large enough to exercise the buffer.)
# ---------------------------------------------------------------------------
_PROBE_NAME = "_cuda_probe_top"
_PROBE_PARENT = op("/project1")

try:
    _existing = op(f"/project1/{_PROBE_NAME}")
    if _existing is not None and _existing.valid:
        probe_top = _existing
        _created = False
        print(f"[probe] Reusing existing TOP: {probe_top.path}")
    else:
        probe_top = _PROBE_PARENT.create(constantTOP, _PROBE_NAME)
        probe_top.par.outputresolution = "custom"
        probe_top.par.resolutionw = 64
        probe_top.par.resolutionh = 64
        _created = True
        print(f"[probe] Created Constant TOP: {probe_top.path}")
except Exception as _create_err:
    raise RuntimeError(
        f"[probe] Cannot create/find probe TOP in /project1: {_create_err!r}\n"
        "Is this a valid TD project with a /project1 component?"
    ) from _create_err

# ---------------------------------------------------------------------------
# Main probe loop
# ---------------------------------------------------------------------------
results = []

try:
    for i, (par_val, label, channels, bpc, expected_dtype) in enumerate(FORMATS, start=1):
        row = {
            "row": i,
            "par_value": par_val,
            "label": label,
            "channels_expected": channels,
            "bits_per_channel": bpc,
            "expected_dtype": expected_dtype,
            "actual_pixel_format_str": None,
            "outcome": None,  # "ok" | "exception" | "none_returned" | "param_set_failed"
            "shape_dataType": None,
            "shape_numComps": None,
            "shape_width": None,
            "shape_height": None,
            "mem_size_bytes": None,
            "size_matches_expected": None,
            "error_type": None,
            "error_msg": None,
            "traceback": None,
        }

        # --- Set pixel format ------------------------------------------------
        try:
            probe_top.par.outputresolution = "custom"
            probe_top.par.resolutionw = 64
            probe_top.par.resolutionh = 64
            probe_top.par.format = par_val
        except AttributeError as fmt_err:
            # Param name mismatch — log and skip this row, do not abort the whole probe
            row["outcome"] = "param_set_failed"
            row["error_type"] = type(fmt_err).__name__
            row["error_msg"] = str(fmt_err)
            results.append(row)
            print(f"[probe] [{i:2d}] {label:<38} PARAM-SET-FAILED  {fmt_err}")
            continue
        except Exception as fmt_err:
            row["outcome"] = "param_set_failed"
            row["error_type"] = type(fmt_err).__name__
            row["error_msg"] = str(fmt_err)
            results.append(row)
            print(f"[probe] [{i:2d}] {label:<38} PARAM-SET-FAILED  {fmt_err}")
            continue

        # Force cook so the format change takes effect before cudaMemory()
        probe_top.cook(force=True)
        row["actual_pixel_format_str"] = str(getattr(probe_top, "pixelFormat", "?"))

        # --- Call cudaMemory() -----------------------------------------------
        try:
            mem = probe_top.cudaMemory(_stream_ptr)
        except Exception as cuda_err:
            row["outcome"] = "exception"
            row["error_type"] = type(cuda_err).__name__
            row["error_msg"] = str(cuda_err)
            row["traceback"] = traceback.format_exc()
            results.append(row)
            print(f"[probe] [{i:2d}] {label:<38} EXCEPTION         {cuda_err}")
            continue

        if mem is None:
            row["outcome"] = "none_returned"
            results.append(row)
            print(f"[probe] [{i:2d}] {label:<38} NONE_RETURNED")
            continue

        # --- Success: harvest shape info ------------------------------------
        shape = mem.shape
        row["outcome"] = "ok"
        row["shape_dataType"] = shape.dataType.__name__ if hasattr(shape.dataType, "__name__") else str(shape.dataType)
        row["shape_numComps"] = shape.numComps
        row["shape_width"] = shape.width
        row["shape_height"] = shape.height
        row["mem_size_bytes"] = mem.size

        if isinstance(bpc, int):
            itemsize = bpc // 8
            expected_size = shape.width * shape.height * shape.numComps * itemsize
            row["size_matches_expected"] = mem.size == expected_size
            size_ok_str = "size_ok" if row["size_matches_expected"] else f"size_MISMATCH(expected {expected_size})"
        else:
            # Packed format: report raw bytes for manual analysis
            packed_note = f"packed {mem.size}B for {shape.width}x{shape.height}x{shape.numComps}"
            row["size_matches_expected"] = packed_note
            size_ok_str = packed_note

        print(
            f"[probe] [{i:2d}] {label:<38} OK  "
            f"dtype={row['shape_dataType']:<10} "
            f"numComps={shape.numComps}  {size_ok_str}"
        )
        results.append(row)

finally:
    # Destroy the scratch TOP (only if we created it this run)
    if _created:
        try:
            probe_top.destroy()
            print("[probe] Scratch TOP destroyed.")
        except Exception:
            print("[probe] Warning: could not destroy scratch TOP (may persist).")

# ---------------------------------------------------------------------------
# Write JSON results
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_out_path = _out_dir / f"cuda_memory_probe_{_ts}.json"
_payload = {
    "probe_timestamp": _ts,
    "stream_ptr": hex(_stream_ptr),
    "stream_type": "cupy_non_blocking" if _stream_ptr != 0 else "default_stream_0",
    "td_version": str(getattr(app, "version", "unknown")),
    "cuda_version": str(getattr(app, "cudaVersion", "unknown")),
    "results": results,
}
with open(_out_path, "w", encoding="utf-8") as _f:
    json.dump(_payload, _f, indent=2, default=str)
print(f"\n[probe] JSON written → {_out_path}")

# ---------------------------------------------------------------------------
# Print summary table to Textport
# ---------------------------------------------------------------------------
_SEP = "-" * 90
print(f"\n{'':=<90}")
print("CUDA MEMORY DTYPE PROBE — SUMMARY")
print(f"{'':=<90}")
print(f"{'#':>2}  {'Pixel Format':<38}  {'Outcome':<16}  {'dtype':<10}  {'comps':>5}  size_check")
print(_SEP)
for r in results:
    _dtype = r["shape_dataType"] or ""
    _comps = str(r["shape_numComps"] or "")
    _size = str(r["size_matches_expected"] or "")
    _out = r["outcome"] or ""
    print(f"{r['row']:>2}  {r['label']:<38}  {_out:<16}  {_dtype:<10}  {_comps:>5}  {_size}")
print(_SEP)

_ok_count = sum(1 for r in results if r["outcome"] == "ok")
_fail_count = len(results) - _ok_count
print(f"  {_ok_count} OK  |  {_fail_count} failed  |  stream_ptr={hex(_stream_ptr)}")
print(f"{'':=<90}")
