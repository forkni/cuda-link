"""
probe_addwarning_executedat.py — TD addWarning() operator-type acceptance probe.

PURPOSE:
  Determine which TD operator types actually support addWarning() and whether
  the call produces a visible yellow badge.

  Phase E shipped with cooking_op.addWarning(msg) on the Execute DAT, wrapped
  in contextlib.suppress(Exception).  No yellow badge appeared in TD at runtime.
  This probe determines whether:
    H1 — Execute DAT raises on addWarning (not a Script-family op)
    H2 — Execute DAT silently no-ops (addWarning runs but badge not rendered)
    H3 — parent COMP supports addWarning directly
  ... and which Script-family op types are valid badge hosts.

USAGE (recommended — Textport):
  1. In TD: Dialogs → Textport
  2. Paste and run:
       exec(open('F:/RD_PROJECTS/COMPONENTS/cuda-link/verification/probe_addwarning_executedat.py').read())

OUTPUT:
  - Printed summary in Textport
  - JSON file: verification/results/probe_addwarning_<timestamp>.json

IMPORTANT:
  Before running, update CUDA_LINK_COMP_PATH below to match the actual path of
  the CUDAIPCLink COMP in your current .toe (check the Network editor).
  Default assumes /project1/CUDAIPCLink_to_Touchdesigner.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — adjust to match your .toe network path
# ---------------------------------------------------------------------------
CUDA_LINK_COMP_PATH = "/project1/CUDAIPCLink_to_Touchdesigner"
EXECUTE_DAT_NAME = "execute1"  # name of the Execute DAT inside the COMP

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
try:
    _this_file = Path(__file__).resolve()
    _out_dir = _this_file.parent / "results"
except NameError:
    _out_dir = Path("F:/RD_PROJECTS/COMPONENTS/cuda-link/verification/results")

_out_dir.mkdir(parents=True, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_out_path = _out_dir / f"probe_addwarning_{_ts}.json"

# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

results = {
    "ts": datetime.now().isoformat(),
    "cuda_link_comp_path": CUDA_LINK_COMP_PATH,
    "tests": [],
}


def _probe(label: str, target_op, msg: str = "PROBE: addwarning test") -> dict:
    """Try addWarning on target_op, record the outcome."""
    entry = {
        "label": label,
        "op_type": str(type(target_op).__name__) if target_op is not None else "None",
        "op_family": getattr(target_op, "family", "?") if target_op is not None else "?",
        "has_addWarning_attr": hasattr(target_op, "addWarning") if target_op is not None else False,
        "has_warnings_attr": hasattr(target_op, "warnings") if target_op is not None else False,
    }
    if target_op is None:
        entry["result"] = "SKIP: op is None"
        return entry
    try:
        target_op.addWarning(msg)
        entry["result"] = "OK — no exception"
        if hasattr(target_op, "warnings"):
            try:
                entry["warnings_after"] = str(target_op.warnings())
            except Exception as we:
                entry["warnings_after"] = f"ERROR reading .warnings(): {we}"
        else:
            entry["warnings_after"] = "N/A — no .warnings() method"
    except Exception as e:
        entry["result"] = f"EXCEPTION: {type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc()
    return entry


# ---------------------------------------------------------------------------
# Test 1 — Execute DAT inside the COMP (the Phase E subject)
# ---------------------------------------------------------------------------

comp = op(CUDA_LINK_COMP_PATH)
print(f"\n[probe] COMP found: {comp}")
execute_dat = comp.op(EXECUTE_DAT_NAME) if comp is not None else None
print(f"[probe] Execute DAT found: {execute_dat}")

results["tests"].append(_probe("Execute DAT (onFrameEnd host)", execute_dat))

# ---------------------------------------------------------------------------
# Test 2 — parent COMP itself  (H3 / F2a candidate)
# ---------------------------------------------------------------------------

results["tests"].append(_probe("parent COMP (ownerComp)", comp))

# ---------------------------------------------------------------------------
# Test 3 — fresh Script DAT (control — known-good baseline)
# ---------------------------------------------------------------------------

script_dat = None
try:
    script_dat = op("/project1").create(scriptDAT, "_probe_script_dat_tmp")
    results["tests"].append(_probe("Script DAT (control)", script_dat))
finally:
    if script_dat is not None:
        try:
            script_dat.destroy()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Test 4 — OP Execute DAT (another Execute-type variant) if available
# ---------------------------------------------------------------------------

op_execute = None
try:
    op_execute = op("/project1").create(opexecuteDAT, "_probe_opexecute_tmp")
    results["tests"].append(_probe("OP Execute DAT (control variant)", op_execute))
finally:
    if op_execute is not None:
        try:
            op_execute.destroy()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Extra: addWarning on a Script CHOP (another Script-family candidate)
# ---------------------------------------------------------------------------

script_chop = None
try:
    script_chop = op("/project1").create(scriptCHOP, "_probe_script_chop_tmp")
    results["tests"].append(_probe("Script CHOP", script_chop))
finally:
    if script_chop is not None:
        try:
            script_chop.destroy()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PROBE RESULTS: addWarning operator-type acceptance")
print("=" * 70)
for t in results["tests"]:
    badge_works = t.get("result", "").startswith("OK")
    status = "PASS" if badge_works else "FAIL"
    print(f"  [{status}] {t['label']}")
    print(f"         type={t['op_type']}  family={t['op_family']}")
    print(f"         has_addWarning={t['has_addWarning_attr']}")
    print(f"         result: {t['result']}")
    if "warnings_after" in t:
        print(f"         warnings(): {t['warnings_after']}")
print("=" * 70)
print(f"\n[probe] Full results → {_out_path}")

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------
with open(_out_path, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
