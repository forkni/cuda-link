"""
probe_cook_context_addwarning.py — Script TOP cook-context addWarning probe (v2).

PURPOSE:
  Phase F found:
    H5  — addWarning() raises "Cannot set warning outside of cook" from onFrameEnd
    H6  — .tox-loaded COMPs are locked; create() is rejected at runtime

  This probe tests the remaining questions:
    H5a — Script TOP onCook IS a valid cook context (addWarning succeeds there)
    H5b — Warning badge propagates from Script TOP child to its parent COMP
           boundary tile in TD's network view

  Both H5a AND H5b must hold for the F2-Script-TOP-cook fix path to be viable.

  The probe runs against a FRESH container COMP at /project1 (always editable)
  rather than the locked cuda-link COMP.  H5b is a generic TD UI behaviour —
  it doesn't depend on which COMP the Script TOP lives in.

  TD auto-provisions a callbacks DAT when a Script TOP is created.  The probe
  only creates the Script TOP; the auto-generated DAT is modified in place.

USAGE (recommended — Textport at root level, NOT inside any COMP):
  1. In TD: Dialogs → Textport
  2. Paste and run:
       exec(open('F:/RD_PROJECTS/COMPONENTS/cuda-link/verification/probe_cook_context_addwarning.py').read())
  3. After the output prints, look at the Network Editor (/project1 level).
     Does the '_probe_cook_ctx_container' COMP tile show a YELLOW badge?
     That answers H5b.
  4. Run the cleanup command printed at the end.
  5. Edit the JSON and set CHECK_BOUNDARY_BADGE to 'yes' or 'no'.

OUTPUT:
  - Printed summary + cleanup command in Textport
  - JSON: verification/results/probe_cook_context_<timestamp>.json
    (CHECK_BOUNDARY_BADGE requires manual entry — see step 5 above)

NOTE:
  ROOT_CONTAINER below must be an editable COMP in your network.
  Default: /project1.  Change if your top-level container has a different path.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_CONTAINER = "/project1"  # editable parent where the test COMP goes
CONTAINER_NAME = "_probe_cook_ctx_container"
SCRIPT_TOP_NAME = "_probe_cook_ctx_top"
PROBE_MSG = "PROBE-H5a: cook context addWarning test"

# Keep False so you can inspect the badge before cleanup.
DESTROY_AFTER_PROBE = False

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
_CANONICAL_OUT = Path("F:/RD_PROJECTS/COMPONENTS/cuda-link/verification/results")
try:
    _this_file = Path(__file__).resolve()
    _candidate = _this_file.parent / "results"
    # When exec'd from textport, __file__ may resolve to F:\project1\...
    # Guard: only use the candidate if it's actually under the cuda-link tree.
    _out_dir = _candidate if "cuda-link" in str(_this_file) else _CANONICAL_OUT
except NameError:
    _out_dir = _CANONICAL_OUT

_out_dir.mkdir(parents=True, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_out_path = _out_dir / f"probe_cook_context_{_ts}.json"

# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------
results = {
    "ts": datetime.now().isoformat(),
    "root_container": ROOT_CONTAINER,
    "probe_msg": PROBE_MSG,
    "CHECK_BOUNDARY_BADGE": (
        "FILL IN MANUALLY: 'yes' or 'no' — does the _probe_cook_ctx_container "
        "COMP tile show a yellow badge in the Network Editor BEFORE you run cleanup?"
    ),
    "tests": [],
}

# ---------------------------------------------------------------------------
# Locate root container
# ---------------------------------------------------------------------------
root = op(ROOT_CONTAINER)
print(f"\n[probe] Root container: {root}")

if root is None:
    results["tests"].append({"label": "SETUP", "result": f"FAIL: op not found at {ROOT_CONTAINER}"})
    with open(_out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Clean up any leftover container from a previous run
# ---------------------------------------------------------------------------
_leftover = root.op(CONTAINER_NAME)
if _leftover is not None:
    try:
        _leftover.destroy()
        print(f"[probe] Cleaned up leftover: {CONTAINER_NAME}")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Test 1 — create container COMP + Script TOP, force-cook, check addWarning
# ---------------------------------------------------------------------------
container = None
script_top = None

test1 = {
    "label": "H5a: Script TOP onCook addWarning",
    "H5a_cook_context_ok": None,
    "callbacks_dat_found": None,
    "script_top_warnings_after": None,
    "container_has_warnings_method": None,
    "container_warnings_after": None,
    "result": None,
}

try:
    # 1a. Create a fresh container COMP
    container = root.create(baseCOMP, CONTAINER_NAME)
    test1["container_created"] = True

    # 1b. Create the Script TOP (TD auto-provisions a callbacks DAT)
    script_top = container.create(scriptTOP, SCRIPT_TOP_NAME)
    test1["script_top_created"] = True

    # 1c. Locate the auto-generated callbacks DAT via par.callbacks
    callbacks = None
    try:
        callbacks = script_top.par.callbacks.eval()
        test1["callbacks_dat_found"] = True if callbacks is not None else False
        test1["callbacks_dat_name"] = callbacks.name if callbacks is not None else None
    except Exception as _de:
        test1["callbacks_dat_found"] = False
        test1["callbacks_dat_error"] = str(_de)

    if callbacks is None:
        test1["result"] = "FAIL — callbacks DAT not found via script_top.par.callbacks.eval()"
        raise RuntimeError("callbacks DAT inaccessible")  # appended once at line ~192

    # 1d. Write the onCook callback
    callbacks.text = f"# Probe onCook callback\ndef onCook(scriptOp):\n    scriptOp.addWarning('{PROBE_MSG}')\n"
    test1["callbacks_text_written"] = True

    # 1e. Force-cook → fires onCook → addWarning
    script_top.cook(force=True)
    test1["cook_fired"] = True
    test1["H5a_cook_context_ok"] = True  # no exception = cook context was valid

    # 1f. Read warnings on the Script TOP
    if hasattr(script_top, "warnings"):
        try:
            test1["script_top_warnings_after"] = list(script_top.warnings())
        except Exception as _we:
            test1["script_top_warnings_after"] = f"ERROR: {_we}"
    else:
        test1["script_top_warnings_after"] = "N/A — Script TOP has no .warnings() method"

    # 1g. Read warnings on the container COMP (H5b programmatic check)
    test1["container_has_warnings_method"] = hasattr(container, "warnings")
    if hasattr(container, "warnings"):
        try:
            test1["container_warnings_after"] = list(container.warnings())
        except Exception as _ce:
            test1["container_warnings_after"] = f"ERROR: {_ce}"
    else:
        test1["container_warnings_after"] = "N/A — COMP has no .warnings() method"

    test1["result"] = "PASS — cook fired; check script_top_warnings_after for H5a, visual badge for H5b"

except RuntimeError:
    pass  # already recorded above
except Exception as e:
    test1["result"] = f"EXCEPTION: {type(e).__name__}: {e}"
    test1["traceback"] = traceback.format_exc()
    if "cook" in str(e).lower() and "outside" in str(e).lower():
        test1["H5a_cook_context_ok"] = False

results["tests"].append(test1)

# ---------------------------------------------------------------------------
# Test 2 — clear the warning (empty onCook), re-check
# ---------------------------------------------------------------------------
test2 = {"label": "Clear: force-cook with empty onCook, check warnings gone", "result": None}

if script_top is not None and test1.get("callbacks_dat_found"):
    try:
        callbacks = script_top.par.callbacks.eval()
        callbacks.text = "def onCook(scriptOp):\n    pass\n"
        script_top.cook(force=True)
        test2["clear_cook_fired"] = True

        if hasattr(script_top, "warnings"):
            try:
                test2["script_top_warnings_after_clear"] = list(script_top.warnings())
            except Exception:
                test2["script_top_warnings_after_clear"] = "ERROR"

        if hasattr(container, "warnings"):
            try:
                test2["container_warnings_after_clear"] = list(container.warnings())
            except Exception:
                test2["container_warnings_after_clear"] = "ERROR"

        # Reinstate the warning so the badge is visible for the visual check
        if not DESTROY_AFTER_PROBE:
            callbacks.text = f"def onCook(scriptOp):\n    scriptOp.addWarning('{PROBE_MSG}')\n"
            script_top.cook(force=True)
            test2["warning_reinstated_for_visual_check"] = True

        test2["result"] = "PASS — clear cook completed"
    except Exception as e:
        test2["result"] = f"EXCEPTION: {type(e).__name__}: {e}"
        test2["traceback"] = traceback.format_exc()
else:
    test2["result"] = "SKIP — Script TOP not created or callbacks DAT inaccessible"

results["tests"].append(test2)

# ---------------------------------------------------------------------------
# Auto-cleanup (only if DESTROY_AFTER_PROBE = True)
# ---------------------------------------------------------------------------
if DESTROY_AFTER_PROBE and container is not None:
    try:
        container.destroy()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Write JSON
# ---------------------------------------------------------------------------
with open(_out_path, "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PROBE RESULTS: Script TOP cook-context addWarning (H5a + H5b)")
print("=" * 70)

for t in results["tests"]:
    r = str(t.get("result", ""))
    status = "PASS" if r.startswith("PASS") else ("SKIP" if r.startswith("SKIP") else "FAIL")
    print(f"  [{status}] {t['label']}")
    print(f"         result:                       {r}")
    if "H5a_cook_context_ok" in t:
        print(f"         H5a cook context ok:          {t['H5a_cook_context_ok']}")
    if "script_top_warnings_after" in t:
        print(f"         script_top.warnings():        {t['script_top_warnings_after']}")
    if "container_warnings_after" in t:
        print(f"         container.warnings():         {t['container_warnings_after']}")
    if "script_top_warnings_after_clear" in t:
        print(f"         top.warnings() after clear:   {t['script_top_warnings_after_clear']}")
    if "container_warnings_after_clear" in t:
        print(f"         container.warnings() cleared: {t['container_warnings_after_clear']}")

print("=" * 70)

if not DESTROY_AFTER_PROBE:
    print()
    print(">>> H5b VISUAL CHECK (do this NOW before running cleanup):")
    print(f">>>   In the Network Editor at /project1, find '{CONTAINER_NAME}'.")
    print(">>>   Does its tile show a YELLOW badge/indicator?")
    print(f">>>   Inside it, '{SCRIPT_TOP_NAME}' should have warning: '{PROBE_MSG}'")
    print()
    print(">>> CLEANUP (paste into Textport when done):")
    print(f">>>   op('{ROOT_CONTAINER}').op('{CONTAINER_NAME}').destroy()")
    print()

print(f"[probe] JSON → {_out_path}")
print("        Edit CHECK_BOUNDARY_BADGE in the JSON with 'yes' or 'no'.")
print()
print(json.dumps(results, indent=2))
