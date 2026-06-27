# Spout Bridge COMP Build Guide

Step-by-step instructions for building the `CUDALinkSpoutBridge_v<ver>.tox` component in
TouchDesigner.

> **Why a separate process?** See [ADR-0007](adr/0007-spout-as-launcher-not-transport.md) for the
> full rationale. Short version: TouchDesigner already ships native **Spout In/Out TOPs** for the
> in-TD case. cuda-link-spout's unique value is **headless Python↔Spout on-GPU** (a torch/cupy
> tensor becoming a Spout sender with one CUDA de-swizzle copy, never touching CPU). That value lives
> in the `cuda_link_spout.bridge` sidecar — this COMP makes it visible and droppable.

**⚠️ Important**: `.tox` files are TouchDesigner's binary component format and cannot be generated
from code. This guide provides manual assembly instructions.

---

## What this COMP is — and isn't

| It IS | It is NOT |
|---|---|
| A **launcher/supervisor** — spawns and monitors the `bridge.py` sidecar subprocess | A Spout **transport** — it never imports `cuda_link_spout` in-process |
| A visible TD knob for a previously terminal-only feature | A replacement for native **Spout In/Out TOPs** (use those when TD is both source and sink) |
| Blast-contained — sidecar crash ↔ closed console; Status flips to `Exited (code n)`; TD keeps running | A GPU-work component — it does **no** GPU work and imports no `cuda_link` code |

### Prerequisites

- `cuda_link_spout` installed on the Python interpreter the bridge will run on. Use the Phase 1
  `--spout` installer:
  ```
  python scripts\install_td_library.py --mode 5 --td-python "<interp>" --spout
  ```
  If `Pythonexe` is left blank the COMP auto-resolves via `py -3`; the resolved path is printed to
  the TD Textport on every Start so you can verify the match.
- NVIDIA GPU, CUDA 12.x, and Spout 2.x SDK present (required by the sidecar).
- Windows only (`_spout_bridge.pyd` is a Windows-native extension).

---

## Component structure

Unlike `CUDAIPCLink`, this COMP does **no GPU work** — it is a pure subprocess manager. It needs
**no In TOP, no buffer TOPs, no Script TOPs, and no cuda_link mirror DATs**.

```
CUDALinkSpoutBridge (Base COMP)
├── SpoutBridgeExt     (Text DAT)             ← paste td_exporter/SpoutBridgeExt.py
├── TDHost             (Text DAT)             ← paste td_exporter/TDHost.py
├── spout_bridge_parexec (Parameter Execute DAT) ← paste td_exporter/spout_bridge_parexec.py
└── spout_bridge_exec  (Execute DAT)          ← paste td_exporter/spout_bridge_exec.py
```

| Operator | Type | Source file | Notes |
|---|---|---|---|
| `SpoutBridgeExt` | Text DAT | `td_exporter/SpoutBridgeExt.py` | Extension class; **DAT name must match the class name** so `parent().ext.SpoutBridgeExt` resolves |
| `TDHost` | Text DAT | `td_exporter/TDHost.py` | Provides `RealTDHost` for the extension expression; all TD imports are `TYPE_CHECKING`-only so the file loads outside TD |
| `spout_bridge_parexec` | Parameter Execute DAT | `td_exporter/spout_bridge_parexec.py` | `onValueChange` dispatcher; monitors Active, Restart, Direction, Spoutname, Ipcname, Pythonexe, Debug |
| `spout_bridge_exec` | Execute DAT | `td_exporter/spout_bridge_exec.py` | `onStart`/`onExit` lifecycle; `onFrameStart` crash watch |

---

## Custom parameters

All 8 parameters live on a single custom page named **"Spout Bridge"**. There are **no per-direction
conditionals** — every parameter is relevant in both `out` and `in` mode.

| Name | Label | Type | Default | Notes |
|---|---|---|---|---|
| `Active` | Active | Toggle | OFF | ON = snapshot params + spawn sidecar; OFF = `CTRL_BREAK`→terminate→kill |
| `Restart` | Restart | Pulse | — | Kills + respawns sidecar with current params (picks up config changes) |
| `Direction` | Direction | Menu | `out` | `out` = ipc→Spout sender; `in` = Spout receiver→ipc |
| `Spoutname` | Spout Name | String | `cuda_link` | Spout sender name to create (`out`) or subscribe to (`in`). On `in`, blank = bind to the active sender |
| `Ipcname` | IPC Name | String | `cudalink_ipc_spout` | cuda-link SHM name. **Must match the producing process's `shm_name` exactly** — a mismatch is silent (data goes nowhere) |
| `Pythonexe` | Python Exe | String | _(blank)_ | Full path to the Python interpreter. Blank → auto-resolved via `py -3`; resolved path printed on every Start |
| `Debug` | Debug | Toggle | OFF | Echoes the full resolved argv + lifecycle/crash events to the TD Textport |
| `Status` | Status | String (Read Only) | `Stopped` | Set by the extension: `Stopped` / `Running (PID n)` / `Exited (code n)` / `Changed — press Restart` |

### Authoring notes (Component Editor)

- Page name: **"Spout Bridge"**. Page order: `Active → Restart → Direction → Spoutname → Ipcname → Pythonexe → Debug → Status`.
- `Direction` — type **Menu**, Menu Source **Constant**:
  - Menu Names: `out in`
  - Menu Labels: `out (ipc→Spout)` / `in (Spout→ipc)`
  - Default: `out`
- `Restart` — type **Pulse** (fires `onPulse` in the Parameter Execute DAT).
- `Active`, `Debug` — type **Toggle**.
- `Spoutname`, `Ipcname`, `Pythonexe`, `Status` — type **String**.
- `Status` — enable the **Read Only** flag. The extension writes it via `set_param_value`; the user
  should not edit it directly.
- **No `Device` parameter.** This COMP targets a single-GPU workflow; the sidecar always starts on
  device 0 (the `--device` CLI flag default). Direct CLI users retain full multi-GPU selection via
  `--device`.

---

## Step-by-step assembly

### Step 1: Create Base COMP

1. In TouchDesigner, right-click in the Network Editor.
2. Select **COMP → Base**.
3. Rename to `CUDALinkSpoutBridge`.

### Step 2: Add custom parameters

Right-click the COMP → **Customize Component** to open the Component Editor.

1. Click **+** to add a new page. Name it `Spout Bridge`.
2. Add the 8 parameters from the table above in order. Use the type, default, and Menu settings as
   described in the authoring notes.
3. Close the Component Editor.

### Step 3: Create text DATs

Inside the `CUDALinkSpoutBridge` COMP, create the following DATs in order. Paste each file's
contents exactly as-is.

#### 3a. SpoutBridgeExt Text DAT

1. Create a **Text DAT**, rename to `SpoutBridgeExt`.
2. Paste the entire contents of `td_exporter/SpoutBridgeExt.py`.

This is the extension class. The DAT name (`SpoutBridgeExt`) must match the Python class name so
`parent().ext.SpoutBridgeExt` resolves correctly.

#### 3b. TDHost Text DAT

1. Create a **Text DAT**, rename to `TDHost`.
2. Paste the entire contents of `td_exporter/TDHost.py`.

This module provides `RealTDHost(owner_comp)` — the seam object the extension expression instantiates.
All TD runtime imports inside this file are guarded by `TYPE_CHECKING`, so it loads cleanly outside TD
(unit tests, linting).

#### 3c. spout_bridge_parexec Parameter Execute DAT

1. Create a **Parameter Execute DAT**, rename to `spout_bridge_parexec`.
2. Paste the entire contents of `td_exporter/spout_bridge_parexec.py`.
3. In the Parameter Execute DAT's **Parameters** page, enable monitoring for these parameters:
   `Active`, `Restart`, `Direction`, `Spoutname`, `Ipcname`, `Pythonexe`, `Debug`.

This DAT dispatches `onValueChange` by `par.name` to per-handler functions. Config params (`Direction`,
`Spoutname`, `Ipcname`, `Pythonexe`) are **snapshot-at-spawn** — changing one while the sidecar is
running sets `Status = "Changed — press Restart"` but does NOT respawn automatically.

#### 3d. spout_bridge_exec Execute DAT

1. Create an **Execute DAT**, rename to `spout_bridge_exec`.
2. Paste the entire contents of `td_exporter/spout_bridge_exec.py`.
3. Enable these toggles on the Execute DAT's **Callbacks** page:
   - **Start**: ON — calls `ext.start()` if Active is ON when the project loads
   - **Exit**: ON — calls `ext.stop()` on project close (graceful sidecar shutdown)
   - **Frame Start**: ON — calls `ext.poll_status()` every cook to detect sidecar crashes

### Step 4: Register the extension

1. Select the `CUDALinkSpoutBridge` Base COMP (the parent COMP, not any DAT inside it).
2. Open the **Extensions** parameter page.
3. Set **Extension 1 → Object** to:
   ```
   op('SpoutBridgeExt').module.SpoutBridgeExt(op('TDHost').module.RealTDHost(me))
   ```
4. Enable **Promote** (creates the `parent().ext.SpoutBridgeExt` accessor the DATs use).

> **Why this expression?** `SpoutBridgeExt.__init__(host)` takes the `TDHost` seam object (not the
> COMP directly) so the extension is testable under `FakeTDHost` without a live TD environment.
> Therefore the extension expression must construct `RealTDHost(me)` explicitly and pass it in.
>
> This differs from `CUDAIPCLink`'s Step 4 (`op('CUDAIPCExporter').module.CUDAIPCExtension`) where
> `CUDAIPCExtension.__init__(ownerComp)` takes the COMP and builds its own `RealTDHost` internally.
> Both patterns are correct — they reflect different design choices about where the seam lives.

**Verify**: Open the TD Textport (Alt+T) and type:
```python
op('/project1/CUDALinkSpoutBridge').ext.SpoutBridgeExt
```
Expected: `<SpoutBridgeExt.SpoutBridgeExt object at 0x...>`

### Step 5: Save as .tox

1. Right-click the `CUDALinkSpoutBridge` Base COMP.
2. Select **Save Component .tox...**
3. Save to `TOXES\CUDALinkSpoutBridge_v<ver>.tox` inside the project root.

**Naming convention**: `CUDALinkSpoutBridge_v<ver>.tox` (matches the version you're releasing).
The `TOXES\` subfolder keeps versioned binaries separate from source.

---

## Usage in projects

### Drop the COMP

Drag `CUDALinkSpoutBridge_v<ver>.tox` from Windows Explorer into your TD network, or use
**File → Import Component .tox**. No In TOP wiring needed — this COMP takes no image input.

### Configure parameters

1. **Direction**: `out` (ipc→Spout) or `in` (Spout→ipc).
2. **Ipcname**: set to the cuda-link channel's `shm_name`. **This must match exactly** — the sidecar
   and the cuda-link process identify each other by this string alone. A mismatch is silent; the bridge
   runs but data goes nowhere. The Start printout echoes the IPC name you set.
3. **Spoutname**: the Spout sender name to create (`out`) or receive (`in`). For `in`, leave blank
   to bind to whichever sender is currently active.
4. **Pythonexe**: leave blank unless `py -3` does not resolve to the interpreter where
   `cuda_link_spout` is installed. Set to the full path (e.g.
   `C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe`) when using a venv or a
   non-default system Python.
5. **Active → ON**: spawns the sidecar. A **console window opens** — that window IS the bridge log.
   The TD Textport also receives the Start printout (direction, names, resolved interpreter).

### Snapshot-at-spawn

Parameters are read once at spawn. Editing `Spoutname`, `Ipcname`, `Direction`, or `Pythonexe` while
the sidecar is running sets `Status = "Changed — press Restart"`. Press **Restart** to kill and
respawn with the updated values.

---

## First-launch example / smoke test

This self-contained loopback demo lives in `CUDA_Link_Example.toe` (the numbered save you add the
COMP to). It proves the full path *TD TOP → CUDA-IPC → sidecar → Spout → back into TD* with no
terminal and no external app.

### Network layout

```
source (Noise TOP)
    │
    └──► CUDAIPCLink  ─[CUDA-IPC channel: cudalink_ipc_spout]──► CUDALinkSpoutBridge
                                                                        │
                                                              (Spout sender "cuda_link")
                                                                        │
                                                              spout_in (Spout In TOP) ──► out (Null TOP)
```

| Operator | Type | Key settings |
|---|---|---|
| `source` | Noise TOP (or Movie File In) | RGBA8, any resolution |
| `CUDAIPCLink` | `TOXES/CUDAIPCLink_v1.11.0.tox` | **Mode = Sender**, `Ipcmemname = cudalink_ipc_spout`, `Active = ON`; wire `input` from `source` |
| `CUDALinkSpoutBridge` | NEW `.tox` | `Direction = out`, `Ipcname = cudalink_ipc_spout`, `Spoutname = cuda_link`, `Active = ON` |
| `spout_in` | Spout In TOP (built-in) | `Sender Name = cuda_link` |
| `out` | Null TOP | wired from `spout_in` |

### Smoke-test procedure

1. Open `CUDA_Link_Example.toe`. The `source` TOP shows a pattern.
2. Verify `CUDAIPCLink` `Active = ON` — the COMP writes frames into `cudalink_ipc_spout`.
3. Set `CUDALinkSpoutBridge` `Active = ON`. A **console window opens**. The Start printout shows:
   ```
   [Spout Bridge] Starting bridge...
     Direction : out
     Spout name: 'cuda_link'
     IPC name  : 'cudalink_ipc_spout'
     Python    : C:\...\python.exe
   [Spout Bridge] Sidecar started (PID 12345).
   ```
4. Within one or two seconds, the native `spout_in` TOP lights up with the `source` pattern. The
   bridge auto-derived the frame geometry from the first IPC frame — no Width/Height/Format was
   configured anywhere.
5. Bridge `Status` reads `Running (PID 12345)`.
6. Toggle `Active = OFF`. The console closes; `Status` flips to `Stopped`.

**If `spout_in` stays blank:**
- Open the bridge `spout_bridge_exec` console and look for error lines.
- Verify `Ipcname` matches the `CUDAIPCLink` `Ipcmemname` exactly (same string, case-sensitive).
- Verify `cuda_link_spout` is installed on the resolved interpreter (the Start printout shows which
  interpreter ran; compare with `python -c "import cuda_link_spout; print(cuda_link_spout.__version__)"`
  on the same interpreter).

---

## Troubleshooting

### `Status = "Error: Interpreter not found: 'python'"`

`Pythonexe` is blank and bare `python` is not on PATH (or resolves to TD's embedded interpreter,
which doesn't have `cuda_link_spout`). **Fix**: set `Pythonexe` to the full path of the interpreter
where `cuda_link_spout` is installed.

### `Status = "Exited (code 1)"` immediately after `Active = ON`

One of:
- **Wrong interpreter** — `cuda_link_spout` is not installed there. The sidecar's console (which
  briefly flashes open) shows a `ModuleNotFoundError`. Set `Pythonexe` to the correct interpreter.
- **`cuda_link_spout` not installed at all** — run the Phase 1 `--spout` installer on the target
  interpreter.
- **Native build mismatch** — the `_spout_bridge.pyd` was built against a different CUDA or Python
  version. Rebuild: `utils\build_spout_wheel.cmd`, then reinstall.

### `Status = "Running (PID n)"` but `spout_in` receives nothing

- **Name mismatch** — `Ipcname` does not equal the cuda-link process's `shm_name`. Both must be the
  same byte-identical string. The Start printout echoes the IPC name the bridge was given.
- **Producer not running** — the cuda-link sender (`CUDAIPCLink` COMP or Python `Exporter`) must be
  `Active = ON` before or during the bridge start.
- **BGRA-vs-RGBA order** — the `out` path assumes **RGBA** channel order (auto-derived from dtype
  only; BGRA8 cannot be distinguished from RGBA8 via CUDA-IPC metadata). If your producer emits BGRA,
  run the bridge directly from the CLI with `--fmt BGRA8`. This is a known limitation; a future COMP
  param can expose it.

### `Status = "Changed — press Restart"`

A config parameter was edited while the sidecar was running. Press **Restart** to kill the current
sidecar and respawn with the updated values.

### Extension not found

**Error**: `AttributeError: 'NoneType' object has no attribute 'ext'`

Verify Step 4 was completed correctly. The **Object** field on the Extensions page must be exactly:
```
op('SpoutBridgeExt').module.SpoutBridgeExt(op('TDHost').module.RealTDHost(me))
```
Promote must be **ON**. Both Text DATs (`SpoutBridgeExt` and `TDHost`) must be inside the COMP and
their DAT names must match exactly (case-sensitive).

---

## Reinstalling after code changes

The `cuda_link_spout` Python package (including `bridge.py`) is installed as a wheel — the running
sidecar uses the **installed** copy, not your working tree. After modifying `bridge.py` or any
`spout/` Python file, rebuild and reinstall:

```cmd
utils\build_spout_wheel.cmd
python scripts\install_td_library.py --mode 5 --td-python "<interp>" --spout
```

The core `cuda_link` wheel and the TD-exporter Text DATs (`SpoutBridgeExt.py`, `spout_bridge_*.py`)
are **not** pip-installed — no reinstall is needed for them.

---

## Appendix: File reference

| File | Location | Purpose |
|---|---|---|
| `SpoutBridgeExt.py` | `td_exporter/` | Extension class: subprocess handle, `start`/`stop`/`restart`, `poll_status`, interpreter resolution |
| `TDHost.py` | `td_exporter/` | `RealTDHost` / `FakeTDHost` seam (shared with `CUDAIPCLink`) |
| `spout_bridge_parexec.py` | `td_exporter/` | Parameter Execute DAT: `onValueChange` dispatch + `onPulse` |
| `spout_bridge_exec.py` | `td_exporter/` | Execute DAT: `onStart`, `onExit`, `onFrameStart` crash watch |
| `bridge.py` | `spout/src/cuda_link_spout/` | CLI entry point (`--dir`, `--ipc`, `--spout`, `--fmt`, `--device`) |
| `ADR-0007` | `docs/adr/0007-spout-as-launcher-not-transport.md` | Why a sidecar launcher, not an embedded transport |
| `TOX_BUILD_GUIDE.md` | `docs/` | Equivalent guide for the `CUDAIPCLink` COMP (reference for conventions) |

### bridge.py CLI surface (direct use)

```
python -m cuda_link_spout.bridge --dir out --ipc <shm_name> --spout <sender_name>
python -m cuda_link_spout.bridge --dir in  --ipc <shm_name> --spout <sender_name>

Optional:
  --fmt BGRA8              Override the auto-derived pixel format (out only)
  --width N --height N     Override the auto-derived geometry (out only, for back-compat)
  --device N               CUDA device index (default 0; LUID-matched D3D11 adapter)
```

---

**Build Date**: 2026-06-27
**Component Version**: (fill in when saving .tox)
**TouchDesigner Version**: 2022.x or later
