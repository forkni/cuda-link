# TouchDesigner .tox Build Guide

Step-by-step instructions for building the `CUDAIPCLink_v1.12.2.tox` component in TouchDesigner.

> **Historical release**: `TOXES/CUDAIPCLink_v1.7.2.tox` is available as a GitHub Release asset.

**⚠️ Important**: `.tox` files are TouchDesigner's binary component format and cannot be generated from code. This guide provides manual assembly instructions.

---

## Component Structure

There are two assembly modes. Choose one:

### Library mode (recommended — fewer DATs)

Requires `cuda_link` installed externally (for example, via `install_td_library.cmd`) in a
location TouchDesigner can import. Activate that location either by setting
`CUDALINK_LIB_PATH` before launching TouchDesigner or by adding it to TouchDesigner Preferences
→ Python 32/64 bit Module Path. The bootstrap module loads the package and registers the 15 mirror
names in `sys.modules` — so those mirror Text DATs are not needed.

```text
CUDAIPCExporter (Base COMP)
├── CUDALinkBootstrap (Text DAT)     ← NEW — must be FIRST; copy from td_exporter/CUDALinkBootstrap.py
├── TDHost            (Text DAT)     ← Copy from td_exporter/TDHost.py
├── TDConfig          (Text DAT)     ← Copy from td_exporter/TDConfig.py
├── TDSender          (Text DAT)     ← Copy from td_exporter/TDSender.py
├── TDReceiver        (Text DAT)     ← Copy from td_exporter/TDReceiver.py
├── CUDAIPCExporter   (Text DAT)     ← Copy from td_exporter/CUDAIPCExtension.py  (facade)
├── callbacks         (Execute DAT)  ← Copy from td_exporter/callbacks_template.py
├── parexecute        (Par Execute DAT) ← Copy from td_exporter/parexecute_callbacks.py
├── input             (In TOP)       ← User wires their source TOP here
├── ExportBuffer      (Null TOP)     ← Receives input directly; cudaMemory() reads from here
├── ImportBuffer      (Script TOP)   ← Receiver mode only; set Callbacks DAT → op('script_top_callbacks')
├── warning_emitter   (Script TOP)   ← Status badge; set Callbacks DAT → op('script_top_callbacks')
└── info              (Text DAT)     ← Optional version/author info
```

### Classic / fallback mode (no install required — all mirror DATs included)

All 15 mirror Text DATs must be present. If `cuda_link` is not importable, the bootstrap
no-ops (printing a fallback-mode notice to the Textport) — sibling import resolution works
as before.

```text
CUDAIPCExporter (Base COMP)
├── CUDALinkBootstrap (Text DAT)     ← NEW — must be FIRST; copy from td_exporter/CUDALinkBootstrap.py
├── Env               (Text DAT)     ← Copy from td_exporter/Env.py  (mirror: src/cuda_link/_env.py)
├── FrameProfile      (Text DAT)     ← Copy from td_exporter/FrameProfile.py
├── CUDAIPCWrapper    (Text DAT)     ← Copy from td_exporter/CUDAIPCWrapper.py
├── CUDARuntimeTypes  (Text DAT)     ← Copy from td_exporter/CUDARuntimeTypes.py
├── CUDAGraphs        (Text DAT)     ← Copy from td_exporter/CUDAGraphs.py
├── NVMLObserver      (Text DAT)     ← Copy from td_exporter/NVMLObserver.py
├── SHMProtocol       (Text DAT)     ← Copy from td_exporter/SHMProtocol.py
├── ActivationBarrier (Text DAT)     ← Copy from td_exporter/ActivationBarrier.py
├── Doorbell          (Text DAT)     ← Copy from td_exporter/Doorbell.py
├── NVTXShim          (Text DAT)     ← Copy from td_exporter/NVTXShim.py
├── ExporterPort      (Text DAT)     ← Copy from td_exporter/ExporterPort.py
├── ImporterPort      (Text DAT)     ← Copy from td_exporter/ImporterPort.py
├── CUDAAdapters      (Text DAT)     ← Copy from td_exporter/CUDAAdapters.py
├── Exporter          (Text DAT)     ← Copy from td_exporter/Exporter.py
├── Importer          (Text DAT)     ← Copy from td_exporter/Importer.py
├── TDHost            (Text DAT)     ← Copy from td_exporter/TDHost.py
├── TDConfig          (Text DAT)     ← Copy from td_exporter/TDConfig.py
├── TDSender          (Text DAT)     ← Copy from td_exporter/TDSender.py
├── TDReceiver        (Text DAT)     ← Copy from td_exporter/TDReceiver.py
├── CUDAIPCExporter   (Text DAT)     ← Copy from td_exporter/CUDAIPCExtension.py  (facade)
├── callbacks         (Execute DAT)  ← Copy from td_exporter/callbacks_template.py
├── parexecute        (Par Execute DAT) ← Copy from td_exporter/parexecute_callbacks.py
├── input             (In TOP)       ← User wires their source TOP here
├── ExportBuffer      (Null TOP)     ← Receives input directly; cudaMemory() reads from here
├── ImportBuffer      (Script TOP)   ← Receiver mode only; set Callbacks DAT → op('script_top_callbacks')
├── warning_emitter   (Script TOP)   ← Status badge; set Callbacks DAT → op('script_top_callbacks')
└── info              (Text DAT)     ← Optional version/author info
```

---

## Step-by-Step Assembly

### Step 1: Create Base COMP

1. In TouchDesigner, right-click in the Network Editor
2. Select **COMP → Base**
3. Rename the component to `CUDAIPCExporter`

### Step 2: Add Custom Parameters

Right-click the `CUDAIPCExporter` COMP and select **Customize Component** to open the Component Editor.

#### Create "CUDA IPC" Parameter Page

Click the **+** button to add a new parameter page, name it `"CUDA IPC"`.

#### Add Parameters

| Name | Label | Type | Default | Help Text |
|------|-------|------|---------|-----------|
| `Ipcmemname` | IPC Memory Name | String | `cudalink_ipc_TD>>Python` (Sender) / `cudalink_ipc_Python>>TD` (Receiver) | SharedMemory name for IPC handle transfer. Must match Python's `shm_name`. |
| `Active` | Active | Toggle | `True` (1) | Enable/disable IPC export. When off, export_frame() returns immediately. |
| `Debug` | Debug | Toggle | `False` (0) | Enable verbose performance logging (prints avg metrics every ~97 frames). |
| `Numslots` | Ring Buffer Slots | Int (Menu) | `3` | Number of ring buffer slots for pipelining. Menu: 2, 3, 4 |
| `Mode` | Mode | String (Menu) | `Sender` | Operation mode: Sender exports TD textures to Python; Receiver imports frames from Python back into TD. |

**For `Numslots` menu parameter**:

- Menu Source: **Constant**
- Menu Names: `2 3 4`
- Menu Labels: `2 Slots 3 Slots 4 Slots`

**For `Mode` menu parameter**:

- Menu Source: **Constant**
- Menu Names: `Sender Receiver`
- Menu Labels: `Sender Receiver`

**Appearance tip**: Use Page Order to arrange parameters in a logical flow (Mode → Ipcmemname → Active → Numslots → Debug).

### Step 3: Create Text DATs

**Library mode** (with `cuda_link` installed via `install_td_library.cmd`): create the 6 DATs
in section 3a–3f only — the 15 mirror DATs (3g+) are resolved from the installed package.

**Classic/fallback mode** (no install): create ALL DATs in sections 3a–3i.

Inside the `CUDAIPCExporter` COMP, create Text DATs in the order shown below. Imports between
them resolve automatically because all Text DATs in the same COMP share a module namespace.

**Tip**: If pasting, use **Text Port** mode (Alt+T) for easier editing. Alternatively, set
**File** on the DAT page to the source path and enable **Load on Start**.

#### 3a. CUDALinkBootstrap Text DAT ← FIRST — required in both modes

1. Create a **Text DAT**, rename to `CUDALinkBootstrap`
2. Paste the entire contents of `td_exporter/CUDALinkBootstrap.py`

This DAT **must load before all other Text DATs** (TouchDesigner loads them in the order they
appear in the COMP editor). When set, it injects `CUDALINK_LIB_PATH` onto `sys.path`, then
imports `cuda_link` from the paths already visible to TouchDesigner and registers `sys.modules`
aliases. If `cuda_link` cannot be imported, it no-ops (printing a fallback-mode notice to the
Textport — see below) and fallback mode takes effect automatically.

> **Library mode verify**: After loading, you should see in the Textport:
> `[CUDALinkBootstrap] Library mode active — cuda_link submodules aliased as bare module names.`
> **Fallback mode**: `[CUDALinkBootstrap] Fallback mode — using sibling Text DAT mirrors.`

#### 3b. TDHost Text DAT

1. Create a **Text DAT**, rename to `TDHost`
2. Paste the entire contents of `td_exporter/TDHost.py`

This module provides the `RealTDHost` and `RealTOPHandle` adapters that isolate all `ownerComp.par.*` and `top.cudaMemory()` calls from the engine logic.

#### 3c. TDConfig Text DAT

1. Create a **Text DAT**, rename to `TDConfig`
2. Paste the entire contents of `td_exporter/TDConfig.py`

This module provides the `TDSenderConfig` frozen dataclass that centralises all `CUDALINK_*` environment-variable reads.

#### 3d. TDSender Text DAT

1. Create a **Text DAT**, rename to `TDSender`
2. Paste the entire contents of `td_exporter/TDSender.py`

This module provides `TDSenderEngine` — the Sender-mode engine that owns GPU ring-buffer allocation, IPC handle export, SHM write-back, and CUDA graph capture.

#### 3e. TDReceiver Text DAT

1. Create a **Text DAT**, rename to `TDReceiver`
2. Paste the entire contents of `td_exporter/TDReceiver.py`

This module provides `TDReceiverEngine` — the Receiver-mode engine that owns SHM attachment, IPC handle opening, and Script TOP copyCUDAMemory calls.

#### 3f. CUDAIPCExporter Text DAT

1. Create a **Text DAT**, rename to `CUDAIPCExporter`
2. Paste contents from `td_exporter/CUDAIPCExtension.py`
3. Confirm the import block reads:

   ```python
   with contextlib.suppress(ImportError):
       import CUDALinkBootstrap  # noqa: F401
   ```

   The `contextlib.suppress` guard ensures the extension loads cleanly in classic mode
   (no bootstrap DAT present).

---

**Classic/fallback mode only — add these 15 mirror DATs (Steps 3g–3i):**

#### 3g. Env / FrameProfile / CUDAIPCWrapper / CUDARuntimeTypes / CUDAGraphs Text DATs

For each, create a **Text DAT** with the name shown and paste the matching file:

| DAT name | Source file |
|---|---|
| `Env` | `td_exporter/Env.py` |
| `FrameProfile` | `td_exporter/FrameProfile.py` |
| `CUDAIPCWrapper` | `td_exporter/CUDAIPCWrapper.py` |
| `CUDARuntimeTypes` | `td_exporter/CUDARuntimeTypes.py` |
| `CUDAGraphs` | `td_exporter/CUDAGraphs.py` |

#### 3h. NVMLObserver / SHMProtocol / ActivationBarrier / Doorbell / NVTXShim Text DATs

| DAT name | Source file |
|---|---|
| `NVMLObserver` | `td_exporter/NVMLObserver.py` |
| `SHMProtocol` | `td_exporter/SHMProtocol.py` |
| `ActivationBarrier` | `td_exporter/ActivationBarrier.py` |
| `Doorbell` | `td_exporter/Doorbell.py` |
| `NVTXShim` | `td_exporter/NVTXShim.py` |

#### 3i. ExporterPort / ImporterPort / CUDAAdapters / Exporter / Importer Text DATs

| DAT name | Source file |
|---|---|
| `ExporterPort` | `td_exporter/ExporterPort.py` |
| `ImporterPort` | `td_exporter/ImporterPort.py` |
| `CUDAAdapters` | `td_exporter/CUDAAdapters.py` |
| `Exporter` | `td_exporter/Exporter.py` |
| `Importer` | `td_exporter/Importer.py` |

### Step 4: Register Extension

1. Select the `CUDAIPCExporter` Base COMP (parent component, not the Text DAT inside)
2. Open the **Extensions** parameter page
3. Set **Extension 1**:
   - **Object**: `op('CUDAIPCExporter').module.CUDAIPCExtension`
   - **Promote**: Toggle ON (this creates `me.ext.CUDAIPCExtension` accessor)

**Verification**: Open the **Textport** (Alt+T) and type:

```python
op('/project1/CUDAIPCExporter').ext.CUDAIPCExtension
```

You should see: `<CUDAIPCExporter.CUDAIPCExtension object at 0x...>`

### Step 5: Create Execute DAT Callback

1. Inside the `CUDAIPCExporter` COMP, create an **Execute DAT**, rename to `callbacks`
2. Paste the contents from `td_exporter/callbacks_template.py`
3. Enable the following toggles on the **Execute DAT → Callbacks** page:
   - **Frame Start**: ON
   - **Frame End**: ON (REQUIRED for sender optimization)
   - **On Exit**: ON

**Important**: The `onFrameEnd` callback calls `ext.export_frame()` with no arguments. The extension resolves `ExportBuffer` internally. The data flow through the component is: `input → ExportBuffer → export_frame()`.

### Step 6: Create In TOP

1. Inside the `CUDAIPCExporter` COMP, create an **In TOP**, rename to `input`
2. This is a pass-through input that users will wire their source TOP to
3. Wire output: `input` → `ExportBuffer` (Null TOP that feeds `cudaMemory()`)

**Note**: The In TOP has no parameters to configure - it's purely a connection point.

**Status indicators**: The component gives three visual signals when something is wrong:

| State | Visual | Cause |
|---|---|---|
| **Warning** | COMP node body tints **yellow** + `warning_emitter` yellow badge (inside COMP) | Unsupported pixel format — all float16 variants, 10-bit RGB / 2-bit Alpha, 11-bit float RGB |
| **Error** | COMP node body tints **red** + red `addScriptError` badge + `warning_emitter` badge (inside COMP) | Engine-fatal error (IPC/GPU init failure) |
| **Healthy** | Original node color restored, badges cleared | Condition cleared automatically |

On warnings: change the source TOP's Pixel Format to 8/16-bit fixed or 32-bit float and the COMP recovers within one frame. The COMP body tint is the primary visual; opening the COMP shows the `warning_emitter` Script TOP with the warning message attached as a local badge. Note that TD does not propagate child-operator warnings to the parent COMP boundary tile (H5b — see ARCHITECTURE.md).

### Step 6b: Configure ImportBuffer for TD 2025+ (Optional Optimization)

If using TouchDesigner 2025 or later, enable the `modoutsidecook` toggle on the ImportBuffer Script TOP for improved receiver performance:

1. Select the `ImportBuffer` Script TOP inside the component
2. Open the **Script TOP** parameter page
3. Enable **Modify Outside of Cook** toggle (ON)

**Benefits**:

- Eliminates force-cook overhead (~0.03ms per frame)
- Removes 1-frame resolution change delay
- Simplifies data flow (Execute DAT drives import directly)

**Note**: If `modoutsidecook` is OFF or the parameter doesn't exist (TD 2023), the component automatically falls back to the force-cook path via Script TOP onCook. No code changes needed for backward compatibility.

### Step 6c: Create warning_emitter Script TOP

1. Inside the `CUDAIPCExporter` COMP, create a **Script TOP**, rename to `warning_emitter`.
2. Open the Script TOP parameter page, locate **Callbacks DAT** and set it to
   `op('script_top_callbacks')` — the same DAT already used by `ImportBuffer`.
   The shared `onCook` dispatches by `scriptOp.name`; no extra DAT is needed.
3. Set **Cook Type** → **Off (Pulse to Cook)**.
   The operator cooks only when `RealTDHost` force-cooks it on status transitions — there
   is no need for continuous cooking.
4. Leave it unwired: `warning_emitter` has no inputs and no outputs connected to the
   rendering chain. It exists solely as a status badge host.

**What it does**: `onCook` reads `ownerComp.fetch("cuda_link_status_msg", None)` and calls
`scriptOp.addWarning(msg)` when a status message is present. The badge clears automatically
when the next cook finds no message (after `clear_status` unstores the key and
force-cooks the TOP). The badge is visible inside the COMP alongside the COMP-body tint.

### Step 7: Optional Info DAT

Create a **Text DAT** named `info` with version/author information:

```text
CUDA IPC Exporter v1.12.2
Zero-copy GPU texture export via CUDA IPC

Author: StreamDiffusion Performance Team
Date: 2026-08-11
License: MIT
```

---

## Step 8: Save as .tox

1. Right-click the `CUDAIPCExporter` Base COMP
2. Select **Save Component .tox...**
3. Save to: `TOXES\CUDAIPCLink_v1.12.2.tox` inside the project root

**Naming convention**: Use `CUDAIPCLink_v1.12.2.tox` (matches version) for clarity. The `TOXES\` subfolder keeps versioned binaries separate from source files.

---

## Usage in Projects

### Load the .tox

1. Drag `CUDAIPCLink_v1.12.2.tox` from Windows Explorer into your TD network
2. Or use **File → Import Component .tox**

### Wire a Source TOP

1. Create or select your source TOP (e.g., Movie File In TOP, Render TOP, etc.)
2. Wire it to the `CUDAIPCExporter` COMP's `input` In TOP:
   - Click the source TOP's output connector
   - Drag to the `CUDAIPCExporter` COMP
   - Select `input` from the viewer list

### Configure Parameters

1. **Mode**: Set to `Sender` (exporting TD textures to Python) or `Receiver` (importing Python frames into TD)
2. **Ipcmemname**: Set to a unique name (e.g., `"my_project_ipc"`)
   - This MUST match the `shm_name` in your Python `Importer`/`Exporter` code
3. **Active**: Toggle ON to start exporting/importing
4. **Numslots**: Leave at 3 (optimal for most cases; ignored in Receiver mode)
5. **Debug**: Toggle ON to see performance metrics every ~97 frames

### Verify Operation (Sender Mode)

Open the **Textport** (Alt+T) and look for:

```text
[CUDAIPCExporter] Extension initialized on <CUDAIPCExporter>
[CUDAIPCExporter] Loaded CUDA runtime
[CUDAIPCExporter] Allocated GPU buffer slot 0: 8.0 MB at 0x00007fff12340000
[CUDAIPCExporter] Created 3 IPC buffer slots with events
[CUDAIPCExporter] Created new SharedMemory: my_project_ipc (433 bytes)
[CUDAIPCExporter] Initialization complete - ready for zero-copy GPU transfer
```

### Receiver Mode

When **Mode** = `Receiver`, the component imports GPU frames from a Python `Exporter`:

1. Set **Mode** to `Receiver`
2. Set **Ipcmemname** to match your Python `Exporter`'s `shm_name`
3. Add a **Script TOP** (name it `ImportBuffer`) inside the COMP
4. In the Script TOP's **DAT** field, reference `script_top_callbacks.py`
5. The extension uses `copyCUDAMemory()` to import each frame into the Script TOP

The `callbacks_template.py` `onFrameStart()` handles Receiver mode automatically: it calls `import_frame(ImportBuffer)` to pull the latest frame from Python and write it into the Script TOP. The Script TOP's resolution auto-updates to match the incoming frame size.

If you see errors, check:

- CUDA 12.x is installed
- GPU is NVIDIA with CUDA support
- No other process is using the same `Ipcmemname`

---

## Python Side Setup

Once the TD exporter is running, connect from Python:

```python
from cuda_link import Importer, ImportSpec, ImportOutcome

# Use SAME name as TD's Ipcmemname parameter
importer = Importer.open(
    ImportSpec(
        shm_name="my_project_ipc",  # ← MUST MATCH TD parameter
        shape=(1080, 1920, 4),       # height, width, channels (match your source TOP resolution)
        dtype="float32",             # or "float16", "uint8"
        timeout_ms=5000.0,
    )
)

print("✓ Connected to TouchDesigner CUDA IPC")
result = importer.get_frame()  # returns ImportResult
if result.outcome is ImportOutcome.NEW_FRAME:
    tensor = result.frame  # torch.Tensor on GPU
    print(f"Received frame: {tensor.shape}")
```

---

## Troubleshooting

### Extension not found

**Error**: `AttributeError: 'NoneType' object has no attribute 'ext'`

**Solution**: Verify Step 4 (Register Extension) was completed correctly. The **Object** field must reference the extension class in the Text DAT: `op('CUDAIPCExporter').module.CUDAIPCExtension`

### CUDA runtime DLL not found

**Error**: `[CUDAIPCExporter] Initialization failed: ... cudart64_110.dll not found`

**Solution**: The extension probes full CUDA Toolkit 13.x and 12.x install paths first, then falls back to bare DLL names already loaded in the process, including legacy 11.x names. If TD is installed correctly this error should not occur. Verify your CUDA Toolkit installation or reinstall from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

### SharedMemory already exists

**Error**: `FileExistsError: Cannot create SharedMemory '...' (already exists)`

**Solution**: Another TD instance or Python process is using the same `Ipcmemname`. Either:

1. Use a different name (append a suffix like `"_2"`)
2. Restart TouchDesigner to clean up stale SharedMemory

### Frame export not working

**Symptom**: No error messages, but Python importer receives zero frames or stale data.

**Diagnosis**:

1. Check TD's **Active** parameter is ON
2. Verify the source TOP is actually cooking (check its **Cook** performance monitor)
3. Enable **Debug** in TD and look for `"Frame N: wrote to slot X"` messages every ~97 frames

---

## Advanced: Custom Integration

### Multiple IPC Exporters

You can use multiple `CUDAIPCExporter` components in one project:

```text
/project1
  ├─ MainExporter      (Ipcmemname="main_camera")
  │    └─ input  ←─  Camera TOP
  └─ ControlNetExporter (Ipcmemname="controlnet")
       └─ input  ←─  Edge Detection TOP
```

Python side:

```python
from cuda_link import Importer, ImportSpec, ImportOutcome

main_importer = Importer.open(ImportSpec(shm_name="main_camera"))
cn_importer = Importer.open(ImportSpec(shm_name="controlnet"))

main_result = main_importer.get_frame()
cn_result = cn_importer.get_frame()
if main_result.outcome is ImportOutcome.NEW_FRAME:
    main_frame = main_result.frame
if cn_result.outcome is ImportOutcome.NEW_FRAME:
    cn_frame = cn_result.frame
```

### Dynamic Resolution Handling

The exporter **automatically re-initializes** when the source TOP resolution changes. No manual intervention needed.

**Note**: The Python importer detects the version change and re-opens IPC handles automatically.

---

## Appendix: File Reference

**Core glue (both modes):**

| File | Location | Purpose |
|------|----------|---------|
| `CUDALinkBootstrap.py` | `td_exporter/` | **NEW** — sys.path injector + sys.modules alias registry; must be first DAT |
| `TDHost.py` | `td_exporter/` | `RealTDHost`/`RealTOPHandle` adapters isolating TD runtime access |
| `TDConfig.py` | `td_exporter/` | `TDSenderConfig` frozen dataclass for all `CUDALINK_*` env-var reads |
| `TDSender.py` | `td_exporter/` | `TDSenderEngine` — Sender-mode engine (GPU alloc, IPC export, SHM write) |
| `TDReceiver.py` | `td_exporter/` | `TDReceiverEngine` — Receiver-mode engine (SHM attach, IPC open, copyCUDAMemory) |
| `CUDAIPCExtension.py` | `td_exporter/` | Thin facade (`~300 LOC`) — delegates to `TDSenderEngine` or `TDReceiverEngine` |
| `callbacks_template.py` | `td_exporter/` | Execute DAT callback template |
| `parexecute_callbacks.py` | `td_exporter/` | Parameter Execute DAT callbacks (Active, Mode, Debug, etc.) |
| `script_top_callbacks.py` | `td_exporter/` | Shared Script TOP onCook — ImportBuffer frame import (Receiver mode) **and** warning_emitter status badge (both Script TOPs point their Callbacks DAT here) |
| `benchmark_timestamp.py` | `td_exporter/` | Benchmark helper: SharedMemory timestamp channel |

**Classic/fallback mode only — mirror DATs (auto-generated by `scripts/sync_td_wrapper.py`):**

| File | Location | Canonical source |
|------|----------|-----------------|
| `Env.py` | `td_exporter/` | `src/cuda_link/_env.py` |
| `FrameProfile.py` | `td_exporter/` | `src/cuda_link/_profile.py` |
| `CUDAIPCWrapper.py` | `td_exporter/` | `src/cuda_link/cuda_ipc_wrapper.py` |
| `CUDARuntimeTypes.py` | `td_exporter/` | `src/cuda_link/cuda_runtime_types.py` |
| `CUDAGraphs.py` | `td_exporter/` | `src/cuda_link/cuda_graphs.py` |
| `NVMLObserver.py` | `td_exporter/` | `src/cuda_link/nvml_observer.py` |
| `SHMProtocol.py` | `td_exporter/` | `src/cuda_link/shm_protocol.py` |
| `ActivationBarrier.py` | `td_exporter/` | `src/cuda_link/activation_barrier.py` |
| `NVTXShim.py` | `td_exporter/` | `src/cuda_link/_nvtx.py` |
| `ExporterPort.py` | `td_exporter/` | `src/cuda_link/_exporter_port.py` |
| `ImporterPort.py` | `td_exporter/` | `src/cuda_link/_importer_port.py` |
| `CUDAAdapters.py` | `td_exporter/` | `src/cuda_link/_cuda_adapters.py` |
| `Exporter.py` | `td_exporter/` | `src/cuda_link/exporter.py` |
| `Importer.py` | `td_exporter/` | `src/cuda_link/importer.py` |

**Build output:**

| File | Location | Purpose |
|------|----------|---------|
| `CUDAIPCLink_v1.12.2.tox` | `TOXES/` | Final built .tox component |
| `install_td_library.cmd` | repo root | Library-mode installer launcher (runs `scripts/install_td_library.py`) |
| `scripts/install_td_library.py` | `scripts/` | Multi-target installer — 5 modes: system site-packages, user, conda, TD Preferences, custom |

---

## Next Steps

- Read **[Architecture](ARCHITECTURE.md)** to understand the SharedMemory protocol
- See **[Integration Examples](INTEGRATION_EXAMPLES.md)** for complete workflows
- See **[Benchmarks](BENCHMARKS.md)** for measured performance numbers

---

**Build Date**: 2026-05-29
**Component Version**: 1.12.2
**TouchDesigner Version**: 2022.x or later (2025.x recommended for `modoutsidecook` optimization)
