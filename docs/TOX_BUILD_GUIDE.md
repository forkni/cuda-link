# TouchDesigner .tox Build Guide

Step-by-step instructions for building the `CUDAIPCLink_v0.7.0.tox` component in TouchDesigner.

**⚠️ Important**: `.tox` files are TouchDesigner's binary component format and cannot be generated from code. This guide provides manual assembly instructions.

---

## Component Structure

```
CUDAIPCExporter (Base COMP)
├── CUDAIPCWrapper    (Text DAT)     ← Copy from td_exporter/CUDAIPCWrapper.py
├── CUDAIPCExporter   (Text DAT)     ← Copy from td_exporter/CUDAIPCExtension.py
├── callbacks         (Execute DAT)  ← Copy from td_exporter/callbacks_template.py
├── parexecute        (Par Execute DAT) ← Copy from td_exporter/parexecute_callbacks.py
├── input             (In TOP)       ← User wires their source TOP here
├── dtype_converter   (Transform TOP) ← Pixel format auto-conversion (see Step 6c)
├── ExportBuffer      (Null TOP)     ← Receives dtype_converter output; cudaMemory() reads from here
├── ImportBuffer      (Script TOP)   ← Receiver mode only; copy from td_exporter/script_top_callbacks.py
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
| `Ipcmemname` | IPC Memory Name | String | `cudalink_output_ipc` | SharedMemory name for IPC handle transfer. Must match Python's `shm_name`. |
| `Active` | Active | Toggle | `True` (1) | Enable/disable IPC export. When off, export_frame() returns immediately. |
| `Debug` | Debug | Toggle | `False` (0) | Enable verbose performance logging (prints avg metrics every 100 frames). |
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

Inside the `CUDAIPCExporter` COMP, create two Text DATs:

#### 3a. CUDAIPCWrapper Text DAT

1. Create a **Text DAT**, rename to `CUDAIPCWrapper`
2. On the **DAT** page:
   - **File** → Browse to `td_exporter/CUDAIPCWrapper.py`
   - Click **Load on Start** toggle (optional, for loading from disk)
3. **OR** paste the entire contents of `td_exporter/CUDAIPCWrapper.py` into the DAT

**Tip**: If pasting, use **Text Port** mode (Alt+T) for easier editing.

#### 3b. CUDAIPCExporter Text DAT

1. Create a **Text DAT**, rename to `CUDAIPCExporter`
2. Paste contents from `td_exporter/CUDAIPCExtension.py`
3. Verify the import line reads: `from CUDAIPCWrapper import get_cuda_runtime`
   - This works because both Text DATs are in the same COMP namespace

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

**Important**: Ensure the Execute DAT references `op('input')` for the source TOP. Users will wire their actual TOP to this In TOP.

### Step 6: Create In TOP

1. Inside the `CUDAIPCExporter` COMP, create an **In TOP**, rename to `input`
2. This is a pass-through input that users will wire their source TOP to

**Note**: The In TOP has no parameters to configure - it's purely a connection point.

### Step 6b: Add dtype_converter Transform TOP (Sender mode)

Inside the `CUDAIPCExporter` COMP, add a **Transform TOP** named `dtype_converter`:

1. Create a **Transform TOP**, rename to `dtype_converter`
2. Set the **Pixel Format** parameter to `"Use Input"` (default — pass-through, zero overhead)
3. Wire input: `input` In TOP → `dtype_converter`
4. Wire output: `dtype_converter` → `ExportBuffer` (Null TOP or the node that feeds `cudaMemory()`)

**Purpose**: TouchDesigner 2025 (CUDA 12.8) rejects `rgba16float` formats from `cudaMemory()`. The extension automatically detects unsupported source formats (float16) and sets `dtype_converter.par.format = "rgba32float"` on the first affected frame — skipping that one frame while the conversion takes effect. For all other formats (uint8, uint16 fixed, float32) the node stays at `"Use Input"` with zero overhead.

**This node is managed automatically** — no manual format changes are needed.

### Step 6c: Configure ImportBuffer for TD 2025+ (Optional Optimization)

If using TouchDesigner 2025 or later, enable the `modoutsidecook` toggle on the ImportBuffer Script TOP for improved receiver performance:

1. Select the `ImportBuffer` Script TOP inside the component
2. Open the **Script TOP** parameter page
3. Enable **Modify Outside of Cook** toggle (ON)

**Benefits**:
- Eliminates force-cook overhead (~0.03ms per frame)
- Removes 1-frame resolution change delay
- Simplifies data flow (Execute DAT drives import directly)

**Note**: If `modoutsidecook` is OFF or the parameter doesn't exist (TD 2023), the component automatically falls back to the force-cook path via Script TOP onCook. No code changes needed for backward compatibility.

### Step 7: Optional Info DAT

Create a **Text DAT** named `info` with version/author information:

```
CUDA IPC Exporter v1.0.0
Zero-copy GPU texture export via CUDA IPC

Author: StreamDiffusion Performance Team
Date: 2026-01-30
License: MIT
```

---

## Step 8: Save as .tox

1. Right-click the `CUDAIPCExporter` Base COMP
2. Select **Save Component .tox...**
3. Save to: `TOXES\CUDAIPCLink_v0.7.0.tox` inside the project root

**Naming convention**: Use `CUDAIPCLink_v0.7.0.tox` (matches version) for clarity. The `TOXES\` subfolder keeps versioned binaries separate from source files.

---

## Usage in Projects

### Load the .tox

1. Drag `CUDAIPCLink_v0.7.0.tox` from Windows Explorer into your TD network
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
   - This MUST match the `shm_name` in your Python `CUDAIPCImporter`/`CUDAIPCExporter` code
3. **Active**: Toggle ON to start exporting/importing
4. **Numslots**: Leave at 3 (optimal for most cases; ignored in Receiver mode)
5. **Debug**: Toggle ON to see performance metrics every 100 frames

### Verify Operation (Sender Mode)

Open the **Textport** (Alt+T) and look for:

```
[CUDAIPCExporter] Extension initialized on <CUDAIPCExporter>
[CUDAIPCExporter] Loaded CUDA runtime
[CUDAIPCExporter] Allocated GPU buffer slot 0: 8.0 MB at 0x00007fff12340000
[CUDAIPCExporter] Created 3 IPC buffer slots with events
[CUDAIPCExporter] Created new SharedMemory: my_project_ipc (433 bytes)
[CUDAIPCExporter] Initialization complete - ready for zero-copy GPU transfer
```

### Receiver Mode

When **Mode** = `Receiver`, the component imports GPU frames from a Python `CUDAIPCExporter`:

1. Set **Mode** to `Receiver`
2. Set **Ipcmemname** to match your Python `CUDAIPCExporter`'s `shm_name`
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
from cuda_link import CUDAIPCImporter

# Use SAME name as TD's Ipcmemname parameter
importer = CUDAIPCImporter(
    shm_name="my_project_ipc",  # ← MUST MATCH TD parameter
    shape=(1080, 1920, 4),       # height, width, channels (match your source TOP resolution)
    dtype="float32",             # or "float16", "uint8"
    debug=True                   # Enable debug logging
)

# Wait for initialization
if importer.is_ready():
    print("✓ Connected to TouchDesigner CUDA IPC")

    # Get frames
    tensor = importer.get_frame()  # torch.Tensor on GPU
    print(f"Received frame: {tensor.shape}")
else:
    print("✗ Connection failed - check SharedMemory name matches")
```

---

## Troubleshooting

### Extension not found

**Error**: `AttributeError: 'NoneType' object has no attribute 'ext'`

**Solution**: Verify Step 4 (Register Extension) was completed correctly. The **Object** field must reference the correct Text DAT: `op('CUDAIPCExporter').module.CUDAIPCExporter`

### CUDA runtime DLL not found

**Error**: `[CUDAIPCExporter] Initialization failed: ... cudart64_110.dll not found`

**Solution**: The extension loads `cudart64_110.dll` (CUDA 11.0, bundled with TouchDesigner) first, then falls back to `cudart64_12.dll`. If TD is installed correctly this error should not occur. If it does, verify your TouchDesigner installation is intact or reinstall CUDA Toolkit 12.x from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

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
3. Enable **Debug** in TD and look for `"Frame N: wrote to slot X"` messages every 100 frames

---

## Advanced: Custom Integration

### Multiple IPC Exporters

You can use multiple `CUDAIPCExporter` components in one project:

```
/project1
  ├─ MainExporter      (Ipcmemname="main_camera")
  │    └─ input  ←─  Camera TOP
  └─ ControlNetExporter (Ipcmemname="controlnet")
       └─ input  ←─  Edge Detection TOP
```

Python side:
```python
main_importer = CUDAIPCImporter(shm_name="main_camera", ...)
cn_importer = CUDAIPCImporter(shm_name="controlnet", ...)

main_frame = main_importer.get_frame()
cn_frame = cn_importer.get_frame()
```

### Dynamic Resolution Handling

The exporter **automatically re-initializes** when the source TOP resolution changes. No manual intervention needed.

**Note**: The Python importer detects the version change and re-opens IPC handles automatically.

---

## Appendix: File Reference

| File | Location | Purpose |
|------|----------|---------|
| `CUDAIPCWrapper.py` | `td_exporter/` | CUDA runtime ctypes wrapper |
| `CUDAIPCExtension.py` | `td_exporter/` | TD extension class (main logic) |
| `callbacks_template.py` | `td_exporter/` | Execute DAT callback template |
| `parexecute_callbacks.py` | `td_exporter/` | Parameter Execute DAT callbacks (Active, Mode, Debug, etc.) |
| `script_top_callbacks.py` | `td_exporter/` | Script TOP onCook callback (Receiver mode ImportBuffer) |
| `benchmark_timestamp.py` | `td_exporter/` | Benchmark helper: SharedMemory timestamp channel |
| `CUDAIPCLink_v0.7.0.tox` | `TOXES/` | Final built .tox component |

---

## Next Steps

- Read **[Architecture](ARCHITECTURE.md)** to understand the SharedMemory protocol
- See **[Integration Examples](INTEGRATION_EXAMPLES.md)** for complete workflows
- Run benchmarks to verify performance on your hardware

---

**Build Date**: 2026-02-09
**Component Version**: 1.0.0
**TouchDesigner Version**: 2022.x or later
