# cuda-link-spout

Optional native add-on that bridges **cuda-link's CUDA-IPC GPU frames** to and from
**Spout** — the Windows GPU texture-sharing standard spoken by Resolume, Unreal,
OBS, Notch, Unity, TouchDesigner, MadMapper, vvvv, and more.

> Full design rationale: [`../docs/competitive/spout-bridge-design.md`](../docs/competitive/spout-bridge-design.md).
> Strategic context (why Spout, not NDI): [`../docs/competitive/plugin-expansion-analysis.md`](../docs/competitive/plugin-expansion-analysis.md).

This package **relaxes cuda-link's pure-Python rule** ([ADR-0006](../docs/adr/0006-stay-pure-python-no-rust.md))
on purpose — it is a *separate*, opt-in component with a native (C++/pybind11)
module. The core `cuda_link` wheel is untouched: if you never install
`cuda-link-spout`, you never pay its dependency cost.

## Why a bridge (and what it costs)

cuda-link shares **linear CUDA memory** via CUDA IPC (CUDA↔CUDA only). Spout shares
**D3D11 textures** via DXGI shared handles. Bridging requires the bridge process to
own a D3D11 device on the CUDA-matched adapter and perform **one device-to-device
"de-swizzle" copy** per frame (linear ↔ CUDA array) — there is no literal zero-copy
from linear memory to a texture. The pixels still never leave the GPU.

cuda-link's real edge is the **GPU→ML-framework** leg (CUDA pointer → zero-copy
torch/cupy tensor), which Spout cannot do without its own GL/DX→CUDA interop copy.
Use **cuda-link on CUDA↔CUDA legs**, **this bridge on CUDA↔graphics-app legs**.

## Architecture

```
SpoutSender / SpoutReceiver        (pure Python, fully unit-tested)
        │  depends only on …
        ▼
SpoutBackend  (Protocol)           ← seam (ADR-0001 port-adapter pattern)
   ├── FakeSpoutBackend            in-memory; drives all CI tests, no GPU
   └── _NativeSpoutBackend         wraps the compiled _spout_bridge module
                                    (CUDA ↔ D3D11 interop + spoutDX), Windows-only
```

Everything except the GPU copy is behind `SpoutBackend`, so the API is testable on
any machine via `FakeSpoutBackend`.

## Usage

```python
from cuda_link_spout import SpoutSender, SpoutSenderSpec, SpoutReceiver, SpoutReceiverSpec, ReceiveOutcome

# Egress: GPU tensor -> Spout (drag into Resolume/UE/OBS as a source)
with SpoutSender.open(SpoutSenderSpec("ai_out", 1024, 1024, "RGBA8")) as tx:
    tx.send(tensor)                # torch/cupy GPU tensor, or a SpoutFrame

# Ingress: Spout sender -> GPU device buffer
with SpoutReceiver.open(SpoutReceiverSpec("resolume_out")) as rx:
    frame = rx.receive()
    if frame.outcome is ReceiveOutcome.NEW_FRAME:
        use(frame.ptr, frame.width, frame.height, frame.fmt)
```

Bridge mode (no native import in your own process — runs as a sidecar):

```bash
python -m cuda_link_spout.bridge --dir out --ipc my_texture_ipc --spout ai_out --width 1024 --height 1024 --fmt RGBA8
python -m cuda_link_spout.bridge --dir in  --spout resolume_out --ipc ai_input_ipc
```

## Build (Windows + CUDA + Spout2)

```bat
git clone https://github.com/leadedge/Spout2 C:\src\Spout2
pip install . --config-settings=cmake.define.SPOUT2_ROOT=C:/src/Spout2
```

Requires the CUDA Toolkit (12.x or 13.x), the Windows SDK (D3D11/DXGI), and a
C++17 compiler. The native module targets sm_75+ (CUDA 13 dropped older arches).

## Development / tests (any OS, no GPU)

The pure-Python layer is tested against `FakeSpoutBackend`:

```bash
cd spout
pytest tests/ -q            # runs without a GPU or the native module
pytest tests/ -q -m "not requires_spout"
```

`spout/tests/conftest.py` puts `spout/src` on `sys.path`, so no install is needed.

## Status & roadmap

| Phase | Item | State |
|---|---|---|
| P0 | Egress (cuda-link → Spout), simple 2-copy path | native source written; **needs Windows compile + smoke test** |
| P1 | Ingress (Spout → cuda-link), simple path | native source written; **needs Windows compile + smoke test** |
| P2 | Bridge-mode sidecar CLI | implemented (`bridge.py`) |
| P3 | Optimized 1-copy (external-memory / keyed-mutex), fused format/flip kernel | designed, not implemented |
| P4 | Packaging matrix, ADR for the native boundary, examples | pyproject/CMake in place; ADR TBD |

The pure-Python API, format mapping, backend seam, fake backend, and CLI
arg/spec handling are implemented and unit-tested now. The native CUDA↔D3D11
copy must be compiled and validated on a Windows CUDA box before release.
