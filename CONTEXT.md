# CUDA-Link Domain Vocabulary

This file defines the canonical names for concepts in this codebase. Use these terms exactly when discussing architecture, writing docs, or naming code.

---

## TD-Side Extension Terms

**TDHost** — the adapter protocol (`TDHost` / `TOPHandle`) that isolates all TouchDesigner runtime calls (`ownerComp.par.*`, `op(...)`, `top.cudaMemory()`, `copyCUDAMemory()`) from the engine logic. `RealTDHost` / `RealTOPHandle` are the production implementations; `FakeTDHost` / `FakeTOPHandle` are the test doubles.

**TOPHandle** — the sub-adapter wrapping a single TouchDesigner TOP operator. Methods: `cuda_memory()`, `pixel_format`, `set_format()`, `copy_cuda_memory()`, `shape`.

**TDSenderEngine** — the Sender-mode engine class (`TDSender.py`). Owns all GPU ring-buffer allocation, IPC handle export, SHM write-back, CUDA graph capture, and activation-barrier signalling. Created by the facade when mode is `Sender`.

**TDReceiverEngine** — the Receiver-mode engine class (`TDReceiver.py`). Owns SHM attachment, IPC handle opening, per-frame GPU event sync, and Script TOP `copyCUDAMemory` calls. Created by the facade when mode is `Receiver`.

**Facade-with-delegation** — the architectural pattern used by `CUDAIPCExtension`. The facade (~300 LOC) exposes the public API unchanged from v1.x and delegates all work to the current engine. Mode switches tear down the old engine and construct a fresh one — guaranteeing zero cross-mode state leak.

**TDSenderConfig** — frozen dataclass (`TDConfig.py`) that centralises all `CUDALINK_*` environment-variable reads. Constructed once at extension init via `TDSenderConfig.from_env()`. Passed to the engine constructor; engines read only `self._config.<field>`.

**textDAT binding** — every `.py` file in `td_exporter/` corresponds to a Text DAT inside the `CUDAIPCExporter` Base COMP. Imports between them resolve within the COMP namespace (e.g., `from TDSender import TDSenderEngine` finds the `TDSender` sibling DAT). PascalCase module names (no `MOD` suffix) match the existing `CUDAIPCWrapper`/`ActivationBarrier` convention.

---

## Protocol Terms

**SHM protocol** — the v0.5.0 binary layout written to `multiprocessing.shared_memory.SharedMemory`. Layout: 20-byte header (magic + version + num\_slots + write\_idx), then N × 128-byte slots (IPC mem handle + IPC event handle), then shutdown flag (1B), metadata (20B), timestamp (8B).

**IPC handle** — a `cudaIpcMemHandle_t` (64 bytes) or `cudaIpcEventHandle_t` (64 bytes) written into a SHM slot. Consumed by the receiver to open a peer GPU buffer or wait on a GPU event without D2H transfer.

**Ring buffer** — the N-slot circular buffer of GPU allocations. `write_idx` (monotonically increasing uint32 in SHM) identifies the slot last written. Readers use `(write_idx - 1) % N`.

**Activation barrier** — a cross-process reference count (`ActivationBarrier.py`) held by the Sender during active export. Receiver waits for it to reach 1 before opening IPC handles, preventing a race where the consumer opens a handle the producer hasn't written yet.

---

## Python-Side Terms

**CUDAIPCImporter** — the Python-side consumer (`src/cuda_link/cuda_ipc_importer.py`). Opens IPC handles written by the TD Sender and returns GPU tensors/arrays via `get_frame()` / `get_frame_numpy()`.

**CUDAIPCExporter** — the Python-side producer (`src/cuda_link/cuda_ipc_exporter.py`). Writes IPC handles to SHM for the TD Receiver to consume via `import_frame()`.
