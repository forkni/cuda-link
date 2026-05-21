# ctypes Audit — cuda-link v1.5

**Audited file:** `src/cuda_link/cuda_ipc_wrapper.py`  
**Supporting files:** `src/cuda_link/cuda_runtime_types.py`, `src/cuda_link/cuda_graphs.py`  
**Audit date:** 2026-05-21  
**Auditor:** Claude Code (claude-sonnet-4-6)  
**Reference:** Python 3 ctypes documentation — §Calling functions, §Structures and unions, §Utility functions

---

## 1. CUDA Runtime Function Binding Table

All 41 CUDA runtime functions bound in `_setup_function_signatures()` are enumerated below. Columns:

- **argtypes declared** — whether `argtypes` is explicitly set (prevents ctypes from silently passing wrong types)
- **restype correct** — `c_int` for `cudaError_t`, `c_char_p` for string-returning functions
- **pointer-arg style OK** — `byref(x)` for output scalars/structs; `POINTER(T)` in argtypes for output params; plain value for input handles
- **Notes** — any observations

| # | Function | argtypes declared | restype correct | pointer-arg style OK | Notes |
|---|----------|:-----------------:|:---------------:|:--------------------:|-------|
| 1 | `cudaMalloc` | Yes | Yes (`c_int`) | Yes | `POINTER(c_void_p)` out-param; called with `byref(dev_ptr)` |
| 2 | `cudaFree` | Yes | Yes (`c_int`) | Yes | Input `c_void_p`; value passed directly |
| 3 | `cudaMallocHost` | Yes | Yes (`c_int`) | Yes | `POINTER(c_void_p)` out-param; called with `byref(ptr)` |
| 4 | `cudaFreeHost` | Yes | Yes (`c_int`) | Yes | Input `c_void_p` |
| 5 | `cudaMemcpy` | Yes | Yes (`c_int`) | Yes | `c_void_p, c_void_p, c_size_t, c_int` — complete 4-arg signature |
| 6 | `cudaIpcGetMemHandle` | Yes | Yes (`c_int`) | Yes | `POINTER(cudaIpcMemHandle_t)` out-param; called with `byref(handle)` |
| 7 | `cudaIpcOpenMemHandle` | Yes | Yes (`c_int`) | Yes | `POINTER(c_void_p)` out-param; `cudaIpcMemHandle_t` passed by value (struct copy) — correct for this API |
| 8 | `cudaIpcCloseMemHandle` | Yes | Yes (`c_int`) | Yes | Input `c_void_p` |
| 9 | `cudaIpcGetEventHandle` | Yes | Yes (`c_int`) | Yes | `POINTER(cudaIpcEventHandle_t)` out-param; `CUDAEvent_t` (`c_uint64`) in-param |
| 10 | `cudaIpcOpenEventHandle` | Yes | Yes (`c_int`) | Yes | `POINTER(CUDAEvent_t)` out-param; `cudaIpcEventHandle_t` by value |
| 11 | `cudaEventCreateWithFlags` | Yes | Yes (`c_int`) | Yes | `POINTER(CUDAEvent_t)` out-param |
| 12 | `cudaEventRecord` | Yes | Yes (`c_int`) | Yes | Both args in-values (`CUDAEvent_t`, `CUDAStream_t` are `c_uint64` aliases) |
| 13 | `cudaEventQuery` | Yes | Yes (`c_int`) | Yes | Single in-value |
| 14 | `cudaEventSynchronize` | Yes | Yes (`c_int`) | Yes | Single in-value |
| 15 | `cudaEventDestroy` | Yes | Yes (`c_int`) | Yes | Single in-value |
| 16 | `cudaEventElapsedTime` | Yes | Yes (`c_int`) | Yes | `POINTER(c_float)` out-param; called with `byref(elapsed_ms)` |
| 17 | `cudaDeviceSynchronize` | Yes (`[]`) | Yes (`c_int`) | N/A | Zero-arg function; empty list is correct |
| 18 | `cudaGetLastError` | Yes (`[]`) | Yes (`c_int`) | N/A | Zero-arg function |
| 19 | `cudaPeekAtLastError` | Yes (`[]`) | Yes (`c_int`) | N/A | Zero-arg function |
| 20 | `cudaHostRegister` | Yes | Yes (`c_int`) | Yes | `c_void_p, c_size_t, c_uint`; called with explicit casts |
| 21 | `cudaHostUnregister` | Yes | Yes (`c_int`) | Yes | Single `c_void_p` |
| 22 | `cudaGetErrorString` | Yes | Yes (`c_char_p`) | N/A | In-value `c_int`; `c_char_p` restype correct for static string return |
| 23 | `cudaStreamCreateWithFlags` | Yes | Yes (`c_int`) | Yes | `POINTER(CUDAStream_t)` out-param |
| 24 | `cudaStreamDestroy` | Yes | Yes (`c_int`) | Yes | In-value `CUDAStream_t` |
| 25 | `cudaStreamWaitEvent` | Yes | Yes (`c_int`) | Yes | All in-values |
| 26 | `cudaStreamSynchronize` | Yes | Yes (`c_int`) | Yes | In-value `CUDAStream_t` |
| 27 | `cudaMemcpyAsync` | Yes | Yes (`c_int`) | Yes | Complete 5-arg signature including `CUDAStream_t` |
| 28 | `cudaMemGetInfo` | Yes | Yes (`c_int`) | Yes | Two `POINTER(c_size_t)` out-params; called with `byref()` |
| 29 | `cudaSetDevice` | Yes | Yes (`c_int`) | N/A | In-value `c_int`; called without signature in `__init__` (see Finding F-1) |
| 30 | `cudaGetDevice` | Yes | Yes (`c_int`) | Yes | `POINTER(c_int)` out-param; called with `byref(device)` |
| 31 | `cudaStreamQuery` | Yes | Yes (`c_int`) | Yes | In-value `CUDAStream_t` |
| 32 | `cudaDeviceCanAccessPeer` | Yes | Yes (`c_int`) | Yes | `POINTER(c_int)` out-param + 2 in-value `c_int` |
| 33 | `cudaDeviceGetStreamPriorityRange` | Yes | Yes (`c_int`) | Yes | Two `POINTER(c_int)` out-params |
| 34 | `cudaStreamCreateWithPriority` | Yes | Yes (`c_int`) | Yes | `POINTER(CUDAStream_t)` out-param + `c_uint` + `c_int` |
| 35 | `cudaPointerGetAttributes` | Yes | Yes (`c_int`) | Yes | `POINTER(cudaPointerAttributes)` out-param; `c_void_p` in-param |
| 36 | `cudaHostAlloc` | Yes | Yes (`c_int`) | Yes | `POINTER(c_void_p)` out-param; called with `byref(ptr)` |
| 37 | `cudaDeviceGetAttribute` | Yes | Yes (`c_int`) | Yes | `POINTER(c_int)` out-param + 2 in-value `c_int` |
| 38 | `cudaStreamBeginCapture` | Yes | Yes (`c_int`) | Yes | `CUDAStream_t` + `c_int` in-values |
| 39 | `cudaStreamEndCapture` | Yes | Yes (`c_int`) | Yes | `CUDAStream_t` in-value + `POINTER(CUDAGraph_t)` out-param |
| 40 | `cudaGraphInstantiateWithFlags` | Yes | Yes (`c_int`) | Yes | `POINTER(CUDAGraphExec_t)` out-param + `CUDAGraph_t` + `c_uint64` |
| 41 | `cudaGraphLaunch` | Yes | Yes (`c_int`) | Yes | Two in-values |
| 42 | `cudaGraphDestroy` | Yes | Yes (`c_int`) | Yes | In-value `CUDAGraph_t` |
| 43 | `cudaGraphExecDestroy` | Yes | Yes (`c_int`) | Yes | In-value `CUDAGraphExec_t` |
| 44 | `cudaGraphGetNodes` | Yes | Yes (`c_int`) | Yes | `CUDAGraph_t` in-val + `POINTER(CUDAGraphNode_t)` + `POINTER(c_size_t)`; `None` passed for first query (count-only) |
| 45 | `cudaRuntimeGetVersion` | Yes | Yes (`c_int`) | Yes | `POINTER(c_int)` out-param |
| 46 | `cudaGraphExecMemcpyNodeSetParams` | Yes | Yes (`c_int`) | Yes | Two in-value handles + `POINTER(cudaMemcpy3DParms)`; called with `byref(params)` |
| 47 | `cudaGraphExecMemcpyNodeSetParams1D` | Yes | Yes (`c_int`) | Yes | Complete 6-arg signature; explicit `c_void_p()` wrapping in call site |
| 48 | `cudaGraphExecEventRecordNodeSetEvent` | Yes | Yes (`c_int`) | Yes | Three in-values |
| 49 | `cudaGraphExecEventWaitNodeSetEvent` | Yes | Yes (`c_int`) | Yes | Three in-values |

**Total CUDA runtime functions audited: 49**

---

## 2. Win32 Helper Bindings

### 2.1 `kernel32` — `GetModuleFileNameW`

```python
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
_kernel32.GetModuleFileNameW.restype = ctypes.c_uint32
```

| Check | Status | Notes |
|-------|--------|-------|
| `argtypes` declared | Yes | `[c_void_p, c_wchar_p, c_uint32]` |
| `restype` correct | Yes | `c_uint32` — `GetModuleFileNameW` returns `DWORD` (unsigned 32-bit) |
| `WinDLL` vs `CDLL` | Correct | Win32 functions use `__stdcall`; `WinDLL` is correct on 32-bit; on 64-bit Windows all calls use a single unified calling convention so either works, but `WinDLL` signals intent clearly |
| `use_last_error=True` | Yes | Correct — enables `ctypes.get_last_error()` to retrieve `GetLastError()` value after call |
| `winmode=` flag | Not applicable | `kernel32` is a trusted system DLL; hijack hardening is not required here |

**Assessment:** Fully correct.

### 2.2 `winmm` — Not present

No `winmm` binding exists in the codebase. This is expected — there is no multimedia timer or waveform audio usage.

---

## 3. Findings

### F-1 — `cudaSetDevice` called before `argtypes` are registered (MINOR, correctness risk)

**Location:** `CUDARuntimeAPI.__init__`, line 110:

```python
self.cudart.cudaSetDevice(device)
```

This call occurs immediately after `self._setup_function_signatures()` returns, so `argtypes` for `cudaSetDevice` has already been set by the time this line executes. However, the call order is fragile: `cudaSetDevice` is invoked at the very bottom of `__init__`, and `_setup_function_signatures()` is called just above it on line 105 — so in the current code the signature is set in time.

The real risk is subtler: if `_setup_function_signatures()` were ever split or if `cudaSetDevice` were called during a subclass `__init__` before `_setup_function_signatures()` ran, the argument would be passed without type enforcement. ctypes will default to treating the Python `int` as a C `int` in practice, so this is unlikely to cause a runtime failure today, but it is a maintenance hazard.

**Proposed fix (documentation only — no code change):** Add a comment above the `cudaSetDevice(device)` call confirming that `_setup_function_signatures()` has already registered argtypes, or move the `cudaSetDevice` call to the end of `_setup_function_signatures()` itself.

---

### F-2 — Handle types aliased to `c_uint64` rather than `c_void_p` (INTENTIONAL, well-documented)

```python
CUDAEvent_t = c_uint64
CUDAStream_t = c_uint64
CUDAGraph_t = c_uint64
CUDAGraphExec_t = c_uint64
CUDAGraphNode_t = c_uint64
```

The canonical choice for opaque CUDA handles on POSIX is `c_void_p`. These are instead typed as `c_uint64`. The source file comment attributes this to a PyTorch pull request (pytorch/pytorch#162920) addressing overflow on Windows x64 when ctypes uses `c_void_p` to hold 64-bit pointer values — specifically that `c_void_p` returns `None` when the pointer is 0, which is ambiguous for a null handle versus "not yet initialized."

Using `c_uint64` also means:
- `byref()` on a `c_uint64()` variable works correctly as an output parameter.
- The integer value is directly accessible as `.value` with no `None` ambiguity.
- Passing a `c_uint64` as an in-parameter to a function declared with `CUDAStream_t` in argtypes is type-safe via ctypes coercion.

This is a deliberate and sound choice for Windows x64. It is not a bug.

---

### F-3 — `cudaGetErrorString` restype is `c_char_p` — ownership semantics (INFORMATIONAL)

```python
self.cudart.cudaGetErrorString.restype = ctypes.c_char_p
```

`c_char_p` tells ctypes to auto-convert the returned `const char*` to a Python `bytes` object. The CUDA documentation states that `cudaGetErrorString` returns a pointer to a statically allocated string — the pointer is valid for the lifetime of the process and must not be freed. Using `c_char_p` as restype is correct: ctypes copies the bytes into a Python object and does not attempt to free the pointer.

The call site decodes immediately: `.decode("utf-8")` — also correct. No issue.

---

### F-4 — `cudaGraphGetNodes` called with `None` as nodes array (INTENTIONAL, correct)

```python
result = self.cudart.cudaGraphGetNodes(graph, None, byref(count))
```

argtypes declares the second argument as `POINTER(CUDAGraphNode_t)`. Passing `None` where a `POINTER` type is expected causes ctypes to pass a null pointer (`0x0`) — this is the documented two-pass idiom for `cudaGraphGetNodes`: call once with a null `nodes` pointer to retrieve the count, then call again with an allocated array. This is correct usage.

---

### F-5 — `cudaIpcOpenMemHandle` passes `cudaIpcMemHandle_t` struct by value (CORRECT, verify size)

```python
self.cudart.cudaIpcOpenMemHandle.argtypes = [
    POINTER(c_void_p),
    cudaIpcMemHandle_t,   # passed by value — struct copy
    c_uint,
]
```

The CUDA C API prototype is:

```c
cudaError_t cudaIpcOpenMemHandle(void** devPtr,
                                  cudaIpcMemHandle_t handle,
                                  unsigned int flags);
```

The handle is indeed passed by value in the C API. The struct is 64 bytes (`c_byte * 64`). Passing a 64-byte struct by value via ctypes on Windows x64 is correct — the ABI handles large struct-by-value arguments via hidden pointer when necessary, but ctypes manages this correctly when `argtypes` is declared. The same pattern applies to `cudaIpcEventHandle_t` (also 64 bytes).

**Verification:** `ctypes.sizeof(cudaIpcMemHandle_t)` == 64, matching `CUDA_IPC_HANDLE_SIZE`. No issue.

---

### F-6 — `cudaHostAlloc` default flags in call site vs argtypes (INFORMATIONAL)

```python
# argtypes:
self.cudart.cudaHostAlloc.argtypes = [POINTER(c_void_p), c_size_t, c_uint]

# call site in malloc_host_alloc:
result = self.cudart.cudaHostAlloc(byref(ptr), c_size_t(size), c_uint(flags))
```

The explicit `c_size_t` and `c_uint` casts in the call site are redundant given that `argtypes` is declared (ctypes performs the conversion automatically), but they are harmless and arguably improve readability. This is a style preference, not a bug.

---

### F-7 — `cudaMemcpyAsync` argtypes uses `CUDAStream_t` alias not `c_void_p` (CORRECT)

```python
self.cudart.cudaMemcpyAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_int, CUDAStream_t]
```

`CUDAStream_t` is `c_uint64`. Since stream handles on Windows x64 are 64-bit values (not pointer-width-dependent in the problematic `c_void_p` sense — see F-2), `c_uint64` is the correct type here. Consistent with all other stream/event argtypes declarations.

---

### F-8 — `cudaPointerAttributes` struct layout vs CUDA SDK header (VERIFY)

```python
class cudaPointerAttributes(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("device", c_int),
        ("devicePointer", c_void_p),
        ("hostPointer", c_void_p),
    ]
```

The CUDA SDK (`cuda_runtime_api.h`, CUDA 11+) declares `cudaPointerAttributes` as:

```c
struct cudaPointerAttributes {
    enum cudaMemoryType type;   // int-sized enum
    int device;
    void *devicePointer;
    void *hostPointer;
};
```

The ctypes layout matches: two `c_int` fields (4 bytes each) followed by two `c_void_p` fields (8 bytes each on x64). Total size = 24 bytes. On Windows x64 with default struct alignment the layout is `[int(4), int(4), void*(8), void*(8)]` — no implicit padding between the two `int` fields (they pack to 8 bytes together), then pointers 8-byte aligned. This is correct.

Note: CUDA 10.x used a slightly different `cudaPointerAttributes` that included a `memoryType` field as a legacy alias. In CUDA 11+ this field was removed. The current struct matches CUDA 11+ and CUDA 12.x. No issue for the stated CUDA 11.x–12.x target.

---

### F-9 — No `CFUNCTYPE` / `WINFUNCTYPE` callbacks (EXPECTED)

A grep of both files confirms zero use of `CFUNCTYPE`, `WINFUNCTYPE`, or any ctypes callback type. There is no risk of GIL re-entry via callbacks. As expected.

---

### F-10 — `CDLL` loaded without `use_errno`/`use_last_error` for cudart (INFORMATIONAL)

```python
dll = ctypes.CDLL(name)
dll = ctypes.CDLL(dll_path, winmode=0)
```

`use_errno=True` and `use_last_error=True` are useful for C library functions that set `errno` or the Win32 `GetLastError()` value on failure. CUDA runtime functions do not use `errno` — they return `cudaError_t` as their primary error indicator. Omitting these flags is correct.

`winmode=0` is applied when loading by full path (the fallback path), which disables the DLL search path and is the recommended hijack-hardening mode for untrusted load paths. The name-based loads (the primary path) do not pass `winmode=0`. This is an intentional trade-off documented in the code comments: name-based loads must succeed with the standard Windows DLL search so that an already-loaded DLL instance is reused (to share CUDA context with PyTorch).

---

### F-11 — `cudaMemcpy3DParms` struct layout (VERIFY)

```python
class cudaMemcpy3DParms(ctypes.Structure):
    _fields_ = [
        ("srcArray", c_void_p),   # 8 bytes
        ("srcPos",   cudaPos),    # 3 × size_t = 24 bytes
        ("srcPtr",   cudaPitchedPtr),  # void*(8) + 3×size_t(24) = 32 bytes
        ("dstArray", c_void_p),   # 8 bytes
        ("dstPos",   cudaPos),    # 24 bytes
        ("dstPtr",   cudaPitchedPtr),  # 32 bytes
        ("extent",   cudaExtent), # 3 × size_t = 24 bytes
        ("kind",     c_int),      # 4 bytes
    ]
```

Expected total on x64: 8 + 24 + 32 + 8 + 24 + 32 + 24 + 4 = **156 bytes**, plus up to 4 bytes of trailing padding to align to 8 bytes → **160 bytes**.

The CUDA SDK header `cuda_runtime_api.h` defines the same struct layout. ctypes computes alignment and padding automatically following the platform ABI. The layout matches the SDK struct. No issue.

---

## 4. Verdict

### Overall Correctness: SOUND

All 49 CUDA runtime function bindings are correctly declared with:
- Complete `argtypes` lists (no missing trailing arguments found)
- Correct `restype` on every function (`c_int` for `cudaError_t`, `c_char_p` for `cudaGetErrorString`)
- Appropriate pointer-argument style (`byref()` for output parameters, value passing for opaque handle inputs, `POINTER(T)` in argtypes for all output-pointer positions)
- Correct struct layouts matching CUDA SDK headers for `cudaIpcMemHandle_t`, `cudaIpcEventHandle_t`, `cudaPointerAttributes`, and `cudaMemcpy3DParms`

The Win32 `kernel32` binding is correctly declared with `WinDLL`, `use_last_error=True`, and correct `argtypes`/`restype`.

### Bugs requiring code changes: **None**

No missing argtypes, no wrong restype, no silent-data-corruption risk was found.

### Minor observations (no code change needed)

| ID | Severity | Description |
|----|----------|-------------|
| F-1 | Minor maintenance risk | `cudaSetDevice` called in `__init__` after `_setup_function_signatures()` — correct today but call-order sensitive |
| F-2 | Informational | Handle types use `c_uint64` instead of `c_void_p` — intentional, sound, matches PyTorch convention |
| F-6 | Style | Redundant explicit casts in `cudaHostAlloc` call site — harmless |
| F-8 | Verify | `cudaPointerAttributes` layout targets CUDA 11+; confirm no CUDA 10.x deployment target |
| F-10 | Informational | `winmode=0` applied only on full-path fallback loads — documented intentional trade-off |
| F-11 | Verify | `cudaMemcpy3DParms` layout is correct on x64; confirm with `ctypes.sizeof()` if supporting 32-bit builds |
