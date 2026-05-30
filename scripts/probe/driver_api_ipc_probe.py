"""
Driver-API IPC probe: validates that cuDevicePrimaryCtxRetain + cuCtxSetCurrent
before cudaMalloc produces IPC handles portable across TD processes.

HYPOTHESIS: If the producer allocates GPU memory while bound to the device's
primary CUDA context (via driver API), a second TD process (also in its primary
context) can open the resulting IPC handle.  This is distinct from what the
shipped runtime-API cudaSetDevice() fix achieves — cudaSetDevice cannot pop
TD's interop context from the driver-API stack.

--- USAGE ---

Step 1 — in the Sender TD textport (where TD's interop context is active):

    exec(open(r"D:\\cuda-link\\scripts\\probe\\driver_api_ipc_probe.py").read())
    probe_producer()

    Expected output:
        [PROBE] cuInit OK
        [PROBE] saved_ctx=0x<td_interop_ctx>, primary_ctx=0x<primary_ctx>
        [PROBE PRODUCER] Switched to primary context
        [PROBE PRODUCER] cudaMalloc OK: 0x<gpu_addr>
        [PROBE PRODUCER] IPC handle prefix (16B): <32 hex chars>
        [PROBE PRODUCER] Handle written to: D:\\cuda-link\\scratch\\probe_handle.bin
        [PROBE PRODUCER] TD context restored
        [PROBE PRODUCER] DONE — now run probe_consumer() in Receiver TD

Step 2 — in the Receiver TD textport (can be same or second TD instance):

    exec(open(r"D:\\cuda-link\\scripts\\probe\\driver_api_ipc_probe.py").read())
    probe_consumer()

    PASS: [PROBE CONSUMER] PROBE OPEN OK: 0x<gpu_addr>
    FAIL: [PROBE CONSUMER] PROBE OPEN FAIL: invalid resource handle (error 400)

--- WHAT EACH OUTCOME MEANS ---

PASS → driver-API primary context switch on the producer side is the fix.
        Proceed with Step 2.6: wire cuDevicePrimaryCtxRetain + cuCtxSetCurrent
        into CUDARuntimeAPI.set_device() / restore_context().

FAIL (error 400 again) → context-binding is NOT the root cause, or the primary
        context identity still doesn't match across processes. Defer TD→TD to
        v1.5.1 (Step 2.7 fallback).

FAIL (other error) → infrastructure issue (nvcuda.dll not found, cuInit failed,
        etc.). Diagnose and retry. If unblockable in 30 min, defer.

--- IMPORTANT ---

Do NOT free the GPU allocation after probe_producer() returns.  The dev_ptr
is kept in _PROBE_STATE["dev_ptr"] so it stays alive for the consumer test.
Call _probe_cleanup() manually when done, or just restart TD.
"""

import ctypes
import os
from ctypes import POINTER, byref, c_int, c_size_t, c_uint, c_void_p

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HANDLE_FILE = r"D:\cuda-link\scratch\probe_handle.bin"
_HANDLE_BYTES = 64  # cudaIpcMemHandle_t is always 64 bytes (CUDA_IPC_HANDLE_SIZE)

# Module-level state dict — survives across exec() calls in the same textport
# by being placed in the calling frame's globals dict.
_PROBE_STATE: dict = {}


# ---------------------------------------------------------------------------
# ctypes struct (minimal, self-contained — no import from cuda_link packages)
# ---------------------------------------------------------------------------


class _IpcMemHandle(ctypes.Structure):
    """Minimal cudaIpcMemHandle_t surrogate (64-byte opaque blob)."""

    _fields_ = [("internal", ctypes.c_byte * _HANDLE_BYTES)]


assert ctypes.sizeof(_IpcMemHandle) == _HANDLE_BYTES


# ---------------------------------------------------------------------------
# DLL loaders
# ---------------------------------------------------------------------------


def _load_driver_api() -> ctypes.CDLL:
    """Load nvcuda.dll (CUDA Driver API) and bind the 5 symbols we need."""
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError as exc:
        raise RuntimeError(
            f"[PROBE] Cannot load nvcuda.dll: {exc}\nEnsure the CUDA driver is installed and nvcuda.dll is on PATH."
        ) from exc

    # cuInit(unsigned int flags) → CUresult
    drv.cuInit.argtypes = [c_uint]
    drv.cuInit.restype = c_int

    # cuDeviceGet(CUdevice* device, int ordinal) → CUresult
    drv.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
    drv.cuDeviceGet.restype = c_int

    # cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) → CUresult
    drv.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
    drv.cuDevicePrimaryCtxRetain.restype = c_int

    # cuCtxGetCurrent(CUcontext* pctx) → CUresult
    drv.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
    drv.cuCtxGetCurrent.restype = c_int

    # cuCtxSetCurrent(CUcontext ctx) → CUresult
    drv.cuCtxSetCurrent.argtypes = [c_void_p]
    drv.cuCtxSetCurrent.restype = c_int

    return drv


def _load_cudart() -> ctypes.CDLL:
    """Load cudart64_*.dll (CUDA Runtime API) and bind the symbols we need."""
    for name in ["cudart64_12.dll", "cudart64_11.dll", "cudart64_110.dll"]:
        try:
            rt = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("[PROBE] Cannot load cudart64_*.dll. Is CUDA installed?")

    # cudaMalloc(void** devPtr, size_t size) → cudaError_t
    rt.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
    rt.cudaMalloc.restype = c_int

    # cudaFree(void* devPtr) → cudaError_t
    rt.cudaFree.argtypes = [c_void_p]
    rt.cudaFree.restype = c_int

    # cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) → cudaError_t
    rt.cudaIpcGetMemHandle.argtypes = [POINTER(_IpcMemHandle), c_void_p]
    rt.cudaIpcGetMemHandle.restype = c_int

    # cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, uint flags)
    # The handle is passed BY VALUE as a struct.
    rt.cudaIpcOpenMemHandle.argtypes = [POINTER(c_void_p), _IpcMemHandle, c_uint]
    rt.cudaIpcOpenMemHandle.restype = c_int

    # cudaIpcCloseMemHandle(void* devPtr) → cudaError_t
    rt.cudaIpcCloseMemHandle.argtypes = [c_void_p]
    rt.cudaIpcCloseMemHandle.restype = c_int

    # cudaGetErrorString(cudaError_t error) → const char*
    rt.cudaGetErrorString.argtypes = [c_int]
    rt.cudaGetErrorString.restype = ctypes.c_char_p

    return rt


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def _switch_to_primary(drv: ctypes.CDLL, device: int = 0) -> int:
    """Switch the calling thread to the device's CUDA primary context.

    Returns the saved context handle value (integer) for later restore.
    The saved value is 0 / None when no explicit context was current.
    """
    r = drv.cuInit(0)
    if r != 0:
        raise RuntimeError(f"[PROBE] cuInit failed: CUresult={r}")
    print("[PROBE] cuInit OK", flush=True)

    cu_dev = c_int()
    r = drv.cuDeviceGet(byref(cu_dev), device)
    if r != 0:
        raise RuntimeError(f"[PROBE] cuDeviceGet({device}) failed: CUresult={r}")

    saved_ctx = c_void_p()
    r = drv.cuCtxGetCurrent(byref(saved_ctx))
    if r != 0:
        raise RuntimeError(f"[PROBE] cuCtxGetCurrent failed: CUresult={r}")

    primary_ctx = c_void_p()
    r = drv.cuDevicePrimaryCtxRetain(byref(primary_ctx), cu_dev)
    if r != 0:
        raise RuntimeError(f"[PROBE] cuDevicePrimaryCtxRetain failed: CUresult={r}")

    saved_val = saved_ctx.value or 0
    primary_val = primary_ctx.value or 0
    print(
        f"[PROBE] saved_ctx={saved_val:#x}, primary_ctx={primary_val:#x}",
        flush=True,
    )

    r = drv.cuCtxSetCurrent(primary_ctx)
    if r != 0:
        raise RuntimeError(f"[PROBE] cuCtxSetCurrent(primary) failed: CUresult={r}")

    return saved_val  # caller must pass this to _restore_context()


def _restore_context(drv: ctypes.CDLL, saved_ctx_value: int) -> None:
    """Restore the previously-saved driver-API context."""
    r = drv.cuCtxSetCurrent(c_void_p(saved_ctx_value if saved_ctx_value else None))
    if r != 0:
        print(f"[PROBE] WARNING: cuCtxSetCurrent(restore) failed: CUresult={r}", flush=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe_producer(device: int = 0, size: int = 2 * 1024 * 1024) -> None:
    """Producer: allocate GPU memory in primary context, mint IPC handle, write to file.

    Run in the Sender TD textport (where TD's interop context is active).
    The allocation is kept alive in _PROBE_STATE["dev_ptr"] — do NOT free it
    until probe_consumer() has successfully opened the handle.
    """
    os.makedirs(os.path.dirname(_HANDLE_FILE), exist_ok=True)

    drv = _load_driver_api()
    rt = _load_cudart()

    saved = _switch_to_primary(drv, device)
    print("[PROBE PRODUCER] Switched to primary context", flush=True)

    dev_ptr = c_void_p()
    r = rt.cudaMalloc(byref(dev_ptr), size)
    if r != 0:
        _restore_context(drv, saved)
        cstr = rt.cudaGetErrorString(r)
        err = cstr.decode("utf-8") if cstr is not None else f"unknown error {r}"
        raise RuntimeError(f"[PROBE] cudaMalloc({size}) failed: {err} (error {r})")
    print(f"[PROBE PRODUCER] cudaMalloc OK: 0x{dev_ptr.value:016x}", flush=True)

    handle = _IpcMemHandle()
    r = rt.cudaIpcGetMemHandle(byref(handle), dev_ptr)
    if r != 0:
        rt.cudaFree(dev_ptr)
        _restore_context(drv, saved)
        cstr = rt.cudaGetErrorString(r)
        err = cstr.decode("utf-8") if cstr is not None else f"unknown error {r}"
        raise RuntimeError(f"[PROBE] cudaIpcGetMemHandle failed: {err} (error {r})")

    raw_bytes = bytes(handle.internal)
    hex_prefix = raw_bytes[:16].hex()
    print(f"[PROBE PRODUCER] IPC handle prefix (16B): {hex_prefix}", flush=True)

    with open(_HANDLE_FILE, "wb") as f:
        f.write(raw_bytes)
    print(f"[PROBE PRODUCER] Handle written to: {_HANDLE_FILE}", flush=True)

    _restore_context(drv, saved)
    print("[PROBE PRODUCER] TD context restored", flush=True)

    # Keep allocation alive so the handle remains valid for the consumer test.
    _PROBE_STATE["dev_ptr"] = dev_ptr
    _PROBE_STATE["rt"] = rt

    print("[PROBE PRODUCER] DONE — now run probe_consumer() in Receiver TD", flush=True)


def probe_consumer(device: int = 0) -> None:
    """Consumer: read handle from file, open it in primary context.

    Run in the Receiver TD textport (can be a second TD process or same TD).

    PASS: prints 'PROBE OPEN OK: 0x...'
    FAIL: prints 'PROBE OPEN FAIL: <error description> (error N)'
    """
    if not os.path.exists(_HANDLE_FILE):
        print(
            f"[PROBE CONSUMER] ERROR: {_HANDLE_FILE} not found. Run probe_producer() first.",
            flush=True,
        )
        return

    with open(_HANDLE_FILE, "rb") as f:
        raw_bytes = f.read(_HANDLE_BYTES)

    if len(raw_bytes) != _HANDLE_BYTES:
        print(
            f"[PROBE CONSUMER] ERROR: expected {_HANDLE_BYTES} bytes, got {len(raw_bytes)}",
            flush=True,
        )
        return

    hex_prefix = raw_bytes[:16].hex()
    print(f"[PROBE CONSUMER] Read handle prefix (16B): {hex_prefix}", flush=True)

    drv = _load_driver_api()
    rt = _load_cudart()

    # Switch receiver to primary context as well — mirrors what the real Receiver
    # engine does (initialized from a fresh TD process with no prior interop work).
    saved = _switch_to_primary(drv, device)
    print("[PROBE CONSUMER] Switched to primary context", flush=True)

    handle = _IpcMemHandle.from_buffer_copy(raw_bytes)
    out_ptr = c_void_p()
    # cudaIpcMemLazyEnablePeerAccess = 1
    r = rt.cudaIpcOpenMemHandle(byref(out_ptr), handle, 1)

    if r == 0:
        print(f"[PROBE CONSUMER] PROBE OPEN OK: 0x{out_ptr.value:016x}", flush=True)
        rt.cudaIpcCloseMemHandle(out_ptr)
    else:
        err_cstr = rt.cudaGetErrorString(r)
        err_str = err_cstr.decode("utf-8") if err_cstr else f"unknown error {r}"
        print(f"[PROBE CONSUMER] PROBE OPEN FAIL: {err_str} (error {r})", flush=True)

    _restore_context(drv, saved)
    print("[PROBE CONSUMER] TD context restored", flush=True)


def _probe_cleanup() -> None:
    """Free the producer allocation.  Call after probe_consumer() finishes, or on cleanup."""
    dev_ptr = _PROBE_STATE.pop("dev_ptr", None)
    rt = _PROBE_STATE.pop("rt", None)
    if dev_ptr is not None and rt is not None:
        rt.cudaFree(dev_ptr)
        print("[PROBE] GPU allocation freed", flush=True)
    else:
        print("[PROBE] Nothing to free", flush=True)
