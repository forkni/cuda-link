"""
CUDA IPC Wrapper for Windows
Based on vLLM cuda_wrapper.py pattern

Provides ctypes interface to CUDA Runtime API for inter-process communication.
Compatible with both TouchDesigner and Python processes.

Requirements:
- CUDA 11.x or 12.x runtime (cudart64_12.dll preferred; cudart64_11.dll / cudart64_110.dll accepted as fallback)
- Windows operating system
- Same GPU visible to both processes
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import POINTER, byref, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p

try:
    from cuda_link._env import env_bool
except (ImportError, ModuleNotFoundError):
    from Env import env_bool  # type: ignore[no-redef]  # noqa: F401  # td_exporter flat namespace

_logger = logging.getLogger(__name__)

if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    _kernel32.GetModuleFileNameW.restype = ctypes.c_uint32
else:
    _kernel32 = None

try:
    # Prefer the bare name so all callers (CUDAIPCWrapper Text DAT and TDReceiver) share
    # the same cudaIpcMemHandle_t / cudaIpcEventHandle_t class objects — critical for ctypes
    # argtypes checking which uses class identity, not structural equivalence.
    # In library mode the bootstrap pre-sets sys.modules["CUDARuntimeTypes"] to
    # cuda_link.cuda_runtime_types before this module is imported, so this succeeds.
    # In classic mode the sibling CUDARuntimeTypes Text DAT is already in sys.modules.
    from CUDARuntimeTypes import (  # type: ignore[no-redef]  # noqa: E402
        HOST_ALLOC_PORTABLE,
        IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
        CUDAError,
        CUDAEvent_t,
        CUDAGraph_t,
        CUDAGraphExec_t,
        CUDAGraphNode_t,
        CUDAStream_t,
        StreamFlags,
        cudaIpcEventHandle_t,
        cudaIpcMemHandle_t,
        cudaMemcpy3DParms,
        cudaPointerAttributes,
    )
except (ImportError, ModuleNotFoundError):
    # Fallback: pure package context where CUDARuntimeTypes is not yet in sys.modules
    # (e.g. imported before the bootstrap runs, or in a standalone test environment).
    from cuda_link.cuda_runtime_types import (  # noqa: E402
        HOST_ALLOC_PORTABLE,
        IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
        CUDAError,
        CUDAEvent_t,
        CUDAGraph_t,
        CUDAGraphExec_t,
        CUDAGraphNode_t,
        CUDAStream_t,
        StreamFlags,
        cudaIpcEventHandle_t,
        cudaIpcMemHandle_t,
        cudaMemcpy3DParms,
        cudaPointerAttributes,
    )

try:
    from cuda_link.cuda_graphs import CUDAGraphsMixin  # noqa: E402
except ImportError:
    from CUDAGraphs import CUDAGraphsMixin  # type: ignore[no-redef]  # noqa: E402


class CUDARuntimeAPI(CUDAGraphsMixin):
    """CUDA Runtime API wrapper using ctypes.

    Provides access to CUDA IPC functions for zero-copy GPU memory
    sharing between processes.

    Usage:
        cuda = CUDARuntimeAPI()

        # Allocate GPU memory
        dev_ptr = cuda.malloc(buffer_size)

        # Export IPC handle (sender process)
        handle = cuda.ipc_get_mem_handle(dev_ptr)

        # Import IPC handle (receiver process)
        imported_ptr = cuda.ipc_open_mem_handle(handle)

        # Use memory...

        # Close handle (receiver)
        cuda.ipc_close_mem_handle(imported_ptr)

        # Free memory (sender)
        cuda.free(dev_ptr)
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize CUDA runtime library.

        Args:
            device: CUDA device index to bind. Defaults to 0.
                    IPC handles are device-scoped; sender and receiver must
                    use the same device or peer-access must be enabled.
        """
        self.device = device
        self.cudart = self._load_cuda_runtime()
        self._setup_function_signatures()
        # Load the driver API (nvcuda.dll) for primary-context save/restore in set_device().
        # Must follow _setup_function_signatures() so cudart argtypes are already wired.
        self._drv: ctypes.CDLL | None = self._load_driver_api()
        # Establish CUDA primary context on the requested device.
        # Must run AFTER _setup_function_signatures() (argtypes needed) but as the
        # very next statement — ensures context exists before any IPC handle operation.
        # Prevents cudaIpcOpenMemHandle error 400 when a second cudart DLL is loaded
        # alongside torch (which has its own bundled cudart). Each DLL instance needs
        # its own context initialized before IPC handle operations can succeed.
        self.cudart.cudaSetDevice(c_int(device))

        if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
            _logger.warning(
                "CUDA_LAUNCH_BLOCKING=1 is set — all CUDA operations are serialized. "
                "This causes ~30x slower frame rates and should only be used for debugging."
            )

        # Default ON; set CUDALINK_STICKY_ERROR_CHECK=0 to skip the cudaPeekAtLastError call.
        self._sticky_check_enabled: bool = env_bool("CUDALINK_STICKY_ERROR_CHECK", default=True)

    def _load_cuda_runtime(self) -> ctypes.CDLL:
        """Load CUDA runtime DLL.

        Returns:
            ctypes.CDLL: Loaded CUDA runtime library

        Raises:
            RuntimeError: If CUDA runtime cannot be loaded
        """
        # Try full CUDA Toolkit paths FIRST: these are deterministic and immune to
        # DLL search-order side-effects (e.g. os.add_dll_directory calls from torch
        # or other venvs added to sys.path via sitecustomize.py).  If the Toolkit is
        # installed, we always prefer its cudart over a bundled copy.
        dll_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudart64_12.dll",
        ]
        last_path_err: OSError | None = None
        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                try:
                    dll = ctypes.CDLL(dll_path, winmode=0)
                    self._log_dll_path(dll, dll_path)
                    return dll
                except OSError as e:
                    _logger.debug("Skipped %s: %s (winerror=%s)", dll_path, e, getattr(e, "winerror", None))
                    last_path_err = e
                    continue

        # Fallback: try by bare name — if cudart is already loaded in this process
        # (e.g., by torch), Windows returns the cached handle, sharing the same runtime
        # instance and CUDA context.  Probed in this order: prefer CUDA 12.x; fall back
        # to 11.x for systems that haven't migrated (TouchDesigner ships cudart64_110.dll).
        # Note: winmode is intentionally omitted here (unlike the absolute-path tier above
        # which uses winmode=0).  For bare-name loads we rely on Windows returning the
        # already-loaded handle from the process DLL cache — not a new DLL search — so
        # the winmode DLL-search-path flags are irrelevant.
        dll_names = ["cudart64_12.dll", "cudart64_11.dll", "cudart64_110.dll"]
        last_name_err: OSError | None = None
        for name in dll_names:
            try:
                dll = ctypes.CDLL(name)
                self._log_dll_path(dll, name)
                return dll
            except OSError as e:
                _logger.debug("Skipped %s: %s (winerror=%s)", name, e, getattr(e, "winerror", None))
                last_name_err = e
                continue

        # Build a diagnostic message that includes the last OS error from each tier.
        # The ctypes reference notes that WinError 126 ("The specified module could not
        # be found") does NOT identify the *missing dependent DLL* — only the entry
        # point that failed to load.  Surface the winerror code so the user can run a
        # dependency tracer (e.g. Dependencies.exe or dumpbin /dependents).
        last_err = last_name_err or last_path_err
        winerror = getattr(last_err, "winerror", None)
        err_detail = f"\nLast OS error: {last_err} (winerror={winerror})" if last_err else ""
        hint_126 = (
            (
                "\nHint: winerror 126 means a *dependent* DLL of cudart was not found, "
                "not cudart itself.  Run 'dumpbin /dependents cudart64_12.dll' or open "
                "the DLL in Dependencies.exe to identify the missing dependency."
            )
            if winerror == 126
            else ""
        )
        raise RuntimeError(
            "Could not load CUDA runtime. Please ensure CUDA 12.x is installed.\n"
            f"Tried paths: {dll_paths}\n"
            f"Tried names: {dll_names}"
            f"{err_detail}{hint_126}"
        )

    @staticmethod
    def _log_dll_path(dll: ctypes.CDLL, hint: str) -> None:
        """Print the resolved filesystem path of a loaded DLL (Windows only).

        Always prints (not gated on debug level) so each TD process textport
        shows which cudart DLL it loaded — critical for diagnosing cross-process
        IPC handle mismatches when two TD instances pick different DLL versions.
        """
        if _kernel32 is None:
            print(f"[CUDAIPC] cudart loaded: {hint} (path resolution unavailable)", flush=True)
            return
        # dll._handle is an undocumented ctypes internal — guard it defensively so a
        # future ctypes change degrades the log line gracefully instead of raising.
        handle = getattr(dll, "_handle", None)
        if not isinstance(handle, int):
            print(f"[CUDAIPC] cudart loaded: {hint} (handle not available)", flush=True)
            return
        try:
            buf = ctypes.create_unicode_buffer(260)
            n = _kernel32.GetModuleFileNameW(ctypes.c_void_p(handle), buf, 260)
            if n == 0:
                # use_last_error=True on the WinDLL means ctypes captured the thread-local
                # error code before restoring it — surface it via WinError for diagnosis.
                err = ctypes.WinError(ctypes.get_last_error())
                print(f"[CUDAIPC] cudart loaded: {hint} (GetModuleFileNameW failed: {err})", flush=True)
            else:
                print(f"[CUDAIPC] cudart loaded: {buf.value}", flush=True)
        except OSError as e:
            print(f"[CUDAIPC] cudart loaded: {hint} (could not resolve path: {e})", flush=True)

    def _load_driver_api(self) -> ctypes.CDLL | None:
        """Load nvcuda.dll (CUDA Driver API) and bind the 5 context-management symbols.

        Returns the loaded DLL, or None if unavailable (Linux, driver not installed).
        Failure is non-fatal: set_device() falls back to the runtime-API path.
        """
        try:
            drv = ctypes.CDLL("nvcuda.dll")
        except OSError:
            _logger.debug("nvcuda.dll unavailable; driver-API context switch will not be used")
            return None

        drv.cuInit.argtypes = [c_uint]
        drv.cuInit.restype = c_int
        drv.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
        drv.cuDeviceGet.restype = c_int
        drv.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
        drv.cuDevicePrimaryCtxRetain.restype = c_int
        drv.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        drv.cuCtxGetCurrent.restype = c_int
        drv.cuCtxSetCurrent.argtypes = [c_void_p]
        drv.cuCtxSetCurrent.restype = c_int

        r = drv.cuInit(0)
        if r != 0:
            _logger.warning("cuInit returned %d; driver-API context switch will not be used", r)
            return None

        return drv

    def _setup_function_signatures(self) -> None:
        """Define function signatures for CUDA runtime functions."""
        # cudaMalloc(void** devPtr, size_t size)
        self.cudart.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        self.cudart.cudaMalloc.restype = c_int

        # cudaFree(void* devPtr)
        self.cudart.cudaFree.argtypes = [c_void_p]
        self.cudart.cudaFree.restype = c_int

        # cudaMallocHost(void** ptr, size_t size) — allocate pinned (page-locked) host memory
        self.cudart.cudaMallocHost.argtypes = [POINTER(c_void_p), c_size_t]
        self.cudart.cudaMallocHost.restype = c_int

        # cudaFreeHost(void* ptr) — free pinned host memory
        self.cudart.cudaFreeHost.argtypes = [c_void_p]
        self.cudart.cudaFreeHost.restype = c_int

        # cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
        self.cudart.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
        self.cudart.cudaMemcpy.restype = c_int

        # cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)
        self.cudart.cudaIpcGetMemHandle.argtypes = [
            POINTER(cudaIpcMemHandle_t),
            c_void_p,
        ]
        self.cudart.cudaIpcGetMemHandle.restype = c_int

        # cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
        self.cudart.cudaIpcOpenMemHandle.argtypes = [
            POINTER(c_void_p),
            cudaIpcMemHandle_t,
            c_uint,
        ]
        self.cudart.cudaIpcOpenMemHandle.restype = c_int

        # cudaIpcCloseMemHandle(void* devPtr)
        self.cudart.cudaIpcCloseMemHandle.argtypes = [c_void_p]
        self.cudart.cudaIpcCloseMemHandle.restype = c_int

        # cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event)
        self.cudart.cudaIpcGetEventHandle.argtypes = [
            POINTER(cudaIpcEventHandle_t),
            CUDAEvent_t,
        ]
        self.cudart.cudaIpcGetEventHandle.restype = c_int

        # cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle)
        self.cudart.cudaIpcOpenEventHandle.argtypes = [
            POINTER(CUDAEvent_t),
            cudaIpcEventHandle_t,
        ]
        self.cudart.cudaIpcOpenEventHandle.restype = c_int

        # cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
        self.cudart.cudaEventCreateWithFlags.argtypes = [POINTER(CUDAEvent_t), c_uint]
        self.cudart.cudaEventCreateWithFlags.restype = c_int

        # cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
        self.cudart.cudaEventRecord.argtypes = [CUDAEvent_t, CUDAStream_t]
        self.cudart.cudaEventRecord.restype = c_int

        # cudaEventQuery(cudaEvent_t event)
        self.cudart.cudaEventQuery.argtypes = [CUDAEvent_t]
        self.cudart.cudaEventQuery.restype = c_int

        # cudaEventSynchronize(cudaEvent_t event)
        self.cudart.cudaEventSynchronize.argtypes = [CUDAEvent_t]
        self.cudart.cudaEventSynchronize.restype = c_int

        # cudaEventDestroy(cudaEvent_t event)
        self.cudart.cudaEventDestroy.argtypes = [CUDAEvent_t]
        self.cudart.cudaEventDestroy.restype = c_int

        # cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
        self.cudart.cudaEventElapsedTime.argtypes = [POINTER(c_float), CUDAEvent_t, CUDAEvent_t]
        self.cudart.cudaEventElapsedTime.restype = c_int

        # cudaDeviceSynchronize()
        self.cudart.cudaDeviceSynchronize.argtypes = []
        self.cudart.cudaDeviceSynchronize.restype = c_int

        # cudaPeekAtLastError() — non-destructive sticky-error read (does NOT clear the error)
        # cudaGetLastError() is intentionally NOT bound: it destructively clears the sticky
        # error, making it unsafe to call in poll paths. Use cudaPeekAtLastError exclusively.
        self.cudart.cudaPeekAtLastError.argtypes = []
        self.cudart.cudaPeekAtLastError.restype = c_int

        # cudaHostRegister(void* ptr, size_t size, unsigned int flags) — page-lock existing host memory
        self.cudart.cudaHostRegister.argtypes = [c_void_p, c_size_t, c_uint]
        self.cudart.cudaHostRegister.restype = c_int

        # cudaHostUnregister(void* ptr) — unregister page-locked host memory
        self.cudart.cudaHostUnregister.argtypes = [c_void_p]
        self.cudart.cudaHostUnregister.restype = c_int

        # cudaGetErrorString(cudaError_t error)
        self.cudart.cudaGetErrorString.argtypes = [c_int]
        self.cudart.cudaGetErrorString.restype = ctypes.c_char_p

        # cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
        self.cudart.cudaStreamCreateWithFlags.argtypes = [POINTER(CUDAStream_t), c_uint]
        self.cudart.cudaStreamCreateWithFlags.restype = c_int

        # cudaStreamDestroy(cudaStream_t stream)
        self.cudart.cudaStreamDestroy.argtypes = [CUDAStream_t]
        self.cudart.cudaStreamDestroy.restype = c_int

        # cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
        self.cudart.cudaStreamWaitEvent.argtypes = [CUDAStream_t, CUDAEvent_t, c_uint]
        self.cudart.cudaStreamWaitEvent.restype = c_int

        # cudaStreamSynchronize(cudaStream_t stream)
        self.cudart.cudaStreamSynchronize.argtypes = [CUDAStream_t]
        self.cudart.cudaStreamSynchronize.restype = c_int

        # cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
        self.cudart.cudaMemcpyAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_int, CUDAStream_t]
        self.cudart.cudaMemcpyAsync.restype = c_int

        # cudaMemGetInfo(size_t* free, size_t* total)
        self.cudart.cudaMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]
        self.cudart.cudaMemGetInfo.restype = c_int

        # cudaSetDevice(int device)
        self.cudart.cudaSetDevice.argtypes = [c_int]
        self.cudart.cudaSetDevice.restype = c_int

        # cudaGetDevice(int* device)
        self.cudart.cudaGetDevice.argtypes = [POINTER(c_int)]
        self.cudart.cudaGetDevice.restype = c_int

        # cudaStreamQuery(cudaStream_t stream)
        self.cudart.cudaStreamQuery.argtypes = [CUDAStream_t]
        self.cudart.cudaStreamQuery.restype = c_int

        # cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)
        self.cudart.cudaDeviceCanAccessPeer.argtypes = [POINTER(c_int), c_int, c_int]
        self.cudart.cudaDeviceCanAccessPeer.restype = c_int

        # cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)
        self.cudart.cudaDeviceGetStreamPriorityRange.argtypes = [POINTER(c_int), POINTER(c_int)]
        self.cudart.cudaDeviceGetStreamPriorityRange.restype = c_int

        # cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority)
        self.cudart.cudaStreamCreateWithPriority.argtypes = [POINTER(CUDAStream_t), c_uint, c_int]
        self.cudart.cudaStreamCreateWithPriority.restype = c_int

        # cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr)
        self.cudart.cudaPointerGetAttributes.argtypes = [POINTER(cudaPointerAttributes), c_void_p]
        self.cudart.cudaPointerGetAttributes.restype = c_int

        # === G1: non-graph helpers (re-enabled Phase 1.1) ===
        # cudaHostAlloc(void** ptr, size_t size, unsigned int flags)
        # Replaces cudaMallocHost with explicit flag control.
        # cudaHostAllocPortable  = 0x01 — accessible from any CUDA context in process
        # cudaHostAllocMapped    = 0x02 — map into device address space
        # cudaHostAllocWriteCombined = 0x04 — write-combined (fast CPU writes, slow CPU reads)
        self.cudart.cudaHostAlloc.argtypes = [POINTER(c_void_p), c_size_t, c_uint]
        self.cudart.cudaHostAlloc.restype = c_int

        # cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
        # Used to query cudaDevAttrAsyncEngineCount (attr=4) — how many DMA copy engines exist.
        self.cudart.cudaDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
        self.cudart.cudaDeviceGetAttribute.restype = c_int

        # === G2: graph lifecycle (re-enabled Phase 1.2) ===
        # CUDA 10.0+ graph capture/build/launch/teardown + runtime-version gate.

        # cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
        # mode: 0=global, 1=thread_local, 2=relaxed
        self.cudart.cudaStreamBeginCapture.argtypes = [CUDAStream_t, c_int]
        self.cudart.cudaStreamBeginCapture.restype = c_int

        # cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
        self.cudart.cudaStreamEndCapture.argtypes = [CUDAStream_t, POINTER(CUDAGraph_t)]
        self.cudart.cudaStreamEndCapture.restype = c_int

        # cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
        #                               unsigned long long flags)   [CUDA 11.4+ stable 3-arg form]
        # Prefer this over cudaGraphInstantiate on any cudart 11.x: the latter changed
        # from 5-arg (CUDA 10.0–11.8) to 3-arg (CUDA 12.0+) — calling the 12.0 3-arg
        # binding against an 11.x DLL mismatches the ABI and crashes (WDDM access
        # violation). cudaGraphInstantiateWithFlags has had a stable 3-arg signature
        # since 11.4 and is available in all 12.x releases as well.
        self.cudart.cudaGraphInstantiateWithFlags.argtypes = [POINTER(CUDAGraphExec_t), CUDAGraph_t, c_uint64]
        self.cudart.cudaGraphInstantiateWithFlags.restype = c_int

        # cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)
        self.cudart.cudaGraphLaunch.argtypes = [CUDAGraphExec_t, CUDAStream_t]
        self.cudart.cudaGraphLaunch.restype = c_int

        # cudaGraphDestroy(cudaGraph_t graph)
        self.cudart.cudaGraphDestroy.argtypes = [CUDAGraph_t]
        self.cudart.cudaGraphDestroy.restype = c_int

        # cudaGraphExecDestroy(cudaGraphExec_t graphExec)
        self.cudart.cudaGraphExecDestroy.argtypes = [CUDAGraphExec_t]
        self.cudart.cudaGraphExecDestroy.restype = c_int

        # cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes)
        # Pass nodes=NULL to query count; then call again with allocated array.
        self.cudart.cudaGraphGetNodes.argtypes = [CUDAGraph_t, POINTER(CUDAGraphNode_t), POINTER(c_size_t)]
        self.cudart.cudaGraphGetNodes.restype = c_int

        # cudaRuntimeGetVersion(int* runtimeVersion)
        # Returns the version as int (e.g., 11040 = CUDA 11.4, 12080 = CUDA 12.8).
        # Used to gate optional API calls (e.g., cudaGraphExecMemcpyNodeSetParams1D
        # requires 11.3+) when the loaded cudart DLL may be a 11.0.x patch.
        self.cudart.cudaRuntimeGetVersion.argtypes = [POINTER(c_int)]
        self.cudart.cudaRuntimeGetVersion.restype = c_int

        # === G3: graph node setters (re-enabled Phase 1.3) ===
        # Per-frame in-place node update for ring-slot remap. Most CUDA-12-flavoured
        # of the 14 (NodeSetParams1D 11.3+; event-node setters 11.4+).

        # cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t, cudaGraphNode_t,
        #                                  const cudaMemcpy3DParms*)
        # Updates a 3D-captured memcpy node. For nodes captured from cudaMemcpyAsync
        # (1D form) use cudaGraphExecMemcpyNodeSetParams1D instead.
        self.cudart.cudaGraphExecMemcpyNodeSetParams.argtypes = [
            CUDAGraphExec_t,
            CUDAGraphNode_t,
            POINTER(cudaMemcpy3DParms),
        ]
        self.cudart.cudaGraphExecMemcpyNodeSetParams.restype = c_int

        # cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t, cudaGraphNode_t,
        #                                    void* dst, const void* src,
        #                                    size_t count, cudaMemcpyKind kind)
        # Updates a 1D memcpy node (captured from cudaMemcpyAsync). CUDA 11.3+.
        self.cudart.cudaGraphExecMemcpyNodeSetParams1D.argtypes = [
            CUDAGraphExec_t,
            CUDAGraphNode_t,
            c_void_p,
            c_void_p,
            c_size_t,
            c_int,
        ]
        self.cudart.cudaGraphExecMemcpyNodeSetParams1D.restype = c_int

        # cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t, cudaGraphNode_t,
        #                                      cudaEvent_t event)
        # Updates the event recorded by an event-record node. CUDA 11.4+.
        self.cudart.cudaGraphExecEventRecordNodeSetEvent.argtypes = [CUDAGraphExec_t, CUDAGraphNode_t, CUDAEvent_t]
        self.cudart.cudaGraphExecEventRecordNodeSetEvent.restype = c_int

        # cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t, cudaGraphNode_t,
        #                                    cudaEvent_t event)
        # Updates the event waited on by an event-wait node. CUDA 11.4+.
        self.cudart.cudaGraphExecEventWaitNodeSetEvent.argtypes = [CUDAGraphExec_t, CUDAGraphNode_t, CUDAEvent_t]
        self.cudart.cudaGraphExecEventWaitNodeSetEvent.restype = c_int
        self._install_errcheck()

    def _install_errcheck(self) -> None:
        """Install ctypes errcheck on all cudart functions that always treat non-zero as fatal.

        Exempted: cudaGetErrorString (c_char_p restype), cudaEventQuery/cudaStreamQuery
        (poll sentinels that allow cudaErrorNotReady), cudaGetLastError/cudaPeekAtLastError
        (read the sticky-error value, raising would defeat their purpose).
        """
        cudart = self.cudart

        def _strict_errcheck(result, func, _args):
            if result != 0:
                cstr = cudart.cudaGetErrorString(result)
                error_str = cstr.decode("utf-8") if cstr is not None else f"unknown error {result}"
                raise RuntimeError(f"CUDA {func.__name__} failed: {error_str} (code {result})")
            return result

        _strict_funcs = (
            "cudaMalloc",
            "cudaFree",
            "cudaMallocHost",
            "cudaFreeHost",
            "cudaMemcpy",
            "cudaIpcGetMemHandle",
            "cudaIpcOpenMemHandle",
            "cudaIpcCloseMemHandle",
            "cudaIpcGetEventHandle",
            "cudaIpcOpenEventHandle",
            "cudaEventCreateWithFlags",
            "cudaEventRecord",
            "cudaEventSynchronize",
            "cudaEventDestroy",
            "cudaEventElapsedTime",
            "cudaDeviceSynchronize",
            "cudaHostRegister",
            "cudaHostUnregister",
            "cudaStreamCreateWithFlags",
            "cudaStreamDestroy",
            "cudaStreamWaitEvent",
            "cudaStreamSynchronize",
            "cudaMemcpyAsync",
            "cudaMemGetInfo",
            "cudaSetDevice",
            "cudaGetDevice",
            "cudaDeviceCanAccessPeer",
            "cudaDeviceGetStreamPriorityRange",
            "cudaStreamCreateWithPriority",
            "cudaPointerGetAttributes",
            "cudaHostAlloc",
            "cudaDeviceGetAttribute",
            "cudaStreamBeginCapture",
            "cudaStreamEndCapture",
            "cudaGraphInstantiateWithFlags",
            "cudaGraphLaunch",
            "cudaGraphDestroy",
            "cudaGraphExecDestroy",
            "cudaGraphGetNodes",
            "cudaRuntimeGetVersion",
            "cudaGraphExecMemcpyNodeSetParams",
            "cudaGraphExecMemcpyNodeSetParams1D",
            "cudaGraphExecEventRecordNodeSetEvent",
            "cudaGraphExecEventWaitNodeSetEvent",
        )
        for fname in _strict_funcs:
            getattr(cudart, fname).errcheck = _strict_errcheck

    def check_error(self, result: int, operation: str) -> None:
        """Check CUDA error code and raise exception if failed.

        Args:
            result: CUDA error code
            operation: Description of the operation that failed

        Raises:
            RuntimeError: If result indicates an error
        """
        if result != CUDAError.SUCCESS:
            cstr = self.cudart.cudaGetErrorString(result)
            error_str = cstr.decode("utf-8") if cstr is not None else f"unknown error {result}"
            error_name = CUDAError.get_name(result)
            raise RuntimeError(f"CUDA {operation} failed: {error_str} (error {result}: {error_name})")

    def peek_last_error(self) -> int:
        """Non-destructively read the thread-local sticky CUDA error.

        Returns SUCCESS (0) normally. A non-zero value means a prior async
        operation (memcpy, kernel) failed and the error was not yet consumed.
        Unlike cudaGetLastError this does NOT clear the latched error state.
        """
        return int(self.cudart.cudaPeekAtLastError())

    def check_sticky_error(self, context: str) -> None:
        """Warn and raise if a sticky CUDA error is latched from a prior async op.

        No-op when CUDALINK_STICKY_ERROR_CHECK=0. Enabled by default.
        Use peek_last_error() directly for the raw value without raising.
        """
        if not self._sticky_check_enabled:
            return
        code = int(self.cudart.cudaPeekAtLastError())
        if code != CUDAError.SUCCESS:
            cstr = self.cudart.cudaGetErrorString(code)
            error_str = cstr.decode("utf-8") if cstr is not None else f"unknown error {code}"
            _logger.warning(
                "Sticky CUDA error detected after %s: %s (code %d). "
                "The CUDA context is poisoned — restart the process. "
                "Set CUDALINK_STICKY_ERROR_CHECK=0 to disable this check.",
                context,
                error_str,
                code,
            )
            raise RuntimeError(
                f"Sticky CUDA error after {context}: {error_str} (code {code}). "
                "The CUDA context is poisoned. Restart the process or set "
                "CUDALINK_STICKY_ERROR_CHECK=0 to disable this check."
            )

    def host_register(self, ptr: int, size: int, flags: int = 0) -> None:
        """Page-lock an existing host allocation via cudaHostRegister.

        Args:
            ptr: Host pointer as integer (e.g., arr.ctypes.data)
            size: Number of bytes to register
            flags: Registration flags (0=default, 1=portable, 2=mapped, 4=write-combined)

        Raises:
            RuntimeError: If registration fails
        """
        self.cudart.cudaHostRegister(c_void_p(ptr), c_size_t(size), c_uint(flags))

    def host_unregister(self, ptr: int) -> None:
        """Unregister a page-locked host allocation registered with host_register().

        Args:
            ptr: Host pointer as integer (same value passed to host_register())

        Raises:
            RuntimeError: If unregistration fails
        """
        self.cudart.cudaHostUnregister(c_void_p(ptr))

    # High-level API

    def malloc(self, size: int) -> c_void_p:
        """Allocate GPU memory.

        Args:
            size: Number of bytes to allocate

        Returns:
            Device pointer to allocated memory

        Raises:
            RuntimeError: If allocation fails
        """
        dev_ptr = c_void_p()
        self.cudart.cudaMalloc(byref(dev_ptr), size)
        return dev_ptr

    def free(self, dev_ptr: c_void_p) -> None:
        """Free GPU memory.

        Args:
            dev_ptr: Device pointer to free

        Raises:
            RuntimeError: If free fails
        """
        self.cudart.cudaFree(dev_ptr)

    def set_device(self, device: int) -> int:
        """Switch the calling thread to the device's CUDA primary context.

        Uses the driver API (nvcuda.dll) to save the current context and install
        the primary context, ensuring cudaMalloc allocates in a context whose IPC
        handles are portable to other processes.

        On systems where nvcuda.dll is unavailable, falls back to the runtime-API
        cudaSetDevice (handles the empty-driver-stack case only).

        Returns an opaque integer token for restore_context(). The caller must
        call restore_context() after the allocation block is complete.
        """
        if self._drv is not None:
            cu_dev = c_int()
            self._drv.cuDeviceGet(byref(cu_dev), device)
            saved_ctx = c_void_p()
            self._drv.cuCtxGetCurrent(byref(saved_ctx))
            primary_ctx = c_void_p()
            self._drv.cuDevicePrimaryCtxRetain(byref(primary_ctx), cu_dev)
            self._drv.cuCtxSetCurrent(primary_ctx)
            return saved_ctx.value or 0
        # Fallback: runtime API only — effective when driver-API stack is empty
        self.cudart.cudaSetDevice(c_int(device))
        return 0

    def restore_context(self, token: int) -> None:
        """Restore the driver-API context saved by the preceding set_device() call.

        token is the value returned by set_device(). No-op if the driver API was
        unavailable (token will be 0 and self._drv will be None).
        """
        if self._drv is not None:
            self._drv.cuCtxSetCurrent(c_void_p(token if token else None))

    def malloc_host(self, size: int) -> c_void_p:
        """Allocate pinned (page-locked) host memory via cudaMallocHost.

        Pinned memory enables direct DMA for D2H transfers, eliminating the
        CUDA driver's internal staging copy that pageable memory requires.

        Note: this project is single-GPU by construction (get_cuda_runtime rejects
        a second device). Multi-GPU would require cudaHostAlloc with
        cudaHostAllocPortable for cross-device visibility (Handbook §5.1).

        Args:
            size: Number of bytes to allocate

        Returns:
            Host pointer to pinned memory

        Raises:
            RuntimeError: If allocation fails
        """
        ptr = c_void_p()
        self.cudart.cudaMallocHost(byref(ptr), size)
        return ptr

    def free_host(self, ptr: c_void_p) -> None:
        """Free pinned host memory allocated with malloc_host().

        Args:
            ptr: Host pointer to free

        Raises:
            RuntimeError: If free fails
        """
        self.cudart.cudaFreeHost(ptr)

    def memcpy(self, dst: c_void_p, src: c_void_p, count: int, kind: int) -> None:
        """Copy memory (device-to-device, host-to-device, or device-to-host).

        Args:
            dst: Destination pointer
            src: Source pointer
            count: Number of bytes to copy
            kind: MemcpyKind value (e.g. MemcpyKind.DEVICE_TO_DEVICE)

        Raises:
            RuntimeError: If copy fails
        """
        self.cudart.cudaMemcpy(dst, src, count, kind)

    def ipc_get_mem_handle(self, dev_ptr: c_void_p) -> cudaIpcMemHandle_t:
        """Get IPC handle for GPU memory.

        This handle can be transferred to another process via SharedMemory
        or other IPC mechanism.

        Args:
            dev_ptr: Device pointer to export

        Returns:
            IPC handle (128 bytes)

        Raises:
            RuntimeError: If export fails
        """
        handle = cudaIpcMemHandle_t()
        self.cudart.cudaIpcGetMemHandle(byref(handle), dev_ptr)
        return handle

    def ipc_open_mem_handle(self, handle: cudaIpcMemHandle_t, flags: int = IPC_MEM_LAZY_ENABLE_PEER_ACCESS) -> c_void_p:
        """Open IPC handle to access GPU memory from another process.

        Args:
            handle: IPC handle received from another process
            flags: IPC flags (IPC_MEM_LAZY_ENABLE_PEER_ACCESS = cudaIpcMemLazyEnablePeerAccess)

        Returns:
            Device pointer to shared memory

        Raises:
            RuntimeError: If opening fails
        """
        # Guard against ctypes class-identity mismatch: in TD's bare-name import namespace
        # the caller's cudaIpcMemHandle_t may be a *different* class object than the one bound
        # into argtypes (two independent imports of cuda_link.cuda_runtime_types, e.g. a
        # CUDARuntimeTypes mirror Text DAT loaded alongside the library-mode package).
        # ctypes validates by class identity, not structural equivalence, so the call would
        # raise ArgumentError. Rebuild from raw bytes into THIS module's class — POD, 64 bytes.
        if not isinstance(handle, cudaIpcMemHandle_t):
            handle = cudaIpcMemHandle_t.from_buffer_copy(bytes(handle))
        dev_ptr = c_void_p()
        self.cudart.cudaIpcOpenMemHandle(byref(dev_ptr), handle, flags)
        return dev_ptr

    def ipc_close_mem_handle(self, dev_ptr: c_void_p) -> None:
        """Close IPC memory handle.

        Args:
            dev_ptr: Device pointer obtained from ipc_open_mem_handle()

        Raises:
            RuntimeError: If closing fails
        """
        self.cudart.cudaIpcCloseMemHandle(dev_ptr)

    def synchronize(self) -> None:
        """Synchronize all CUDA operations on current device.

        Raises:
            RuntimeError: If synchronization fails
        """
        self.cudart.cudaDeviceSynchronize()

    # CUDA Event API (for async synchronization)

    def create_ipc_event(self) -> CUDAEvent_t:
        """Create CUDA event suitable for IPC (interprocess communication).

        Returns:
            Event handle for cross-process synchronization

        Raises:
            RuntimeError: If event creation fails
        """
        event = CUDAEvent_t()
        # cudaEventInterprocess (4) | cudaEventDisableTiming (2) = 6
        # NVIDIA requires cudaEventDisableTiming when using cudaEventInterprocess
        self.cudart.cudaEventCreateWithFlags(byref(event), 6)
        return event

    def record_event(self, event: CUDAEvent_t, stream: CUDAStream_t | None = None) -> None:
        """Record event on specified stream (or default stream).

        Args:
            event: Event handle to record
            stream: CUDA stream (None = default stream)

        Raises:
            RuntimeError: If event recording fails
        """
        # Convert None to CUDA default stream (0) for ctypes compatibility
        if stream is None:
            stream = CUDAStream_t(0)
        self.cudart.cudaEventRecord(event, stream)

    def query_event(self, event: CUDAEvent_t) -> bool:
        """Query if event has completed (non-blocking).

        Args:
            event: Event handle to query

        Returns:
            True if event completed, False if still pending

        Raises:
            RuntimeError: If query fails with unexpected error
        """
        result = self.cudart.cudaEventQuery(event)
        if result == CUDAError.SUCCESS:
            return True
        elif result == CUDAError.NOT_READY:
            return False
        self.check_error(result, "cudaEventQuery")
        return False

    def wait_event(self, event: CUDAEvent_t) -> None:
        """Wait for event to complete (blocking).

        Args:
            event: Event handle to wait on

        Raises:
            RuntimeError: If wait fails
        """
        self.cudart.cudaEventSynchronize(event)

    def ipc_get_event_handle(self, event: CUDAEvent_t) -> cudaIpcEventHandle_t:
        """Get IPC handle for event (for cross-process signaling).

        Args:
            event: Event created with create_ipc_event()

        Returns:
            IPC event handle (64 bytes)

        Raises:
            RuntimeError: If export fails
        """
        handle = cudaIpcEventHandle_t()
        self.cudart.cudaIpcGetEventHandle(byref(handle), event)
        return handle

    def ipc_open_event_handle(self, handle: cudaIpcEventHandle_t) -> CUDAEvent_t:
        """Open IPC event handle from another process.

        Args:
            handle: IPC event handle received from another process

        Returns:
            Event handle for this process

        Raises:
            RuntimeError: If opening fails
        """
        # Same class-identity guard as ipc_open_mem_handle — see note there.
        if not isinstance(handle, cudaIpcEventHandle_t):
            handle = cudaIpcEventHandle_t.from_buffer_copy(bytes(handle))
        event = CUDAEvent_t()
        self.cudart.cudaIpcOpenEventHandle(byref(event), handle)
        return event

    def destroy_event(self, event: CUDAEvent_t) -> None:
        """Destroy CUDA event.

        Args:
            event: Event handle to destroy

        Raises:
            RuntimeError: If destruction fails
        """
        self.cudart.cudaEventDestroy(event)

    def create_timing_event(self) -> CUDAEvent_t:
        """Create CUDA event suitable for GPU timing (NOT for IPC).

        Returns:
            Event handle for GPU-accurate timing measurements

        Raises:
            RuntimeError: If event creation fails

        Note:
            This creates an event with timing enabled (flags=0).
            Use this for benchmarking, NOT for IPC synchronization.
            IPC events require cudaEventDisableTiming flag.
        """
        event = CUDAEvent_t()
        # flags=0 enables timing (no cudaEventDisableTiming, no cudaEventInterprocess)
        self.cudart.cudaEventCreateWithFlags(byref(event), 0)
        return event

    def create_sync_event(self) -> CUDAEvent_t:
        """Create CUDA event optimized for stream ordering (NOT timing, NOT IPC).

        Returns:
            Event handle for use with stream_wait_event() ordering

        Raises:
            RuntimeError: If event creation fails

        Note:
            Uses cudaEventDisableTiming (0x02). Per NVIDIA docs this provides
            best performance when used with cudaStreamWaitEvent() and
            cudaEventQuery() — removes per-record timing instrumentation overhead.
            Do not use with event_elapsed_time(); use create_timing_event() for that.
        """
        event = CUDAEvent_t()
        # cudaEventDisableTiming = 0x02 — optimal for ordering-only events
        self.cudart.cudaEventCreateWithFlags(byref(event), 0x02)
        return event

    def event_elapsed_time(self, start: CUDAEvent_t, end: CUDAEvent_t) -> float:
        """Get elapsed GPU time between two events.

        Args:
            start: Starting event (must be recorded before end event)
            end: Ending event

        Returns:
            Elapsed time in milliseconds (GPU-measured)

        Raises:
            RuntimeError: If elapsed time query fails

        Note:
            Both events must have timing enabled (created with create_timing_event).
            Events with cudaEventDisableTiming flag cannot be used for timing.
        """
        elapsed_ms = c_float()
        self.cudart.cudaEventElapsedTime(byref(elapsed_ms), start, end)
        return elapsed_ms.value

    def get_device(self) -> int:
        """Return the CUDA device index currently bound to this context.

        Returns:
            Integer device index (matches self.device if context is healthy)

        Raises:
            RuntimeError: If query fails
        """
        device = c_int()
        self.cudart.cudaGetDevice(byref(device))
        return device.value

    def create_stream(self, flags: int = StreamFlags.NON_BLOCKING) -> CUDAStream_t:
        """Create CUDA stream with specified flags.

        Args:
            flags: Stream creation flags. Default StreamFlags.NON_BLOCKING (cudaStreamNonBlocking)

        Returns:
            CUDAStream_t: Opaque stream handle

        Raises:
            RuntimeError: If stream creation fails
        """
        stream = CUDAStream_t()
        self.cudart.cudaStreamCreateWithFlags(byref(stream), flags)
        return stream

    def create_stream_with_priority(
        self, flags: int = StreamFlags.NON_BLOCKING, priority: int | None = None
    ) -> CUDAStream_t:
        """Create CUDA stream at the specified (or highest available) priority.

        On CUDA, stream priority is an integer where a smaller value means
        higher priority. cudaDeviceGetStreamPriorityRange returns [least, greatest]
        where greatest is the most-negative value — i.e., the highest priority.

        Args:
            flags: Stream flags. Default StreamFlags.NON_BLOCKING (cudaStreamNonBlocking).
            priority: Stream priority. None means use highest available (greatest).

        Returns:
            CUDAStream_t: Opaque stream handle

        Raises:
            RuntimeError: If stream creation fails
        """
        if priority is None:
            least = c_int()
            greatest = c_int()
            self.cudart.cudaDeviceGetStreamPriorityRange(byref(least), byref(greatest))
            priority = greatest.value
        stream = CUDAStream_t()
        self.cudart.cudaStreamCreateWithPriority(byref(stream), flags, priority)
        return stream

    def destroy_stream(self, stream: CUDAStream_t) -> None:
        """Destroy CUDA stream.

        Args:
            stream: Stream handle to destroy

        Raises:
            RuntimeError: If destruction fails
        """
        self.cudart.cudaStreamDestroy(stream)

    def stream_wait_event(self, stream: CUDAStream_t, event: CUDAEvent_t, flags: int = 0) -> None:
        """Make stream wait on event (GPU-side, non-blocking to CPU).

        Args:
            stream: Stream to wait
            event: Event to wait for
            flags: Wait flags (default 0)

        Raises:
            RuntimeError: If wait enqueue fails
        """
        self.cudart.cudaStreamWaitEvent(stream, event, flags)

    def stream_synchronize(self, stream: CUDAStream_t) -> None:
        """Wait for all operations on stream to complete (CPU-blocking).

        Args:
            stream: Stream to synchronize

        Raises:
            RuntimeError: If synchronization fails
        """
        self.cudart.cudaStreamSynchronize(stream)

    def memcpy_async(self, dst: c_void_p, src: c_void_p, count: int, kind: int, stream: CUDAStream_t) -> None:
        """Asynchronous memory copy on a stream.

        Args:
            dst: Destination pointer
            src: Source pointer
            count: Number of bytes to copy
            kind: MemcpyKind value (e.g. MemcpyKind.DEVICE_TO_DEVICE)
            stream: CUDA stream for async operation

        Raises:
            RuntimeError: If async copy enqueue fails
        """
        self.cudart.cudaMemcpyAsync(dst, src, count, kind, stream)

    def mem_get_info(self) -> tuple[int, int]:
        """Get free and total device memory in bytes.

        Returns:
            Tuple of (free_bytes, total_bytes)

        Raises:
            RuntimeError: If query fails
        """
        free = c_size_t()
        total = c_size_t()
        self.cudart.cudaMemGetInfo(byref(free), byref(total))
        return free.value, total.value

    def stream_query(self, stream: CUDAStream_t) -> bool:
        """Non-blocking check if all operations on stream have completed.

        Args:
            stream: CUDA stream to query

        Returns:
            True if all stream operations have completed, False if still executing

        Raises:
            RuntimeError: If query fails with an error other than cudaErrorNotReady
        """
        result = self.cudart.cudaStreamQuery(stream)
        if result == CUDAError.SUCCESS:
            return True
        if result == CUDAError.NOT_READY:
            return False
        self.check_error(result, "cudaStreamQuery")
        return False  # unreachable

    def pointer_get_attributes(self, ptr: int) -> cudaPointerAttributes:
        """Query memory type and owning device for a GPU pointer.

        Args:
            ptr: GPU pointer as integer (e.g., tensor.data_ptr())

        Returns:
            cudaPointerAttributes with .type (2=device, 3=managed) and .device (GPU index)

        Raises:
            RuntimeError: If query fails (e.g., unregistered host pointer passed)
        """
        attrs = cudaPointerAttributes()
        self.cudart.cudaPointerGetAttributes(byref(attrs), c_void_p(ptr))
        return attrs

    def device_can_access_peer(self, device: int, peer_device: int) -> bool:
        """Check if device can directly access peer_device memory via IPC/NVLink.

        Useful for validating multi-GPU setups before attempting IPC handle operations.
        On single-GPU systems or systems without peer access, cudaIpcOpenMemHandle
        may fall back to slower paths without warning.

        Args:
            device: Source device ID
            peer_device: Target peer device ID

        Returns:
            True if direct peer access is available, False otherwise

        Raises:
            RuntimeError: If query fails
        """
        can_access = c_int(0)
        self.cudart.cudaDeviceCanAccessPeer(byref(can_access), device, peer_device)
        return bool(can_access.value)

    # --- Phase 1: cudaHostAlloc (replaces cudaMallocHost with portable flag) ---

    def malloc_host_alloc(self, size: int, flags: int = HOST_ALLOC_PORTABLE) -> c_void_p:
        """Allocate pinned host memory via cudaHostAlloc with explicit flags.

        Unlike malloc_host() which calls cudaMallocHost (no flags), this lets
        callers pass HOST_ALLOC_PORTABLE (cudaHostAllocPortable) to make the
        allocation visible from any CUDA context in the process — useful when
        PyTorch and CuPy share the same process.

        Args:
            size:  Number of bytes to allocate.
            flags: OR-combination of:
                   HOST_ALLOC_PORTABLE (cudaHostAllocPortable = 0x01, cross-context visibility)
                   cudaHostAllocMapped      = 0x02 (map into device address space)
                   cudaHostAllocWriteCombined = 0x04 (WC; fast write, slow CPU read)

        Returns:
            Host pointer to allocated pinned memory.

        Raises:
            RuntimeError: If allocation fails.
        """
        ptr = c_void_p()
        self.cudart.cudaHostAlloc(byref(ptr), c_size_t(size), c_uint(flags))
        return ptr

    # --- Phase 0: device attribute query ---

    def get_device_attribute(self, attr: int, device: int | None = None) -> int:
        """Query a cudaDeviceAttr value for a given device.

        Common attrs:
            cudaDevAttrAsyncEngineCount = 4 — number of DMA copy engines

        Args:
            attr:   cudaDeviceAttr integer constant.
            device: GPU device index. Defaults to self.device.

        Returns:
            Integer attribute value.

        Raises:
            RuntimeError: If query fails.
        """
        if device is None:
            device = self.device
        value = c_int()
        self.cudart.cudaDeviceGetAttribute(byref(value), c_int(attr), c_int(device))
        return value.value


# Global singleton instance (lazy initialization)
_cuda_runtime: CUDARuntimeAPI | None = None


def get_cuda_runtime(device: int = 0) -> CUDARuntimeAPI:
    """Get global CUDA runtime instance (singleton).

    The singleton is created on first call. Subsequent calls with a *different*
    device index will raise RuntimeError — a single process context can only
    be bound to one device via this shared-cudart pattern.

    Args:
        device: CUDA device index (default 0). Must match across all callers
                within the same process.

    Returns:
        CUDARuntimeAPI: Global CUDA runtime wrapper

    Raises:
        RuntimeError: If called with a device index that conflicts with the
                      already-initialized singleton.
    """
    global _cuda_runtime
    if _cuda_runtime is None:
        _cuda_runtime = CUDARuntimeAPI(device=device)
    elif _cuda_runtime.device != device:
        raise RuntimeError(
            f"CUDA runtime singleton was initialized for device {_cuda_runtime.device}, "
            f"but caller requested device {device}. A single process can only bind to "
            "one device via the shared-cudart singleton. Create a separate "
            "CUDARuntimeAPI(device=...) instance for multi-device use."
        )
    return _cuda_runtime
