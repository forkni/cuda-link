"""
CUDA IPC Wrapper for Windows
Based on vLLM cuda_wrapper.py pattern

Provides ctypes interface to CUDA Runtime API for inter-process communication.
Compatible with both TouchDesigner and Python processes.

Requirements:
- CUDA 12.x runtime (cudart64_12.dll)
- Windows operating system
- Same GPU visible to both processes
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import POINTER, byref, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p

_logger = logging.getLogger(__name__)

# CUDA handle types - use unsigned 64-bit to prevent overflow on Windows x64
# See: https://github.com/pytorch/pytorch/pull/162920
CUDAEvent_t     = c_uint64  # cudaEvent_t opaque pointer
CUDAStream_t    = c_uint64  # cudaStream_t opaque pointer
CUDAGraph_t     = c_uint64  # cudaGraph_t opaque pointer (CUDA 10.0+)
CUDAGraphExec_t = c_uint64  # cudaGraphExec_t opaque pointer (CUDA 10.0+)
CUDAGraphNode_t = c_uint64  # cudaGraphNode_t opaque pointer (CUDA 10.0+)

# --- CUDA Graph parameter structs ---

class cudaPos(ctypes.Structure):
    """cudaPos: {x, y, z} offsets into an array or pitched memory."""
    _fields_ = [("x", c_size_t), ("y", c_size_t), ("z", c_size_t)]


class cudaPitchedPtr(ctypes.Structure):
    """cudaPitchedPtr: pointer + pitch metadata for 2D/3D copies."""
    _fields_ = [
        ("ptr",   c_void_p),
        ("pitch", c_size_t),
        ("xsize", c_size_t),
        ("ysize", c_size_t),
    ]


class cudaExtent(ctypes.Structure):
    """cudaExtent: width/height/depth dimensions in bytes for 3D copies."""
    _fields_ = [("width", c_size_t), ("height", c_size_t), ("depth", c_size_t)]


class cudaMemcpy3DParms(ctypes.Structure):
    """cudaMemcpy3DParms: full parameter struct for cudaMemcpy3D and graph node updates."""
    _fields_ = [
        ("srcArray", c_void_p),    # cudaArray_t — NULL for linear memory
        ("srcPos",   cudaPos),
        ("srcPtr",   cudaPitchedPtr),
        ("dstArray", c_void_p),    # cudaArray_t — NULL for linear memory
        ("dstPos",   cudaPos),
        ("dstPtr",   cudaPitchedPtr),
        ("extent",   cudaExtent),
        ("kind",     c_int),       # cudaMemcpyKind
    ]


# CUDA IPC Handle structure (64 bytes, CUDA_IPC_HANDLE_SIZE per NVIDIA spec)
class cudaIpcMemHandle_t(ctypes.Structure):
    """CUDA IPC memory handle structure.

    This opaque handle can be transferred between processes via
    SharedMemory or other IPC mechanisms to enable GPU memory sharing.
    """

    _fields_ = [("internal", ctypes.c_byte * 64)]


# CUDA IPC Event Handle structure (64 bytes per NVIDIA spec)
class cudaIpcEventHandle_t(ctypes.Structure):
    """CUDA IPC event handle structure.

    Used for lightweight cross-process synchronization.
    """

    _fields_ = [("reserved", ctypes.c_byte * 64)]


# CUDA pointer attributes — memory type and owning device for a GPU pointer
class cudaPointerAttributes(ctypes.Structure):
    """Result of cudaPointerGetAttributes.

    Useful for validating that a caller-supplied GPU pointer belongs to the
    expected device before issuing D2D operations (C2 affinity check).

    .type values: 0=unregistered, 1=host, 2=device, 3=managed
    .device: GPU index that owns the allocation
    """

    _fields_ = [
        ("type", c_int),  # cudaMemoryType enum (2 = cudaMemoryTypeDevice)
        ("device", c_int),  # GPU device index owning this allocation
        ("devicePointer", c_void_p),
        ("hostPointer", c_void_p),
    ]


# CUDA Error codes (subset)
class CUDAError:
    """CUDA runtime error codes."""

    SUCCESS = 0
    INVALID_VALUE = 1
    MEMORY_ALLOCATION = 2
    INVALID_DEVICE_POINTER = 17
    INVALID_DEVICE = 101
    INVALID_CONTEXT = 201  # Common in same-process IPC testing
    NOT_READY = 600
    PEER_ACCESS_ALREADY_ENABLED = 704

    @staticmethod
    def get_name(code: int) -> str:
        """Get human-readable error name."""
        names = {
            0: "SUCCESS",
            1: "INVALID_VALUE",
            2: "MEMORY_ALLOCATION",
            17: "INVALID_DEVICE_POINTER",
            101: "INVALID_DEVICE",
            201: "INVALID_CONTEXT",
            600: "NOT_READY",
            704: "PEER_ACCESS_ALREADY_ENABLED",
        }
        return names.get(code, f"UNKNOWN_ERROR_{code}")


class CUDARuntimeAPI:
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
        # Establish CUDA primary context on the requested device.
        # Prevents cudaIpcOpenMemHandle error 400 when a second cudart DLL is loaded
        # alongside torch (which has its own bundled cudart). Each DLL instance needs
        # its own context initialized before IPC handle operations can succeed.
        self.cudart.cudaSetDevice(device)

        if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
            _logger.warning(
                "CUDA_LAUNCH_BLOCKING=1 is set — all CUDA operations are serialized. "
                "This causes ~30x slower frame rates and should only be used for debugging."
            )

        # Default ON; set CUDALINK_STICKY_ERROR_CHECK=0 to skip the cudaPeekAtLastError call.
        self._sticky_check_enabled: bool = os.environ.get("CUDALINK_STICKY_ERROR_CHECK", "1") != "0"

    def _load_cuda_runtime(self) -> ctypes.CDLL:
        """Load CUDA runtime DLL.

        Returns:
            ctypes.CDLL: Loaded CUDA runtime library

        Raises:
            RuntimeError: If CUDA runtime cannot be loaded
        """
        # Try by name FIRST: if cudart is already loaded in this process (e.g., by
        # torch), Windows returns the cached handle — ensuring we share the same
        # runtime instance and CUDA context. Loading by full path can create a second
        # independent instance with its own state, breaking cross-process IPC.
        dll_names = ["cudart64_110.dll", "cudart64_12.dll", "cudart64_11.dll"]
        for name in dll_names:
            try:
                dll = ctypes.CDLL(name)
                self._log_dll_path(dll, name)
                return dll
            except OSError:
                continue

        # Fallback: try full toolkit paths when not already in PATH
        dll_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudart64_12.dll",
        ]
        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                try:
                    dll = ctypes.CDLL(dll_path)
                    self._log_dll_path(dll, dll_path)
                    return dll
                except OSError:
                    continue

        raise RuntimeError(
            "Could not load CUDA runtime. Please ensure CUDA 12.x is installed.\n"
            f"Tried names: {dll_names}\n"
            f"Tried paths: {dll_paths}"
        )

    @staticmethod
    def _log_dll_path(dll: ctypes.CDLL, hint: str) -> None:
        """Log the resolved filesystem path of a loaded DLL (Windows only)."""
        try:
            buf = ctypes.create_unicode_buffer(260)
            # GetModuleFileNameW needs HMODULE as c_void_p to avoid 32-bit overflow
            ctypes.windll.kernel32.GetModuleFileNameW(ctypes.c_void_p(dll._handle), buf, 260)
            _logger.debug("Loaded CUDA runtime: %s", buf.value)
        except (OSError, AttributeError) as e:
            _logger.debug("Could not log DLL path: %s", e)

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

        # cudaGetLastError()
        self.cudart.cudaGetLastError.argtypes = []
        self.cudart.cudaGetLastError.restype = c_int

        # cudaPeekAtLastError() — non-destructive sticky-error read (does NOT clear the error)
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

        # --- CUDA Graph API (CUDA 10.0+, full set needed for CUDALINK_USE_GRAPHS) ---

        # cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
        # mode: 0=global, 1=thread_local, 2=relaxed
        self.cudart.cudaStreamBeginCapture.argtypes = [CUDAStream_t, c_int]
        self.cudart.cudaStreamBeginCapture.restype = c_int

        # cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)
        self.cudart.cudaStreamEndCapture.argtypes = [CUDAStream_t, POINTER(CUDAGraph_t)]
        self.cudart.cudaStreamEndCapture.restype = c_int

        # cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
        #                      unsigned long long flags)   [CUDA 12.0 simplified form]
        self.cudart.cudaGraphInstantiate.argtypes = [
            POINTER(CUDAGraphExec_t), CUDAGraph_t, c_uint64
        ]
        self.cudart.cudaGraphInstantiate.restype = c_int

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
        self.cudart.cudaGraphGetNodes.argtypes = [
            CUDAGraph_t, POINTER(CUDAGraphNode_t), POINTER(c_size_t)
        ]
        self.cudart.cudaGraphGetNodes.restype = c_int

        # cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t, cudaGraphNode_t,
        #                                  const cudaMemcpy3DParms*)
        # Updates a 3D-captured memcpy node. For nodes captured from cudaMemcpyAsync
        # (1D form) use cudaGraphExecMemcpyNodeSetParams1D instead.
        self.cudart.cudaGraphExecMemcpyNodeSetParams.argtypes = [
            CUDAGraphExec_t, CUDAGraphNode_t, POINTER(cudaMemcpy3DParms)
        ]
        self.cudart.cudaGraphExecMemcpyNodeSetParams.restype = c_int

        # cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t, cudaGraphNode_t,
        #                                    void* dst, const void* src,
        #                                    size_t count, cudaMemcpyKind kind)
        # Updates a 1D memcpy node (captured from cudaMemcpyAsync). CUDA 11.3+.
        self.cudart.cudaGraphExecMemcpyNodeSetParams1D.argtypes = [
            CUDAGraphExec_t, CUDAGraphNode_t, c_void_p, c_void_p, c_size_t, c_int
        ]
        self.cudart.cudaGraphExecMemcpyNodeSetParams1D.restype = c_int

        # cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t, cudaGraphNode_t,
        #                                      cudaEvent_t event)
        # Updates the event recorded by an event-record node. CUDA 11.4+.
        self.cudart.cudaGraphExecEventRecordNodeSetEvent.argtypes = [
            CUDAGraphExec_t, CUDAGraphNode_t, CUDAEvent_t
        ]
        self.cudart.cudaGraphExecEventRecordNodeSetEvent.restype = c_int

        # cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t, cudaGraphNode_t,
        #                                    cudaEvent_t event)
        # Updates the event waited on by an event-wait node. CUDA 11.4+.
        self.cudart.cudaGraphExecEventWaitNodeSetEvent.argtypes = [
            CUDAGraphExec_t, CUDAGraphNode_t, CUDAEvent_t
        ]
        self.cudart.cudaGraphExecEventWaitNodeSetEvent.restype = c_int

    def check_error(self, result: int, operation: str) -> None:
        """Check CUDA error code and raise exception if failed.

        Args:
            result: CUDA error code
            operation: Description of the operation that failed

        Raises:
            RuntimeError: If result indicates an error
        """
        if result != CUDAError.SUCCESS:
            error_str = self.cudart.cudaGetErrorString(result).decode("utf-8")
            error_name = CUDAError.get_name(result)
            raise RuntimeError(f"CUDA {operation} failed: {error_str} (error {result}: {error_name})")

    def peek_at_last_error(self) -> int:
        """Non-destructively read the thread-local sticky CUDA error.

        Returns SUCCESS (0) normally. A non-zero value means a prior async
        operation (memcpy, kernel) failed and the error was not yet consumed.
        Unlike cudaGetLastError this does NOT clear the latched error state.
        """
        return int(self.cudart.cudaPeekAtLastError())

    def check_sticky_error(self, context: str) -> None:
        """Warn and raise if a sticky CUDA error is latched from a prior async op.

        No-op when CUDALINK_STICKY_ERROR_CHECK=0. Enabled by default.
        Use peek_at_last_error() directly for the raw value without raising.
        """
        if not self._sticky_check_enabled:
            return
        code = int(self.cudart.cudaPeekAtLastError())
        if code != CUDAError.SUCCESS:
            error_str = self.cudart.cudaGetErrorString(code).decode("utf-8")
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
        result = self.cudart.cudaHostRegister(c_void_p(ptr), c_size_t(size), c_uint(flags))
        self.check_error(result, "cudaHostRegister")

    def host_unregister(self, ptr: int) -> None:
        """Unregister a page-locked host allocation registered with host_register().

        Args:
            ptr: Host pointer as integer (same value passed to host_register())

        Raises:
            RuntimeError: If unregistration fails
        """
        result = self.cudart.cudaHostUnregister(c_void_p(ptr))
        self.check_error(result, "cudaHostUnregister")

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
        result = self.cudart.cudaMalloc(byref(dev_ptr), size)
        self.check_error(result, "cudaMalloc")
        return dev_ptr

    def free(self, dev_ptr: c_void_p) -> None:
        """Free GPU memory.

        Args:
            dev_ptr: Device pointer to free

        Raises:
            RuntimeError: If free fails
        """
        result = self.cudart.cudaFree(dev_ptr)
        self.check_error(result, "cudaFree")

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
        result = self.cudart.cudaMallocHost(byref(ptr), size)
        self.check_error(result, "cudaMallocHost")
        return ptr

    def free_host(self, ptr: c_void_p) -> None:
        """Free pinned host memory allocated with malloc_host().

        Args:
            ptr: Host pointer to free

        Raises:
            RuntimeError: If free fails
        """
        result = self.cudart.cudaFreeHost(ptr)
        self.check_error(result, "cudaFreeHost")

    def memcpy(self, dst: c_void_p, src: c_void_p, count: int, kind: int) -> None:
        """Copy memory (device-to-device, host-to-device, or device-to-host).

        Args:
            dst: Destination pointer
            src: Source pointer
            count: Number of bytes to copy
            kind: cudaMemcpyKind (0=H2H, 1=H2D, 2=D2H, 3=D2D)

        Raises:
            RuntimeError: If copy fails
        """
        result = self.cudart.cudaMemcpy(dst, src, count, kind)
        self.check_error(result, "cudaMemcpy")

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
        result = self.cudart.cudaIpcGetMemHandle(byref(handle), dev_ptr)
        self.check_error(result, "cudaIpcGetMemHandle")
        return handle

    def ipc_open_mem_handle(self, handle: cudaIpcMemHandle_t, flags: int = 1) -> c_void_p:
        """Open IPC handle to access GPU memory from another process.

        Args:
            handle: IPC handle received from another process
            flags: IPC flags (1 = cudaIpcMemLazyEnablePeerAccess)

        Returns:
            Device pointer to shared memory

        Raises:
            RuntimeError: If opening fails
        """
        dev_ptr = c_void_p()
        result = self.cudart.cudaIpcOpenMemHandle(byref(dev_ptr), handle, flags)
        self.check_error(result, "cudaIpcOpenMemHandle")
        return dev_ptr

    def ipc_close_mem_handle(self, dev_ptr: c_void_p) -> None:
        """Close IPC memory handle.

        Args:
            dev_ptr: Device pointer obtained from ipc_open_mem_handle()

        Raises:
            RuntimeError: If closing fails
        """
        result = self.cudart.cudaIpcCloseMemHandle(dev_ptr)
        self.check_error(result, "cudaIpcCloseMemHandle")

    def synchronize(self) -> None:
        """Synchronize all CUDA operations on current device.

        Raises:
            RuntimeError: If synchronization fails
        """
        result = self.cudart.cudaDeviceSynchronize()
        self.check_error(result, "cudaDeviceSynchronize")

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
        result = self.cudart.cudaEventCreateWithFlags(byref(event), 6)
        self.check_error(result, "cudaEventCreateWithFlags")
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
        result = self.cudart.cudaEventRecord(event, stream)
        self.check_error(result, "cudaEventRecord")

    def query_event(self, event: c_void_p) -> bool:
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
        result = self.cudart.cudaEventSynchronize(event)
        self.check_error(result, "cudaEventSynchronize")

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
        result = self.cudart.cudaIpcGetEventHandle(byref(handle), event)
        self.check_error(result, "cudaIpcGetEventHandle")
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
        event = CUDAEvent_t()
        result = self.cudart.cudaIpcOpenEventHandle(byref(event), handle)
        self.check_error(result, "cudaIpcOpenEventHandle")
        return event

    def destroy_event(self, event: CUDAEvent_t) -> None:
        """Destroy CUDA event.

        Args:
            event: Event handle to destroy

        Raises:
            RuntimeError: If destruction fails
        """
        result = self.cudart.cudaEventDestroy(event)
        self.check_error(result, "cudaEventDestroy")

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
        result = self.cudart.cudaEventCreateWithFlags(byref(event), 0)
        self.check_error(result, "cudaEventCreateWithFlags(timing)")
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
        result = self.cudart.cudaEventCreateWithFlags(byref(event), 0x02)
        self.check_error(result, "cudaEventCreateWithFlags(sync)")
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
        result = self.cudart.cudaEventElapsedTime(byref(elapsed_ms), start, end)
        self.check_error(result, "cudaEventElapsedTime")
        return elapsed_ms.value

    def get_device(self) -> int:
        """Return the CUDA device index currently bound to this context.

        Returns:
            Integer device index (matches self.device if context is healthy)

        Raises:
            RuntimeError: If query fails
        """
        device = c_int()
        result = self.cudart.cudaGetDevice(byref(device))
        self.check_error(result, "cudaGetDevice")
        return device.value

    def create_stream(self, flags: int = 0x01) -> CUDAStream_t:
        """Create CUDA stream with specified flags.

        Args:
            flags: Stream creation flags. Default 0x01 = cudaStreamNonBlocking

        Returns:
            CUDAStream_t: Opaque stream handle

        Raises:
            RuntimeError: If stream creation fails
        """
        stream = CUDAStream_t()
        result = self.cudart.cudaStreamCreateWithFlags(byref(stream), flags)
        self.check_error(result, "cudaStreamCreateWithFlags")
        return stream

    def create_stream_with_priority(self, flags: int = 0x01, priority: int | None = None) -> CUDAStream_t:
        """Create CUDA stream at the specified (or highest available) priority.

        On CUDA, stream priority is an integer where a smaller value means
        higher priority. cudaDeviceGetStreamPriorityRange returns [least, greatest]
        where greatest is the most-negative value — i.e., the highest priority.

        Args:
            flags: Stream flags. Default 0x01 = cudaStreamNonBlocking.
            priority: Stream priority. None means use highest available (greatest).

        Returns:
            CUDAStream_t: Opaque stream handle

        Raises:
            RuntimeError: If stream creation fails
        """
        if priority is None:
            least = c_int()
            greatest = c_int()
            result = self.cudart.cudaDeviceGetStreamPriorityRange(byref(least), byref(greatest))
            self.check_error(result, "cudaDeviceGetStreamPriorityRange")
            priority = greatest.value
        stream = CUDAStream_t()
        result = self.cudart.cudaStreamCreateWithPriority(byref(stream), flags, priority)
        self.check_error(result, "cudaStreamCreateWithPriority")
        return stream

    def destroy_stream(self, stream: CUDAStream_t) -> None:
        """Destroy CUDA stream.

        Args:
            stream: Stream handle to destroy

        Raises:
            RuntimeError: If destruction fails
        """
        result = self.cudart.cudaStreamDestroy(stream)
        self.check_error(result, "cudaStreamDestroy")

    def stream_wait_event(self, stream: CUDAStream_t, event: CUDAEvent_t, flags: int = 0) -> None:
        """Make stream wait on event (GPU-side, non-blocking to CPU).

        Args:
            stream: Stream to wait
            event: Event to wait for
            flags: Wait flags (default 0)

        Raises:
            RuntimeError: If wait enqueue fails
        """
        result = self.cudart.cudaStreamWaitEvent(stream, event, flags)
        self.check_error(result, "cudaStreamWaitEvent")

    def stream_synchronize(self, stream: CUDAStream_t) -> None:
        """Wait for all operations on stream to complete (CPU-blocking).

        Args:
            stream: Stream to synchronize

        Raises:
            RuntimeError: If synchronization fails
        """
        result = self.cudart.cudaStreamSynchronize(stream)
        self.check_error(result, "cudaStreamSynchronize")

    def memcpy_async(self, dst: c_void_p, src: c_void_p, count: int, kind: int, stream: CUDAStream_t) -> None:
        """Asynchronous memory copy on a stream.

        Args:
            dst: Destination pointer
            src: Source pointer
            count: Number of bytes to copy
            kind: cudaMemcpyKind (0=H2H, 1=H2D, 2=D2H, 3=D2D)
            stream: CUDA stream for async operation

        Raises:
            RuntimeError: If async copy enqueue fails
        """
        result = self.cudart.cudaMemcpyAsync(dst, src, count, kind, stream)
        self.check_error(result, "cudaMemcpyAsync")

    def mem_get_info(self) -> tuple[int, int]:
        """Get free and total device memory in bytes.

        Returns:
            Tuple of (free_bytes, total_bytes)

        Raises:
            RuntimeError: If query fails
        """
        free = c_size_t()
        total = c_size_t()
        result = self.cudart.cudaMemGetInfo(byref(free), byref(total))
        self.check_error(result, "cudaMemGetInfo")
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
        result = self.cudart.cudaPointerGetAttributes(byref(attrs), c_void_p(ptr))
        self.check_error(result, "cudaPointerGetAttributes")
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
        result = self.cudart.cudaDeviceCanAccessPeer(byref(can_access), device, peer_device)
        self.check_error(result, "cudaDeviceCanAccessPeer")
        return bool(can_access.value)

    # --- Phase 1: cudaHostAlloc (replaces cudaMallocHost with portable flag) ---

    def malloc_host_alloc(self, size: int, flags: int = 0x01) -> c_void_p:
        """Allocate pinned host memory via cudaHostAlloc with explicit flags.

        Unlike malloc_host() which calls cudaMallocHost (no flags), this lets
        callers pass cudaHostAllocPortable (0x01) to make the allocation visible
        from any CUDA context in the process — useful when PyTorch and CuPy share
        the same process.

        Args:
            size:  Number of bytes to allocate.
            flags: OR-combination of:
                   cudaHostAllocPortable    = 0x01 (cross-context visibility)
                   cudaHostAllocMapped      = 0x02 (map into device address space)
                   cudaHostAllocWriteCombined = 0x04 (WC; fast write, slow CPU read)

        Returns:
            Host pointer to allocated pinned memory.

        Raises:
            RuntimeError: If allocation fails.
        """
        ptr = c_void_p()
        result = self.cudart.cudaHostAlloc(byref(ptr), c_size_t(size), c_uint(flags))
        self.check_error(result, "cudaHostAlloc")
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
        result = self.cudart.cudaDeviceGetAttribute(byref(value), c_int(attr), c_int(device))
        self.check_error(result, "cudaDeviceGetAttribute")
        return value.value

    # --- Phase 2: CUDA Graph API wrappers ---

    def stream_begin_capture(self, stream: CUDAStream_t, mode: int = 0) -> None:
        """Begin capturing a stream into a CUDA graph.

        After this call, operations enqueued on *stream* are recorded into a
        graph rather than executed immediately. End with stream_end_capture().

        Args:
            stream: Stream to capture.
            mode:   cudaStreamCaptureMode — 0=global (safest), 1=thread_local,
                    2=relaxed. Use 0 unless you know what you're doing.

        Raises:
            RuntimeError: If capture start fails (e.g., stream already capturing).
        """
        result = self.cudart.cudaStreamBeginCapture(stream, c_int(mode))
        self.check_error(result, "cudaStreamBeginCapture")

    def stream_end_capture(self, stream: CUDAStream_t) -> CUDAGraph_t:
        """End stream capture and return the captured graph.

        After this call the stream resumes normal execution mode. The returned
        graph must be instantiated with graph_instantiate() before use, and
        destroyed with graph_destroy() when done.

        Args:
            stream: Stream that was passed to stream_begin_capture().

        Returns:
            CUDAGraph_t handle to the captured graph template.

        Raises:
            RuntimeError: If capture end fails.
        """
        graph = CUDAGraph_t()
        result = self.cudart.cudaStreamEndCapture(stream, byref(graph))
        self.check_error(result, "cudaStreamEndCapture")
        return graph

    def graph_instantiate(self, graph: CUDAGraph_t, flags: int = 0) -> CUDAGraphExec_t:
        """Instantiate a graph template into an executable graph.

        The executable graph (CUDAGraphExec_t) can be launched repeatedly via
        graph_launch(). The template graph can be destroyed after instantiation.

        Args:
            graph:  CUDAGraph_t template returned by stream_end_capture().
            flags:  cudaGraphInstantiateFlagDeviceLaunch (0x02) for device-side
                    launch; 0 for normal host-side launch.

        Returns:
            CUDAGraphExec_t executable graph handle.

        Raises:
            RuntimeError: If instantiation fails.
        """
        graph_exec = CUDAGraphExec_t()
        result = self.cudart.cudaGraphInstantiate(byref(graph_exec), graph, c_uint64(flags))
        self.check_error(result, "cudaGraphInstantiate")
        return graph_exec

    def graph_launch(self, graph_exec: CUDAGraphExec_t, stream: CUDAStream_t) -> None:
        """Launch an executable graph on a stream (single WDDM submission).

        This replaces N individual API calls (stream_wait_event, memcpy_async,
        record_event) with one batched WDDM submission, reducing kernel-mode
        transition overhead from N×~15µs to ~15µs on Windows WDDM.

        Args:
            graph_exec: Executable graph from graph_instantiate().
            stream:     Stream on which to launch the graph.

        Raises:
            RuntimeError: If launch fails.
        """
        result = self.cudart.cudaGraphLaunch(graph_exec, stream)
        self.check_error(result, "cudaGraphLaunch")

    def graph_get_nodes(self, graph: CUDAGraph_t) -> list[CUDAGraphNode_t]:
        """Return all nodes in a graph in topological (capture) order.

        Useful for discovering node handles after stream capture, before the
        template graph is destroyed.

        Args:
            graph: CUDAGraph_t template (must NOT yet be destroyed).

        Returns:
            List of CUDAGraphNode_t handles in capture order:
            [EventWaitNode (if present), MemcpyNode, EventRecordNode].

        Raises:
            RuntimeError: If query fails.
        """
        count = c_size_t(0)
        result = self.cudart.cudaGraphGetNodes(graph, None, byref(count))
        self.check_error(result, "cudaGraphGetNodes (count)")
        node_array = (CUDAGraphNode_t * count.value)()
        result = self.cudart.cudaGraphGetNodes(graph, node_array, byref(count))
        self.check_error(result, "cudaGraphGetNodes (fill)")
        return list(node_array)

    def graph_destroy(self, graph: CUDAGraph_t) -> None:
        """Destroy a graph template (not the executable — use graph_exec_destroy for that).

        Args:
            graph: Template graph to destroy.

        Raises:
            RuntimeError: If destruction fails.
        """
        result = self.cudart.cudaGraphDestroy(graph)
        self.check_error(result, "cudaGraphDestroy")

    def graph_exec_destroy(self, graph_exec: CUDAGraphExec_t) -> None:
        """Destroy an executable graph and free its resources.

        Args:
            graph_exec: Executable graph to destroy.

        Raises:
            RuntimeError: If destruction fails.
        """
        result = self.cudart.cudaGraphExecDestroy(graph_exec)
        self.check_error(result, "cudaGraphExecDestroy")

    @staticmethod
    def make_memcpy3d_params(
        dst: c_void_p, src: c_void_p, count: int, kind: int
    ) -> cudaMemcpy3DParms:
        """Build a cudaMemcpy3DParms struct for a flat 1D memory copy.

        Represents the copy as a single-row 2D memcpy (height=1, depth=1) so
        that 'count' bytes are transferred from src to dst. This is the required
        form for cudaGraphExecMemcpyNodeSetParams even when the original copy was
        issued as cudaMemcpyAsync (1D form).

        Args:
            dst:   Destination pointer.
            src:   Source pointer.
            count: Number of bytes to copy.
            kind:  cudaMemcpyKind (3 = DeviceToDevice).

        Returns:
            Populated cudaMemcpy3DParms instance.
        """
        params = cudaMemcpy3DParms()
        params.srcArray = None
        params.srcPos   = cudaPos(0, 0, 0)
        params.srcPtr   = cudaPitchedPtr(
            ptr=ctypes.cast(src, c_void_p),
            pitch=count, xsize=count, ysize=1,
        )
        params.dstArray = None
        params.dstPos   = cudaPos(0, 0, 0)
        params.dstPtr   = cudaPitchedPtr(
            ptr=ctypes.cast(dst, c_void_p),
            pitch=count, xsize=count, ysize=1,
        )
        params.extent = cudaExtent(width=count, height=1, depth=1)
        params.kind   = kind
        return params

    def graph_exec_memcpy_node_set_params(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
    ) -> None:
        """Update src/dst/count/kind of a memcpy node in an executable graph.

        This is a CPU-only operation (no WDDM submission). Changes take effect
        on the next graph_launch() call. The extent (count) must match the
        extent used when the graph was captured — only pointer reassignment
        within the same buffer size is supported.

        Args:
            graph_exec: Executable graph containing the node.
            node:       MemcpyNode handle from graph_get_nodes().
            dst:        New destination pointer.
            src:        New source pointer.
            count:      Copy size in bytes (must match captured size).
            kind:       cudaMemcpyKind (must match captured kind).

        Raises:
            RuntimeError: If parameter update fails.
        """
        params = self.make_memcpy3d_params(dst, src, count, kind)
        result = self.cudart.cudaGraphExecMemcpyNodeSetParams(graph_exec, node, byref(params))
        self.check_error(result, "cudaGraphExecMemcpyNodeSetParams")

    def graph_exec_memcpy_node_set_params_1d(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        dst: c_void_p,
        src: c_void_p,
        count: int,
        kind: int,
    ) -> None:
        """Update src/dst/count/kind of a 1D memcpy node in an executable graph.

        Use this for nodes captured from cudaMemcpyAsync (1D form). The 3D variant
        (graph_exec_memcpy_node_set_params) returns INVALID_VALUE on 1D nodes.
        Requires CUDA 11.3+.
        """
        dst_int = dst.value if isinstance(dst, c_void_p) else int(dst)
        src_int = src.value if isinstance(src, c_void_p) else int(src)
        result = self.cudart.cudaGraphExecMemcpyNodeSetParams1D(
            graph_exec, node,
            c_void_p(dst_int), c_void_p(src_int),
            c_size_t(count), c_int(kind),
        )
        self.check_error(result, "cudaGraphExecMemcpyNodeSetParams1D")

    def graph_exec_event_record_node_set_event(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        event: CUDAEvent_t,
    ) -> None:
        """Update the event recorded by an event-record node. CUDA 11.4+.

        CPU-only — takes effect on next graph_launch(). Use this to update the
        per-ring-slot IPC event when the ring slot changes between launches.

        Args:
            graph_exec: Executable graph containing the node.
            node:       EventRecordNode handle from graph_get_nodes().
            event:      New CUDAEvent_t to record.

        Raises:
            RuntimeError: If update fails.
        """
        result = self.cudart.cudaGraphExecEventRecordNodeSetEvent(graph_exec, node, event)
        self.check_error(result, "cudaGraphExecEventRecordNodeSetEvent")

    def graph_exec_event_wait_node_set_event(
        self,
        graph_exec: CUDAGraphExec_t,
        node: CUDAGraphNode_t,
        event: CUDAEvent_t,
    ) -> None:
        """Update the event waited on by an event-wait node. CUDA 11.4+.

        Args:
            graph_exec: Executable graph containing the node.
            node:       EventWaitNode handle from graph_get_nodes().
            event:      New CUDAEvent_t to wait on.

        Raises:
            RuntimeError: If update fails.
        """
        result = self.cudart.cudaGraphExecEventWaitNodeSetEvent(graph_exec, node, event)
        self.check_error(result, "cudaGraphExecEventWaitNodeSetEvent")


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
