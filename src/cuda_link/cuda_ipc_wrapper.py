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

import ctypes
import os
from ctypes import POINTER, byref, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p
from typing import Optional

# CUDA handle types - use unsigned 64-bit to prevent overflow on Windows x64
# See: https://github.com/pytorch/pytorch/pull/162920
CUDAEvent_t = c_uint64  # cudaEvent_t is opaque pointer (unsigned 64-bit)
CUDAStream_t = c_uint64  # cudaStream_t is opaque pointer (unsigned 64-bit)


# CUDA IPC Handle structure (128 bytes per NVIDIA spec)
class cudaIpcMemHandle_t(ctypes.Structure):
    """CUDA IPC memory handle structure.

    This opaque handle can be transferred between processes via
    SharedMemory or other IPC mechanisms to enable GPU memory sharing.
    """

    _fields_ = [("internal", ctypes.c_byte * 128)]


# CUDA IPC Event Handle structure (64 bytes per NVIDIA spec)
class cudaIpcEventHandle_t(ctypes.Structure):
    """CUDA IPC event handle structure.

    Used for lightweight cross-process synchronization.
    """

    _fields_ = [("reserved", ctypes.c_byte * 64)]


# CUDA Error codes (subset)
class CUDAError:
    """CUDA runtime error codes."""

    SUCCESS = 0
    INVALID_VALUE = 1
    MEMORY_ALLOCATION = 2
    INVALID_DEVICE_POINTER = 17
    INVALID_DEVICE = 101
    INVALID_CONTEXT = 201  # Common in same-process IPC testing
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

    def __init__(self) -> None:
        """Initialize CUDA runtime library."""
        self.cudart = self._load_cuda_runtime()
        self._setup_function_signatures()

    def _load_cuda_runtime(self) -> ctypes.CDLL:
        """Load CUDA runtime DLL.

        Returns:
            ctypes.CDLL: Loaded CUDA runtime library

        Raises:
            RuntimeError: If CUDA runtime cannot be loaded
        """
        # Try full paths first (for reliability)
        dll_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudart64_12.dll",
        ]

        for dll_path in dll_paths:
            if os.path.exists(dll_path):
                try:
                    return ctypes.CDLL(dll_path)
                except OSError:
                    continue

        # Fallback: try by name (if in system PATH)
        dll_names = ["cudart64_12.dll", "cudart64_11.dll"]
        for name in dll_names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue

        raise RuntimeError(
            "Could not load CUDA runtime. Please ensure CUDA 12.x is installed.\n"
            f"Tried paths: {dll_paths}\n"
            f"Tried names: {dll_names}"
        )

    def _setup_function_signatures(self) -> None:
        """Define function signatures for CUDA runtime functions."""
        # cudaMalloc(void** devPtr, size_t size)
        self.cudart.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        self.cudart.cudaMalloc.restype = c_int

        # cudaFree(void* devPtr)
        self.cudart.cudaFree.argtypes = [c_void_p]
        self.cudart.cudaFree.restype = c_int

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

    def record_event(self, event: CUDAEvent_t, stream: Optional[CUDAStream_t] = None) -> None:
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
        elif result == 600:  # cudaErrorNotReady
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


# Global singleton instance (lazy initialization)
_cuda_runtime: Optional[CUDARuntimeAPI] = None


def get_cuda_runtime() -> CUDARuntimeAPI:
    """Get global CUDA runtime instance (singleton).

    Returns:
        CUDARuntimeAPI: Global CUDA runtime wrapper
    """
    global _cuda_runtime
    if _cuda_runtime is None:
        _cuda_runtime = CUDARuntimeAPI()
    return _cuda_runtime
