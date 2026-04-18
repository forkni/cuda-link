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
import os
from ctypes import POINTER, byref, c_float, c_int, c_size_t, c_uint, c_uint64, c_void_p

# CUDA handle types - use unsigned 64-bit to prevent overflow on Windows x64
# See: https://github.com/pytorch/pytorch/pull/162920
CUDAEvent_t = c_uint64  # cudaEvent_t is opaque pointer (unsigned 64-bit)
CUDAStream_t = c_uint64  # cudaStream_t is opaque pointer (unsigned 64-bit)


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
            import logging as _logging

            _logging.getLogger(__name__).debug("Loaded CUDA runtime: %s", buf.value)
        except Exception:  # noqa: BLE001
            pass

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

    def malloc_host(self, size: int) -> c_void_p:
        """Allocate pinned (page-locked) host memory via cudaMallocHost.

        Pinned memory enables direct DMA for D2H transfers, eliminating the
        CUDA driver's internal staging copy that pageable memory requires.

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
