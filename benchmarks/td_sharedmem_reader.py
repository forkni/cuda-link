"""TouchDesigner Shared Mem Out TOP Reader.

This module provides a Python implementation of TouchDesigner's UT_SharedMem protocol
for reading from Shared Mem Out TOP operators. It uses ctypes Win32 API to open file
mappings and named mutexes, matching the C++ implementation from
TD's Samples/SharedMem/ directory.

Key differences from CUDA IPC:
- CPU SharedMemory (not GPU) - D2H copy already done by TD
- Mutex-based synchronization (not CUDA events)
- TOP_SharedMemHeader protocol (48-byte header + pixel data)
- Returns numpy arrays (not GPU tensors)

Typical latency: ~500-1500μs per frame for 1080p (vs ~2μs for CUDA IPC)
"""

from __future__ import annotations

import ctypes
import struct
from ctypes import wintypes

import numpy as np

# Win32 API constants
FILE_MAP_ALL_ACCESS = 0xF001F
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_ABANDONED = 0x00000080
INFINITE = 0xFFFFFFFF

# TOP SharedMemory protocol constants
TOP_SHM_MAGIC_NUMBER = 0xD95EF835
TOP_SHM_VERSION_NUMBER = 1
TOP_HEADER_SIZE = 48  # sizeof(TOP_SharedMemHeader) with C++ alignment

# UT_SharedMem naming constants
UT_SHM_INFO_DECORATION = "4jhd783h"
UT_SHM_INFO_SIZE = 80  # sizeof(UT_SharedMemInfo) approximate
UT_SHM_MAX_POST_FIX_SIZE = 32

# Load Win32 APIs
kernel32 = ctypes.windll.kernel32
kernel32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.OpenFileMappingW.restype = wintypes.HANDLE

kernel32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
kernel32.MapViewOfFile.restype = wintypes.LPVOID

kernel32.UnmapViewOfFile.argtypes = [wintypes.LPVOID]
kernel32.UnmapViewOfFile.restype = wintypes.BOOL

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
kernel32.CreateMutexW.restype = wintypes.HANDLE

kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD

kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
kernel32.ReleaseMutex.restype = wintypes.BOOL


# TOP_PixelFormat enum (Vulkan-style from TOP_SharedMemHeader.h)
PIXEL_FORMAT_MAP = {
    # 8-bit fixed
    9: (np.uint8, 1, "R8_UNORM"),
    16: (np.uint8, 2, "R8G8_UNORM"),
    37: (np.uint8, 4, "R8G8B8A8_UNORM"),
    43: (np.uint8, 4, "R8G8B8A8_SRGB"),
    44: (np.uint8, 4, "B8G8R8A8_UNORM"),
    50: (np.uint8, 4, "B8G8R8A8_SRGB"),
    # 16-bit fixed
    70: (np.uint16, 1, "R16_UNORM"),
    77: (np.uint16, 2, "R16G16_UNORM"),
    91: (np.uint16, 4, "R16G16B16A16_UNORM"),
    # 16-bit float
    76: (np.float16, 1, "R16_SFLOAT"),
    83: (np.float16, 2, "R16G16_SFLOAT"),
    97: (np.float16, 4, "R16G16B16A16_SFLOAT"),
    # 32-bit float
    100: (np.float32, 1, "R32_SFLOAT"),
    103: (np.float32, 2, "R32G32_SFLOAT"),
    109: (np.float32, 4, "R32G32B32A32_SFLOAT"),
    # Swizzled formats (alpha-only, mono-alpha)
    0xF0001: (np.uint8, 1, "A8_UNORM"),
    0xF0002: (np.uint16, 1, "A16_UNORM"),
    0xF0003: (np.float16, 1, "A16_SFLOAT"),
    0xF0004: (np.float32, 1, "A32_SFLOAT"),
    0xF0005: (np.uint8, 2, "R8A8_UNORM"),
    0xF0006: (np.uint16, 2, "R16A16_UNORM"),
    0xF0007: (np.float16, 2, "R16A16_SFLOAT"),
    0xF0008: (np.float32, 2, "R32A32_SFLOAT"),
}


class TDSharedMemReader:
    """Reader for TouchDesigner Shared Mem Out TOP using UT_SharedMem protocol.

    This class implements the receiver side of TD's UT_SharedMem protocol, including:
    - Opening file mappings ("TouchSHM" + name + postfix)
    - Named mutex synchronization (name + "Mutex")
    - TOP_SharedMemHeader parsing (48-byte header)
    - Pixel format decoding (TOP_PixelFormat enum)

    Example:
        reader = TDSharedMemReader("benchmark_shm", debug=True)
        if reader.connect():
            frame = reader.read_frame()
            if frame is not None:
                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        reader.cleanup()
    """

    def __init__(self, name: str, debug: bool = False):
        """Initialize TD shared memory reader.

        Args:
            name: SharedMemory name matching TD's Shared Mem Out TOP "name" parameter
            debug: Enable verbose debug logging
        """
        self.name = name
        self.debug = debug

        # Handles
        self.mutex_handle: wintypes.HANDLE | None = None
        self.info_mapping: wintypes.HANDLE | None = None
        self.info_view: wintypes.LPVOID | None = None
        self.data_mapping: wintypes.HANDLE | None = None
        self.data_view: wintypes.LPVOID | None = None

        # Metadata (populated after connect())
        self.width: int = 0
        self.height: int = 0
        self.pixel_format: int = 0
        self.pixel_format_name: str = ""
        self.data_size: int = 0
        self.dtype: np.dtype | None = None
        self.num_channels: int = 0
        self.shape: tuple[int, int, int] | None = None

        # Frame tracking for new-frame detection
        self._last_frame_counter: int = 0
        self._name_postfix: str = ""  # UT_SharedMem resize postfix

    def _log(self, msg: str) -> None:
        """Log debug message if debug is enabled."""
        if self.debug:
            print(f"[TDSharedMemReader] {msg}")

    def connect(self) -> bool:
        """Open shared memory segments, mutex, and read initial metadata.

        Returns:
            True if connection succeeded, False otherwise
        """
        try:
            # Step 1: Open mutex (name + "Mutex")
            mutex_name = f"TouchSHM{self.name}Mutex"
            self.mutex_handle = kernel32.CreateMutexW(None, False, mutex_name)
            if not self.mutex_handle:
                print(f"[TDSharedMemReader] ERROR: Failed to open mutex: {mutex_name}")
                return False
            self._log(f"Opened mutex: {mutex_name}")

            # Step 2: Try opening info segment (double "TouchSHM" prefix per UT_SharedMem.cpp createInfo())
            # Info segment contains namePostFix for resize support
            info_name = f"TouchSHMTouchSHM{self.name}{UT_SHM_INFO_DECORATION}"
            self.info_mapping = kernel32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, info_name)
            if self.info_mapping:
                self.info_view = kernel32.MapViewOfFile(self.info_mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0)
                if self.info_view:
                    # Read namePostFix (char[32] at offset 8)
                    postfix_bytes = ctypes.string_at(
                        ctypes.cast(self.info_view, ctypes.c_void_p).value + 8, UT_SHM_MAX_POST_FIX_SIZE
                    )
                    self._name_postfix = postfix_bytes.decode("utf-8", errors="ignore").rstrip("\x00")
                    self._log(f"Info segment found, namePostFix: '{self._name_postfix}'")
            else:
                self._log("Info segment not found (resize not supported)")

            # Step 3: Open data segment (name + postfix)
            data_name = f"TouchSHM{self.name}{self._name_postfix}"
            self.data_mapping = kernel32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, data_name)
            if not self.data_mapping:
                error_code = ctypes.GetLastError()
                print(f"[TDSharedMemReader] ERROR: Failed to open data mapping: {data_name}")
                print(f"[TDSharedMemReader]        Win32 error code: {error_code}")
                print("[TDSharedMemReader]        Ensure TouchDesigner is running with Shared Mem Out TOP configured")
                return False

            self.data_view = kernel32.MapViewOfFile(self.data_mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0)
            if not self.data_view:
                error_code = ctypes.GetLastError()
                print("[TDSharedMemReader] ERROR: Failed to map data view")
                print(f"[TDSharedMemReader]        Win32 error code: {error_code}")
                return False

            self._log(f"Opened data mapping: {data_name}")

            # Step 4: Read header metadata (with mutex lock)
            if not self._read_header():
                return False

            self._log(
                f"Connected: {self.width}x{self.height}, format={self.pixel_format_name}, dtype={self.dtype}, channels={self.num_channels}"
            )
            return True

        except (OSError, RuntimeError, struct.error) as e:
            print(f"[TDSharedMemReader] ERROR: Connection failed with exception: {e}")
            self.cleanup()
            return False

    def _read_header(self) -> bool:
        """Read and parse TOP_SharedMemHeader (must be called with mutex locked).

        Returns:
            True if header valid, False otherwise
        """
        if not self._lock_mutex(timeout_ms=5000):
            return False

        try:
            # Read 48-byte header
            # Format: <Ii ii ff I 4x q i 4x (total 48 bytes)
            # Note: First field is unsigned 'I' because magic has bit 31 set (0xD95EF835)
            header_bytes = ctypes.string_at(self.data_view, TOP_HEADER_SIZE)
            magic, version, width, height, aspectx, aspecty, pixel_format, data_size, data_offset = struct.unpack(
                "<Ii ii ff I 4x q i 4x", header_bytes
            )

            # Validate magic and version (print unconditionally - these are real protocol issues)
            if magic != TOP_SHM_MAGIC_NUMBER:
                print(
                    f"[TDSharedMemReader] ERROR: Invalid magic number: 0x{magic:08x} (expected 0x{TOP_SHM_MAGIC_NUMBER:08x})"
                )
                return False
            if version != TOP_SHM_VERSION_NUMBER:
                print(f"[TDSharedMemReader] ERROR: Unsupported version: {version} (expected {TOP_SHM_VERSION_NUMBER})")
                return False

            # Store metadata
            self.width = width
            self.height = height
            self.pixel_format = pixel_format
            self.data_size = data_size

            # Decode pixel format
            if pixel_format not in PIXEL_FORMAT_MAP:
                self._log(f"Unknown pixel format: {pixel_format}")
                return False

            self.dtype, self.num_channels, self.pixel_format_name = PIXEL_FORMAT_MAP[pixel_format]
            self.shape = (height, width, self.num_channels) if self.num_channels > 1 else (height, width)

            # Pre-allocate frame buffer for zero-copy reads
            self._frame_buffer = np.empty(self.data_size, dtype=np.uint8)

            return True

        finally:
            self._unlock_mutex()

    def _lock_mutex(self, timeout_ms: int = 5000) -> bool:
        """Acquire mutex lock.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            True if lock acquired, False on timeout
        """
        if not self.mutex_handle:
            return False

        result = kernel32.WaitForSingleObject(self.mutex_handle, timeout_ms)
        if result in (WAIT_OBJECT_0, WAIT_ABANDONED):
            return True
        elif result == WAIT_TIMEOUT:
            if self.debug:
                self._log(f"Mutex lock timeout after {timeout_ms}ms")
            return False
        else:
            if self.debug:
                self._log(f"Mutex lock failed with result: {result}")
            return False

    def _unlock_mutex(self) -> bool:
        """Release mutex lock.

        Returns:
            True if unlock succeeded, False otherwise
        """
        if not self.mutex_handle:
            return False
        return bool(kernel32.ReleaseMutex(self.mutex_handle))

    def read_frame(self, timestamp_shm_name: str | None = None) -> np.ndarray | None:
        """Read a frame from shared memory.

        This method locks the mutex, reads the header + pixel data, and returns
        a numpy array. If timestamp_shm_name is provided, it will check the frame
        counter to detect new frames (returns None if same frame as last read).

        Args:
            timestamp_shm_name: Optional timestamp SharedMemory name for new-frame detection

        Returns:
            Numpy array with shape (height, width, channels) or None if no new frame
        """
        if not self.data_view:
            return None

        # Check for new frame if timestamp channel provided
        if timestamp_shm_name:
            try:
                from multiprocessing.shared_memory import SharedMemory

                ts_shm = SharedMemory(name=timestamp_shm_name)
                frame_counter = struct.unpack_from("<I", ts_shm.buf, 0)[0]
                ts_shm.close()

                if frame_counter == self._last_frame_counter:
                    return None  # Same frame as last read
                self._last_frame_counter = frame_counter
            except (OSError, struct.error) as e:
                if self.debug:
                    self._log(f"Timestamp check failed: {e}")

        # Lock mutex for reading
        if not self._lock_mutex(timeout_ms=5000):
            return None

        try:
            # Read pixel data starting at dataOffset (48 bytes)
            # Use pre-allocated buffer with memmove instead of string_at (avoids 33MB Python bytes alloc)
            pixel_data_ptr = ctypes.cast(self.data_view, ctypes.c_void_p).value + TOP_HEADER_SIZE
            ctypes.memmove(self._frame_buffer.ctypes.data, pixel_data_ptr, self.data_size)

            # Convert to target dtype and reshape (zero-copy view)
            # Safe to return view since buffer persists until next read_frame() call
            frame = self._frame_buffer.view(self.dtype).reshape(self.shape)

            return frame

        finally:
            self._unlock_mutex()

    def cleanup(self) -> None:
        """Close all handles and unmap memory."""
        if self.data_view:
            kernel32.UnmapViewOfFile(self.data_view)
            self.data_view = None

        if self.data_mapping:
            kernel32.CloseHandle(self.data_mapping)
            self.data_mapping = None

        if self.info_view:
            kernel32.UnmapViewOfFile(self.info_view)
            self.info_view = None

        if self.info_mapping:
            kernel32.CloseHandle(self.info_mapping)
            self.info_mapping = None

        if self.mutex_handle:
            kernel32.CloseHandle(self.mutex_handle)
            self.mutex_handle = None

        if self.debug:
            self._log("Cleanup complete")
