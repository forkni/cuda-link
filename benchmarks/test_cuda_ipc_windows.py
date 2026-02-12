"""
CUDA IPC Windows Verification Test
Tests if Windows CUDA IPC overhead is acceptable for high-frequency use.

Target: < 100μs per IPC operation (currently SharedMemory ~1.5ms)
If successful: Phase 2 (vLLM-style ctypes IPC) becomes viable
If failed: Stick with Phase 1 only (timing + pinned memory)
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
import traceback
from ctypes import POINTER, byref, c_int, c_uint, c_void_p


# CUDA IPC Handle structure (128 bytes per NVIDIA docs)
class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def load_cuda_runtime() -> ctypes.CDLL:
    """Load CUDA runtime library for Windows."""
    # Try full paths to CUDA runtime DLLs
    dll_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cudart64_12.dll",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cudart64_12.dll",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\cudart64_12.dll",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudart64_11.dll",
    ]

    # Try full paths first
    for dll_path in dll_paths:
        if os.path.exists(dll_path):
            try:
                cudart = ctypes.CDLL(dll_path)
                print(f"[OK] Loaded {os.path.basename(dll_path)} from {os.path.dirname(dll_path)}")
                return cudart
            except OSError as e:
                print(f"[WARN] Failed to load {dll_path}: {e}")
                continue

    # Fallback: try by name (if in system PATH)
    dll_names = ["cudart64_12.dll", "cudart64_11.dll"]
    for name in dll_names:
        try:
            cudart = ctypes.CDLL(name)
            print(f"[OK] Loaded {name} from system PATH")
            return cudart
        except OSError:
            continue

    raise RuntimeError(f"Could not load CUDA runtime. Tried paths: {dll_paths}")


def setup_function_signatures(cudart: ctypes.CDLL) -> None:
    """Define CUDA function signatures for ctypes."""
    # cudaMalloc(void** devPtr, size_t size)
    cudart.cudaMalloc.argtypes = [POINTER(c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = c_int

    # cudaFree(void* devPtr)
    cudart.cudaFree.argtypes = [c_void_p]
    cudart.cudaFree.restype = c_int

    # cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)
    cudart.cudaIpcGetMemHandle.argtypes = [POINTER(cudaIpcMemHandle_t), c_void_p]
    cudart.cudaIpcGetMemHandle.restype = c_int

    # cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
    cudart.cudaIpcOpenMemHandle.argtypes = [
        POINTER(c_void_p),
        cudaIpcMemHandle_t,
        c_uint,
    ]
    cudart.cudaIpcOpenMemHandle.restype = c_int

    # cudaIpcCloseMemHandle(void* devPtr)
    cudart.cudaIpcCloseMemHandle.argtypes = [c_void_p]
    cudart.cudaIpcCloseMemHandle.restype = c_int

    # cudaDeviceSynchronize()
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = c_int

    # cudaGetLastError()
    cudart.cudaGetLastError.argtypes = []
    cudart.cudaGetLastError.restype = c_int

    # cudaGetErrorString(cudaError_t error)
    cudart.cudaGetErrorString.argtypes = [c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p


def check_cuda_error(cudart: ctypes.CDLL, result: int, operation: str) -> bool:
    """Check CUDA error and print message if failed."""
    if result != 0:
        error_str = cudart.cudaGetErrorString(result).decode("utf-8")
        print(f"[FAIL] {operation} failed: {error_str} (error {result})")
        return False
    return True


def test_cuda_ipc(cudart: ctypes.CDLL, num_iterations: int = 100) -> dict[str, float] | None:
    """
    Test CUDA IPC handle creation and opening overhead.

    Returns:
        dict with timing results in microseconds
    """
    # Allocate 4MB GPU memory (512x512x4 float32 = typical frame size)
    buffer_size = 512 * 512 * 4 * 4  # 4MB
    dev_ptr = c_void_p()

    print(f"\n--- CUDA IPC Test ({num_iterations} iterations) ---")
    print(f"Buffer size: {buffer_size / 1024 / 1024:.1f} MB")

    # Step 1: Allocate GPU memory
    result = cudart.cudaMalloc(byref(dev_ptr), buffer_size)
    if not check_cuda_error(cudart, result, "cudaMalloc"):
        return None
    print(f"[OK] Allocated GPU memory at 0x{dev_ptr.value:016x}")

    # Step 2: Create IPC handle (measure time)
    handle = cudaIpcMemHandle_t()

    # Warmup
    for _ in range(10):
        cudart.cudaIpcGetMemHandle(byref(handle), dev_ptr)

    # Benchmark cudaIpcGetMemHandle
    start = time.perf_counter_ns()
    for _ in range(num_iterations):
        result = cudart.cudaIpcGetMemHandle(byref(handle), dev_ptr)
    end = time.perf_counter_ns()

    if not check_cuda_error(cudart, result, "cudaIpcGetMemHandle"):
        cudart.cudaFree(dev_ptr)
        return None

    get_handle_us = (end - start) / num_iterations / 1000
    print(f"[OK] cudaIpcGetMemHandle: {get_handle_us:.1f} us/call")

    # Step 3: Open IPC handle (simulates receiver process)
    # Note: In same process, this may behave differently than cross-process
    opened_ptr = c_void_p()
    cuda_ipc_mem_lazy_enable_peer_access = 1  # Flag

    # Warmup
    for _ in range(5):
        result = cudart.cudaIpcOpenMemHandle(byref(opened_ptr), handle, cuda_ipc_mem_lazy_enable_peer_access)
        if result == 0:
            cudart.cudaIpcCloseMemHandle(opened_ptr)

    # Benchmark cudaIpcOpenMemHandle
    open_times = []
    close_times = []

    for _ in range(num_iterations):
        start = time.perf_counter_ns()
        result = cudart.cudaIpcOpenMemHandle(byref(opened_ptr), handle, cuda_ipc_mem_lazy_enable_peer_access)
        end = time.perf_counter_ns()

        if result != 0:
            # Some errors expected in same-process testing
            error_str = cudart.cudaGetErrorString(result).decode("utf-8")
            if "already mapped" in error_str.lower() or result == 1:
                print(f"[WARNING] Same-process IPC limitation detected: {error_str}")
                print("          (Cross-process IPC may still work)")
                break
            else:
                check_cuda_error(cudart, result, "cudaIpcOpenMemHandle")
                break

        open_times.append((end - start) / 1000)

        start = time.perf_counter_ns()
        cudart.cudaIpcCloseMemHandle(opened_ptr)
        end = time.perf_counter_ns()
        close_times.append((end - start) / 1000)

    open_handle_us = sum(open_times) / len(open_times) if open_times else -1
    close_handle_us = sum(close_times) / len(close_times) if close_times else -1

    if open_handle_us > 0:
        print(f"[OK] cudaIpcOpenMemHandle: {open_handle_us:.1f} us/call")
        print(f"[OK] cudaIpcCloseMemHandle: {close_handle_us:.1f} us/call")

    # Cleanup
    cudart.cudaFree(dev_ptr)
    print("[OK] Freed GPU memory")

    # Results
    total_ipc_us = get_handle_us + (open_handle_us if open_handle_us > 0 else 0)

    return {
        "get_handle_us": get_handle_us,
        "open_handle_us": open_handle_us,
        "close_handle_us": close_handle_us,
        "total_ipc_us": total_ipc_us,
        "iterations": num_iterations,
    }


def main() -> int:
    print("=" * 60)
    print("CUDA IPC Windows Verification Test")
    print("=" * 60)

    try:
        cudart = load_cuda_runtime()
        setup_function_signatures(cudart)

        # Sync to ensure GPU is ready
        cudart.cudaDeviceSynchronize()

        results = test_cuda_ipc(cudart, num_iterations=100)

        if results:
            print("\n" + "=" * 60)
            print("RESULTS SUMMARY")
            print("=" * 60)
            print(f"  cudaIpcGetMemHandle:  {results['get_handle_us']:.1f} us")
            if results["open_handle_us"] > 0:
                print(f"  cudaIpcOpenMemHandle: {results['open_handle_us']:.1f} us")
                print(f"  Total IPC overhead:   {results['total_ipc_us']:.1f} us")
            print()

            # Decision
            threshold_us = 100
            current_sharedmem_us = 1500  # ~1.5ms

            if results["total_ipc_us"] < threshold_us:
                print("[SUCCESS] VERDICT: CUDA IPC is VIABLE")
                print(f"          IPC overhead ({results['total_ipc_us']:.0f}us) < threshold ({threshold_us}us)")
                print(
                    f"          Potential savings: ~{current_sharedmem_us - results['total_ipc_us']:.0f}us vs SharedMemory"
                )
                print("          -> Recommend proceeding with Phase 2 (vLLM-style ctypes IPC)")
            else:
                print("[WARNING] VERDICT: CUDA IPC overhead is HIGH")
                print(f"          IPC overhead ({results['total_ipc_us']:.0f}us) > threshold ({threshold_us}us)")
                print("          Windows performance penalty confirmed")
                print("          -> Recommend Phase 1 only (timing + pinned memory)")

    except (RuntimeError, OSError) as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
