// native_waiter.cpp — native notification-wait accelerator for cuda-link's Importer.
//
// Implements the WaitBackend surface (see _backend.py) as a pybind11 module named
// `_native_waiter`. Windows-only. Unlike cuda-link-spout, this module NEVER links
// or loads cudart itself: at init it resolves cudaEventQuery via GetModuleHandleW
// (never LoadLibraryW) + GetProcAddress from whatever cudart the host process has
// already loaded (mirroring cuda_ipc_wrapper.py's probe order: 13 -> 12 -> 11 ->
// 110, so a torch build against CUDA 13 shares its already-resident runtime). This
// keeps the double-context risk at zero and needs no CUDA Toolkit at build time.
//
// State machine (wait_slot): spin on cudaEventQuery for spin_us, then block until
// the deadline, checking (in order) the write_idx lost-wakeup pre-check, then
// cudaEventQuery, then WaitForSingleObject on the doorbell event for a bounded
// slice. The spin/block loop core is written as a template over the polling
// functions so a test harness can inject fakes without touching real CUDA/Win32
// state (see test_state_machine in native/tests).
//
// Build: see native/CMakeLists.txt. This file is intentionally not compiled in CI
// on Linux; the pure-Python layer is tested via FakeWaitBackend. Validate on a
// Windows box before release.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <tuple>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// cudart resolution — GetModuleHandleW only, never LoadLibraryW.
// ---------------------------------------------------------------------------

using CudaEventQueryFn = int (*)(void*);  // cudaError_t cudaEventQuery(cudaEvent_t)

CudaEventQueryFn g_cudaEventQuery = nullptr;

CudaEventQueryFn resolve_cuda_event_query() {
    static const wchar_t* kCudartNames[] = {
        L"cudart64_13.dll",
        L"cudart64_12.dll",
        L"cudart64_11.dll",
        L"cudart64_110.dll",
    };
    for (const wchar_t* name : kCudartNames) {
        HMODULE h = GetModuleHandleW(name);
        if (h != nullptr) {
            auto fn = reinterpret_cast<CudaEventQueryFn>(GetProcAddress(h, "cudaEventQuery"));
            if (fn != nullptr) {
                return fn;
            }
        }
    }
    return nullptr;
}

// Status codes mirrored in Python's WaitStatus enum (_backend.py) — keep in sync.
enum WaitStatusCode : int {
    kReadySpin = 0,
    kReadyDoorbell = 1,
    kReadyLate = 2,
    kTimeout = 3,
};

double qpc_now_us(double freq_inv_us) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart) * freq_inv_us;
}

double qpc_freq_inv_us() {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return 1'000'000.0 / static_cast<double>(freq.QuadPart);
}

// Small bounded slice for the block-phase WaitForSingleObject call: bounds how
// stale the write_idx/cudaEventQuery pre-checks can get before re-running them,
// without busy-spinning. Matches the doorbell design's existing bounded
// lost-wakeup window (see _doorbell.py / wait_for_doorbell).
constexpr DWORD kBlockSliceMs = 2;

}  // namespace

// ---------------------------------------------------------------------------
// The spin/block state machine, parameterized over the polling primitives so a
// test harness can inject fakes (see native/tests for the injected-fn variant).
// ---------------------------------------------------------------------------

template <typename EventQueryFn, typename ReadWriteIdxFn, typename WaitDoorbellFn, typename NowUsFn>
std::tuple<int, double, int> wait_slot_impl(
    EventQueryFn event_query,          // (event_ptr) -> bool ready
    ReadWriteIdxFn read_write_idx,     // (write_idx_addr) -> uint32_t
    WaitDoorbellFn wait_doorbell,      // (doorbell_handle, slice_ms) -> bool signaled
    NowUsFn now_us,                    // () -> double microseconds, monotonic
    std::uintptr_t event_ptr,
    std::uintptr_t doorbell_handle,
    std::uintptr_t write_idx_addr,
    std::uint32_t last_write_idx,
    int spin_us,
    int timeout_ms) {
    const double start = now_us();
    const double deadline = start + static_cast<double>(timeout_ms) * 1000.0;
    const double spin_deadline = (spin_us > 0) ? (start + static_cast<double>(spin_us)) : start;

    // --- spin phase ---------------------------------------------------------
    if (spin_us > 0) {
        double t;
        while ((t = now_us()) < spin_deadline && t < deadline) {
            if (event_ptr != 0 && event_query(event_ptr)) {
                return {kReadySpin, t - start, 0};
            }
        }
    }

    // --- block phase ---------------------------------------------------------
    while (true) {
        const double t = now_us();
        if (t >= deadline) {
            return {kTimeout, t - start, 0};
        }

        // Lost-wakeup pre-check (mirrors importer.py's wait_for_doorbell pre-check):
        // a frame may have already landed since last_write_idx was captured, even
        // if the doorbell signal itself was missed.
        if (write_idx_addr != 0) {
            const std::uint32_t cur = read_write_idx(write_idx_addr);
            if (cur != 0 && cur != last_write_idx) {
                return {kReadyLate, now_us() - start, 1};
            }
        }

        if (event_ptr != 0 && event_query(event_ptr)) {
            return {kReadyDoorbell, now_us() - start, 2};
        }

        if (doorbell_handle != 0) {
            if (wait_doorbell(doorbell_handle, kBlockSliceMs)) {
                // Woken by the doorbell; loop back around to re-check write_idx /
                // event above rather than trusting the wake blindly.
                continue;
            }
        } else {
            // No doorbell available — bounded sleep so we don't busy-spin forever
            // (this degrades to roughly the block-phase floor the Python fallback
            // already has; callers should prefer python/auto fallback when the
            // doorbell can't be opened at all).
            Sleep(kBlockSliceMs);
        }
    }
}

// ---------------------------------------------------------------------------
// Real polling primitives wired to Win32 / the resolved cudart pointer.
// ---------------------------------------------------------------------------

namespace {

bool real_event_query(std::uintptr_t event_ptr) {
    if (g_cudaEventQuery == nullptr) {
        return false;
    }
    return g_cudaEventQuery(reinterpret_cast<void*>(event_ptr)) == 0;  // cudaSuccess == 0
}

std::uint32_t real_read_write_idx(std::uintptr_t addr) {
    // volatile read — the SHM header's write_idx is updated by another process.
    return *reinterpret_cast<volatile std::uint32_t*>(addr);
}

bool real_wait_doorbell(std::uintptr_t handle, DWORD slice_ms) {
    HANDLE h = reinterpret_cast<HANDLE>(handle);
    return WaitForSingleObject(h, slice_ms) == WAIT_OBJECT_0;
}

double real_now_us() {
    static const double kFreqInvUs = qpc_freq_inv_us();
    return qpc_now_us(kFreqInvUs);
}

std::tuple<int, double, int> wait_slot(
    std::uintptr_t event_ptr,
    std::uintptr_t doorbell_handle,
    std::uintptr_t write_idx_addr,
    std::uint32_t last_write_idx,
    int spin_us,
    int timeout_ms) {
    return wait_slot_impl(
        real_event_query, real_read_write_idx, real_wait_doorbell, real_now_us,
        event_ptr, doorbell_handle, write_idx_addr, last_write_idx, spin_us, timeout_ms);
}

bool cudart_resolved() {
    return g_cudaEventQuery != nullptr;
}

}  // namespace

PYBIND11_MODULE(_native_waiter, m) {
    m.doc() = "Native notification-wait accelerator for cuda-link's Importer wait path.";

    g_cudaEventQuery = resolve_cuda_event_query();

    m.def("cudart_resolved", &cudart_resolved,
          "Return True if a loaded cudart's cudaEventQuery was resolved at module init.");

    // GIL released for the whole call: all args are passed by value (ints) before
    // release, and the return value is a plain std::tuple<int,double,int> that
    // pybind11 converts to a Python tuple via its automatic caster AFTER the
    // call_guard's destructor re-acquires the GIL — so this is safe (contrast
    // with cuda_link_spout's `receive()`, which manually builds a py::tuple
    // inside the function body and must therefore keep the GIL held throughout).
    m.def("wait_slot", &wait_slot, py::call_guard<py::gil_scoped_release>(),
          "Wait for the per-slot CUDA event or the doorbell, whichever fires first. "
          "Returns (status_int, waited_us, method) where status_int is "
          "0=READY_SPIN 1=READY_DOORBELL 2=READY_LATE 3=TIMEOUT.");
}
