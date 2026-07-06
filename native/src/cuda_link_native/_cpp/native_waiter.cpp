// native_waiter.cpp — native notification-wait accelerator for cuda-link's Importer.
//
// Implements the WaitBackend surface (see _backend.py) as a pybind11 module named
// `_native_waiter`. Windows-only. This module NEVER links or loads cudart itself,
// and — as of the 2026-07-04 fix below — never independently resolves it either.
// The resolved cudaEventQuery function-pointer address is handed in explicitly by
// Python via set_cuda_event_query(), sourced from cuda_link's own CUDARuntimeAPI
// (CUDARuntimeAPI.cudart_event_query_fn_ptr(), cuda_ipc_wrapper.py), which already
// knows exactly which cudart instance the rest of the process is using.
//
// EARLIER DESIGN (removed — kept as a warning): this module used to resolve cudart
// itself via GetModuleHandleW(bare name) + GetProcAddress. That is unsafe: a
// process can have more than one same-named cudart DLL loaded from different
// directories (e.g. one pulled in transitively by torch, one loaded by
// cuda_ipc_wrapper.py via an explicit full path), and Windows does not guarantee
// which instance a bare-name GetModuleHandle lookup returns (confirmed against
// Microsoft's own docs: "two DLLs that have the same base file name and extension
// but are found in different directories are not considered to be the same DLL").
// On this project's own dev machine, the bare-name lookup silently resolved a
// *different* cudart instance than the one holding the real event state, so
// cudaEventQuery on a genuinely-pending event spuriously reported "complete."
// Explicit pointer-passing from Python sidesteps the ambiguity entirely — no
// rediscovery, no guessing, correct by construction.
//
// State machine (wait_slot): spin on cudaEventQuery for spin_us, then block until
// the deadline, checking cudaEventQuery, then WaitForSingleObject on the doorbell
// event for a bounded slice. The spin/block loop core is written as a template
// over the polling functions so a test harness can inject fakes without touching
// real CUDA/Win32 state (see native/tests/test_state_machine.py, which drives it
// through the wait_slot_test() pybind entry point below).
//
// FIXED (2026-07-04, torn-frame race): the block phase used to pre-check the SHM
// write_idx before cudaEventQuery and return READY_LATE as soon as write_idx
// advanced past last_write_idx, treating "a newer frame landed" as equivalent to
// "this slot's own event fired." That reuse of wait_for_doorbell's arrival-
// detection pre-check (any new frame is a valid wake reason there) was wrong for
// _wait_for_slot, which needs THIS slot's cudaEventQuery to actually succeed
// before its GPU copy is guaranteed complete. Under the Python Exporter's async
// default (ExportPolicy.export_sync=False), write_idx can advance several frames
// ahead of a still-in-flight D2D copy on this slot, so exiting on write_idx alone
// risked handing the consumer a torn frame. cudaEventQuery() is now the only
// valid "ready" exit from the block phase; READY_LATE (value 2) stays in the
// status enum for numeric stability but the real implementation never emits it
// again. See docs/plans/PLAN-002-native-waiter.md D2 for the full writeup.
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
// cudart dispatch — the function pointer is supplied by Python (see module
// header comment); this module never resolves it independently.
// ---------------------------------------------------------------------------

using CudaEventQueryFn = int (*)(void*);  // cudaError_t cudaEventQuery(cudaEvent_t)

CudaEventQueryFn g_cudaEventQuery = nullptr;

// Status codes mirrored in Python's WaitStatus enum (_backend.py) — keep in sync.
enum WaitStatusCode : int {
    kReadySpin = 0,
    kReadyDoorbell = 1,
    kReadyLate = 2,  // retired 2026-07-04 (torn-frame fix) — never emitted anymore, kept for ABI.
    kTimeout = 3,
    kReadyPoll = 4,  // block phase caught it with no doorbell in play (bare-Sleep polling).
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
// stale the cudaEventQuery re-check can get before it runs again, without
// busy-spinning. Matches the doorbell design's existing bounded lost-wakeup
// window (see _doorbell.py / wait_for_doorbell).
constexpr DWORD kBlockSliceMs = 2;

}  // namespace

// ---------------------------------------------------------------------------
// The spin/block state machine, parameterized over the polling primitives so a
// test harness can inject fakes (see wait_slot_test() below and
// native/tests/test_state_machine.py).
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
    // write_idx_addr / last_write_idx / read_write_idx are retained in the
    // signature for interface stability with the Python WaitBackend Protocol
    // (importer.py still passes them) but are no longer read here — see the
    // torn-frame fix note in this file's header comment for why the write_idx
    // pre-check that used to consume them was removed.
    (void)write_idx_addr;
    (void)last_write_idx;
    (void)read_write_idx;
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
    // cudaEventQuery is the only valid "ready" exit here (see the torn-frame fix
    // note above the enum). The doorbell wake — or, absent a doorbell, the
    // bounded Sleep — is purely a hint to re-check sooner; a missed/late signal
    // costs at most one kBlockSliceMs slice before the next event_query, never a
    // correctness issue.
    while (true) {
        const double t = now_us();
        if (t >= deadline) {
            return {kTimeout, t - start, 0};
        }

        if (event_ptr != 0 && event_query(event_ptr)) {
            // doorbell_handle != 0 means a doorbell was genuinely in play for this
            // wait — kReadyDoorbell, even if this particular check happened right
            // after a bounded slice elapsed rather than an exact wake (the two are
            // not worth distinguishing further: both mean "the block phase, not
            // the spin phase, caught it"). doorbell_handle == 0 means there was
            // never a doorbell to wake on at all — kReadyPoll, so telemetry never
            // implies a wake that couldn't have happened.
            const bool had_doorbell = doorbell_handle != 0;
            return {had_doorbell ? kReadyDoorbell : kReadyPoll, now_us() - start, had_doorbell ? 2 : 3};
        }

        if (doorbell_handle != 0) {
            wait_doorbell(doorbell_handle, kBlockSliceMs);
            // Whether this returned true (genuine wake) or false (slice elapsed),
            // loop back and re-check the event above -- a wake is only a hint,
            // never a substitute for cudaEventQuery.
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

// ---------------------------------------------------------------------------
// Test-only entry point: drives the REAL wait_slot_impl template with
// Python-injected fake polling functions, so native/tests/test_state_machine.py
// can exercise the actual state machine (including the torn-frame regression
// test) without any real CUDA/Win32 state. Not used by the production wait_slot()
// path above, and — unlike it — does NOT release the GIL: calling back into
// Python callables requires holding it throughout.
// ---------------------------------------------------------------------------

std::tuple<int, double, int> wait_slot_test(
    py::function event_query,       // (event_ptr: int) -> bool
    py::function read_write_idx,    // (write_idx_addr: int) -> int
    py::function wait_doorbell,     // (doorbell_handle: int, slice_ms: int) -> bool
    py::function now_us,            // () -> float, microseconds, caller-controlled clock
    std::uintptr_t event_ptr,
    std::uintptr_t doorbell_handle,
    std::uintptr_t write_idx_addr,
    std::uint32_t last_write_idx,
    int spin_us,
    int timeout_ms) {
    auto event_query_wrap = [&](std::uintptr_t ptr) -> bool { return event_query(ptr).cast<bool>(); };
    auto read_write_idx_wrap = [&](std::uintptr_t addr) -> std::uint32_t {
        return read_write_idx(addr).cast<std::uint32_t>();
    };
    auto wait_doorbell_wrap = [&](std::uintptr_t handle, DWORD slice_ms) -> bool {
        return wait_doorbell(handle, static_cast<int>(slice_ms)).cast<bool>();
    };
    auto now_us_wrap = [&]() -> double { return now_us().cast<double>(); };

    return wait_slot_impl(
        event_query_wrap, read_write_idx_wrap, wait_doorbell_wrap, now_us_wrap,
        event_ptr, doorbell_handle, write_idx_addr, last_write_idx, spin_us, timeout_ms);
}

bool cudart_resolved() {
    return g_cudaEventQuery != nullptr;
}

void set_cuda_event_query(std::uintptr_t fn_ptr) {
    // fn_ptr is the exact address of the cudaEventQuery symbol Python already
    // resolved (CUDARuntimeAPI.cudart_event_query_fn_ptr()) — guaranteed to
    // belong to the same cudart instance the rest of the process is using, no
    // rediscovery needed. A null/zero pointer explicitly un-resolves (returns
    // this module to "not usable"; the Python loader treats that as a signal
    // to fall back to the python wait path).
    g_cudaEventQuery = fn_ptr != 0 ? reinterpret_cast<CudaEventQueryFn>(fn_ptr) : nullptr;
}

}  // namespace

PYBIND11_MODULE(_native_waiter, m) {
    m.doc() = "Native notification-wait accelerator for cuda-link's Importer wait path.";

    m.def("set_cuda_event_query", &set_cuda_event_query,
          "Register the cudaEventQuery function-pointer address resolved by Python's "
          "CUDARuntimeAPI. Must be called once before wait_slot() is used.");
    m.def("cudart_resolved", &cudart_resolved,
          "Return True if set_cuda_event_query() has been called with a non-null pointer.");

    // GIL released for the whole call: all args are passed by value (ints) before
    // release, and the return value is a plain std::tuple<int,double,int> that
    // pybind11 converts to a Python tuple via its automatic caster AFTER the
    // call_guard's destructor re-acquires the GIL — so this is safe (contrast
    // with cuda_link_spout's `receive()`, which manually builds a py::tuple
    // inside the function body and must therefore keep the GIL held throughout).
    m.def("wait_slot", &wait_slot, py::call_guard<py::gil_scoped_release>(),
          "Wait for the per-slot CUDA event or the doorbell, whichever fires first. "
          "Returns (status_int, waited_us, method) where status_int is "
          "0=READY_SPIN 1=READY_DOORBELL 2=READY_LATE(retired, never emitted) "
          "3=TIMEOUT 4=READY_POLL.");

    // Test-only entry point (see the docstring on wait_slot_test's definition
    // above). GIL held throughout -- not used by the production wait_slot() path.
    // Named py::arg()s so native/tests/test_state_machine.py can call it with
    // keyword arguments.
    m.def("wait_slot_test", &wait_slot_test, py::arg("event_query"), py::arg("read_write_idx"),
          py::arg("wait_doorbell"), py::arg("now_us"), py::arg("event_ptr"), py::arg("doorbell_handle"),
          py::arg("write_idx_addr"), py::arg("last_write_idx"), py::arg("spin_us"), py::arg("timeout_ms"),
          "TEST ONLY: drive the real wait_slot_impl state machine with injected "
          "Python fake polling functions (event_query, read_write_idx, "
          "wait_doorbell, now_us). Same return contract as wait_slot(). Used by "
          "native/tests/test_state_machine.py; not part of the production API.");
}
