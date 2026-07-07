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
// the deadline, re-checking cudaEventQuery between bounded naps on a per-thread
// high-resolution waitable timer. The spin/block loop core is written as a template
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
// FIXED (2026-07-06, block-phase timer quantization): the block phase used to
// nap by waiting on the shared doorbell event in bounded slices
// (WaitForSingleObject(doorbell, kBlockSliceMs=2)) — or Sleep(2) without a
// doorbell. Nothing in this process raises the Windows timer resolution
// (cuda_link's importer only does so on Python < 3.11, deliberately), so those
// nominal 2ms waits were serviced on the default ~15.625ms tick and actually
// cost ~13.5-17.8ms each (measured on-host). At sub-60fps cadences roughly a
// third of frames fall past the spin budget into the block phase, each paying
// ~9ms mean / ~16ms p99 versus ~0.8ms on the pure-Python fallback — which is
// immune, because CPython >= 3.11 time.sleep() internally uses a
// high-resolution waitable timer. Fix: nap on a per-thread waitable timer
// created with CREATE_WAITABLE_TIMER_HIGH_RESOLUTION (~0.5ms actual for a
// 200us due time — the same primitive CPython's sleep uses). The doorbell is
// no longer waited on here at all: during THIS wait it can only ever signal
// the NEXT frame's publish, never this slot's copy completion, so waking on it
// was an accidental re-check lottery — and consuming a signal here violated
// _doorbell.py's documented single-consumer contract
// (Importer.wait_for_doorbell is the one legitimate consumer).
// doorbell_handle stays in the signature and still selects the
// READY_DOORBELL-vs-READY_POLL status attribution (cuda_link's adaptive-wait
// telemetry keys on the READY_DOORBELL name). Hosts that cannot create
// high-resolution timers (Windows 10 before 1803) are refused at load time via
// hr_nap_supported() — load_native_backend() raises and the Importer falls
// back to the python wait path — rather than silently degraded back onto the
// coarse tick.
//
// Build: see native/CMakeLists.txt. This file is intentionally not compiled in CI
// on Linux; the pure-Python layer is tested via FakeWaitBackend. Validate on a
// Windows box before release.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <tuple>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Older SDK headers may lack this flag (introduced in Windows 10 1803); the
// value is ABI-stable.
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

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

// Bounded nap between block-phase cudaEventQuery re-checks: bounds how stale a
// re-check can get without busy-spinning. 200us due time on a high-resolution
// waitable timer lands at ~0.5ms actual on measured hosts (scheduler wake
// floor) — the same class as the CPython >= 3.11 time.sleep(1e-4) nap in the
// pure-Python fallback path, and ~26x tighter than the pre-fix
// WaitForSingleObject(doorbell, 2ms), which the default ~15.625ms timer tick
// stretched to ~13.5-17.8ms (see the 2026-07-06 fix note in the header).
constexpr int kBlockNapUs = 200;

// Diagnostic `method` code (third slot of the result tuple) for a block-phase
// catch under the timer-nap design. The pre-fix block phase returned 2
// (doorbell-slice wait) or 3 (bare Sleep); those values are retired but never
// reused, so telemetry can tell fixed and unfixed builds apart at runtime.
constexpr int kMethodTimerNap = 4;

}  // namespace

// ---------------------------------------------------------------------------
// The spin/block state machine, parameterized over the polling primitives so a
// test harness can inject fakes (see wait_slot_test() below and
// native/tests/test_state_machine.py).
// ---------------------------------------------------------------------------

template <typename EventQueryFn, typename ReadWriteIdxFn, typename NapFn, typename NowUsFn>
std::tuple<int, double, int> wait_slot_impl(
    EventQueryFn event_query,          // (event_ptr) -> bool ready
    ReadWriteIdxFn read_write_idx,     // (write_idx_addr) -> uint32_t
    NapFn nap,                         // (nap_us) -> void; bounded pacing nap, never exact
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
    // note above the enum). The nap between re-checks is pure pacing: it bounds
    // how stale the next event_query can get (~one kBlockNapUs-class wake)
    // without busy-spinning, and is never treated as a readiness signal. The
    // doorbell is deliberately NOT waited on (see the 2026-07-06 fix note in the
    // header): during this wait it could only ever announce the NEXT frame's
    // publish, and consuming its signal here violated _doorbell.py's
    // single-consumer contract.
    while (true) {
        const double t = now_us();
        if (t >= deadline) {
            return {kTimeout, t - start, 0};
        }

        if (event_ptr != 0 && event_query(event_ptr)) {
            // doorbell_handle != 0 means a doorbell was genuinely in play for
            // this channel — kReadyDoorbell, so telemetry keeps its historical
            // shape (cuda_link's adaptive-wait bookkeeping keys on that name).
            // doorbell_handle == 0 means there was never a doorbell at all —
            // kReadyPoll. Neither implies the doorbell was waited on; both mean
            // "the block phase, not the spin phase, caught it."
            const bool had_doorbell = doorbell_handle != 0;
            return {had_doorbell ? kReadyDoorbell : kReadyPoll, now_us() - start, kMethodTimerNap};
        }

        // Clamp the nap to the remaining deadline budget (re-read the clock:
        // event_query above may itself have consumed time), rounding up so the
        // nap is never zero and forward progress is guaranteed.
        const double remaining_us = deadline - now_us();
        if (remaining_us <= 0.0) {
            continue;  // the deadline check at the loop top reports the timeout
        }
        const int nap_us = remaining_us < static_cast<double>(kBlockNapUs)
                               ? static_cast<int>(remaining_us) + 1
                               : kBlockNapUs;
        nap(nap_us);
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

// One timer per thread, reused across naps: SetWaitableTimer on a handle
// shared between concurrently-waiting threads would re-arm each other's naps
// (multi-Importer processes can wait concurrently), and creating a fresh timer
// per nap wastes a syscall pair. thread_local + RAII gives interference-free
// reuse, released at thread exit.
class HrNapTimer {
 public:
    HrNapTimer()
        : handle_(CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                         TIMER_ALL_ACCESS)) {}
    ~HrNapTimer() {
        if (handle_ != nullptr) {
            CloseHandle(handle_);
        }
    }
    HrNapTimer(const HrNapTimer&) = delete;
    HrNapTimer& operator=(const HrNapTimer&) = delete;
    HANDLE get() const { return handle_; }

 private:
    HANDLE handle_;
};

bool hr_nap_supported() {
    // One-time probe: CREATE_WAITABLE_TIMER_HIGH_RESOLUTION needs Windows 10
    // 1803+. Without it every available nap primitive is quantized to the
    // ~15.625ms default timer tick — exactly the defect the 2026-07-06 fix
    // removed — so load_native_backend() refuses to activate this backend and
    // the Importer's python fallback (strictly better there) takes over.
    static const bool supported = [] {
        HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                          TIMER_ALL_ACCESS);
        if (h == nullptr) {
            return false;
        }
        CloseHandle(h);
        return true;
    }();
    return supported;
}

void real_nap(int nap_us) {
    thread_local HrNapTimer timer;
    HANDLE h = timer.get();
    if (h != nullptr) {
        LARGE_INTEGER due;
        due.QuadPart = -(static_cast<LONGLONG>(nap_us) * 10);  // negative = relative, 100ns units
        if (SetWaitableTimer(h, &due, 0, nullptr, nullptr, FALSE)) {
            // Bounded cap well above any sane wake (~0.5ms measured): if the
            // timer is somehow lost, the caller's deadline check still governs.
            WaitForSingleObject(h, 10);
            return;
        }
    }
    // Unreachable when hr_nap_supported() gated activation (see
    // load_native_backend); if a timer op still fails (e.g. handle
    // exhaustion), degrade to a coarse Sleep rather than busy-spinning — the
    // wait stays correct, just slower.
    Sleep(1);
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
        real_event_query, real_read_write_idx, real_nap, real_now_us,
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
    py::function nap,               // (nap_us: int) -> None; fakes should advance the test clock
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
    auto nap_wrap = [&](int nap_us) { nap(nap_us); };
    auto now_us_wrap = [&]() -> double { return now_us().cast<double>(); };

    return wait_slot_impl(
        event_query_wrap, read_write_idx_wrap, nap_wrap, now_us_wrap,
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
    m.def("hr_nap_supported", &hr_nap_supported,
          "Return True if this host can create high-resolution waitable timers "
          "(Windows 10 1803+), which the block phase's bounded nap requires. "
          "load_native_backend() refuses to activate the native backend when this "
          "is False — the coarse-tick alternatives would reintroduce the "
          "~15.6ms-quantized waits fixed on 2026-07-06.");

    // GIL released for the whole call: all args are passed by value (ints) before
    // release, and the return value is a plain std::tuple<int,double,int> that
    // pybind11 converts to a Python tuple via its automatic caster AFTER the
    // call_guard's destructor re-acquires the GIL — so this is safe (contrast
    // with cuda_link_spout's `receive()`, which manually builds a py::tuple
    // inside the function body and must therefore keep the GIL held throughout).
    m.def("wait_slot", &wait_slot, py::call_guard<py::gil_scoped_release>(),
          "Wait for the per-slot CUDA event: spin on cudaEventQuery for spin_us, "
          "then re-check it between bounded high-resolution timer naps until "
          "timeout_ms. Returns (status_int, waited_us, method) where status_int is "
          "0=READY_SPIN 1=READY_DOORBELL 2=READY_LATE(retired, never emitted) "
          "3=TIMEOUT 4=READY_POLL.");

    // Test-only entry point (see the docstring on wait_slot_test's definition
    // above). GIL held throughout -- not used by the production wait_slot() path.
    // Named py::arg()s so native/tests/test_state_machine.py can call it with
    // keyword arguments.
    m.def("wait_slot_test", &wait_slot_test, py::arg("event_query"), py::arg("read_write_idx"),
          py::arg("nap"), py::arg("now_us"), py::arg("event_ptr"), py::arg("doorbell_handle"),
          py::arg("write_idx_addr"), py::arg("last_write_idx"), py::arg("spin_us"), py::arg("timeout_ms"),
          "TEST ONLY: drive the real wait_slot_impl state machine with injected "
          "Python fake polling functions (event_query, read_write_idx, "
          "nap, now_us). Same return contract as wait_slot(). Used by "
          "native/tests/test_state_machine.py; not part of the production API.");
}
