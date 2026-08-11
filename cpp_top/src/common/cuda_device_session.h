// CudaDeviceSession -- shared one-shot device/stream setup for both TOPs' constructors.
//
// CudaLinkInTOP and CudaLinkOutTOP each need to, before creating/importing any IPC handles:
// align the CUDA runtime's current device with TD's own CUDA device selection, verify the
// loaded runtime is ABI-compatible with the one this plugin was built against (PLAN-001 §D3),
// probe the device for CUDA IPC support, and create a non-blocking stream. This was previously
// duplicated verbatim in both constructors -- only the stream-creation policy differed: the
// sender requests the greatest scheduling priority via cudaStreamCreateWithPriority (parity
// with the Python exporter's high_priority_stream policy), falling back to a plain
// non-priority stream if the priority-range query fails; the receiver always uses the plain
// non-priority path. 'highPriorityStream' selects between the two.
//
// Not a long-lived resource owner: the caller still owns and destroys the returned stream
// itself (cudaStreamDestroy in its own destructor), exactly as before this extraction.

#pragma once

#include "TOP_CPlusPlusBase.h"

#include <cuda_runtime.h>

#include <string>

namespace cudalink::common {

struct CudaDeviceSession {
    cudaStream_t stream = nullptr;
    std::string error;
    bool fatal = false;
    // Non-fatal diagnostic (skipped/failed probe, WDDM support envelope note, ...). Callers
    // should surface this even when fatal is false -- see CudaLinkInTOP/CudaLinkOutTOP's
    // myInitNote wiring.
    std::string note;

    CudaDeviceSession(TD::TOP_Context* context, bool highPriorityStream) {
        // getCUDADeviceIndex() returns -1 when this node isn't in CUDA execute mode --
        // shouldn't happen for a TOP_ExecuteMode::CUDA plugin, handled defensively rather
        // than asserted against.
        const int cudaDevice = context->getCUDADeviceIndex(nullptr);
        if (cudaDevice >= 0 && cudaSetDevice(cudaDevice) != cudaSuccess) {
            error = "cudaSetDevice(" + std::to_string(cudaDevice) + ") failed";
            fatal = true;
            return;
        }

        // PLAN-001 §D3: guard at init on the loaded runtime's major version matching the one
        // this plugin was compiled against. Mixing CUDA runtime *major* versions between the
        // build and the loaded cudart is only supported when statically linked (this plugin
        // links dynamically), so a mismatch here is refused rather than risking silent ABI
        // drift (the same class of bug fixed on the Python side for cudaPointerAttributes --
        // see docs/adr/0004-legacy-cuda-ipc-over-vmm.md and cuda_runtime_types.py).
        int runtimeVersion = 0;
        if (cudaRuntimeGetVersion(&runtimeVersion) != cudaSuccess) {
            error = "cudaRuntimeGetVersion() failed -- cannot verify CUDA runtime compatibility";
            fatal = true;
            return;
        }
        if (runtimeVersion / 1000 != CUDART_VERSION / 1000) {
            error = "CUDA runtime major version mismatch: this plugin was built against CUDA " +
                    std::to_string(CUDART_VERSION / 1000) + ".x but the loaded runtime is " +
                    std::to_string(runtimeVersion / 1000) +
                    ".x -- refusing to cook (PLAN-001 §D3; mixing major versions is only "
                    "supported when statically linked)";
            fatal = true;
            return;
        }

        // CUDA Runtime API docs: "Users can test their device for IPC functionality by
        // calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport." That attribute
        // (125) is a CUDA 12.0+ addition -- absent from 11.x driver_types.h -- so both the
        // reference to the enumerator and the query itself must be gated: compile-time via
        // #if CUDART_VERSION, and runtime via the cudaRuntimeGetVersion() result above, since
        // a 12.x-built plugin can still end up loading an older cudart at runtime.
#if CUDART_VERSION >= 12000
        if (runtimeVersion >= 12000) {
            int current = 0;
            if (cudaGetDevice(&current) == cudaSuccess) {
                int ipcSupport = 0;
                cudaError_t attrStatus =
                    cudaDeviceGetAttribute(&ipcSupport, cudaDevAttrIpcEventSupport, current);
                if (attrStatus == cudaSuccess) {
                    if (ipcSupport == 0) {
                        error = "device " + std::to_string(current) +
                                " does not support CUDA IPC (cudaDevAttrIpcEventSupport == 0)";
                        fatal = true;
                        return;
                    }
                    // NVIDIA documents legacy CUDA IPC as Linux-only, or Windows-TCC-only (the
                    // simpleIPC sample gates on prop.tccDriver); this project runs it on the
                    // default Windows WDDM driver model, which works in practice but is outside
                    // that documented support envelope -- see ADR-0004.
                    note = "CUDA IPC capability probe: device " + std::to_string(current) +
                           " reports IPC event support. NVIDIA documents legacy CUDA IPC as "
                           "Linux-only / Windows-TCC-only; this plugin relies on Windows-WDDM "
                           "behaviour that works in practice but is outside NVIDIA's documented "
                           "support envelope (see docs/adr/0004-legacy-cuda-ipc-over-vmm.md). If "
                           "IPC calls fail with error 400 (cudaErrorInvalidResourceHandle) or 801 "
                           "(cudaErrorNotSupported), this undocumented-support gap is the most "
                           "likely cause.";
                } else {
                    // A failed cudaDeviceGetAttribute() call latches this error in the
                    // per-thread last-error slot; left uncleared, TD's own CUDA code could trip
                    // over it later. This probe is purely diagnostic and must never abort
                    // construction on an unexpected query failure -- degrade to a note instead.
                    cudaGetLastError();
                    note = "CUDA IPC capability probe: could not query cudaDevAttrIpcEventSupport "
                           "for device " +
                           std::to_string(current) + " (" + cudaGetErrorString(attrStatus) +
                           "). Continuing without this diagnostic; if IPC calls fail with error "
                           "400 or 801, this is a plausible cause.";
                }
            }
        } else {
            note = "CUDA IPC capability probe skipped: runtime version " + std::to_string(runtimeVersion) +
                   " predates cudaDevAttrIpcEventSupport (added in CUDA 12000). If IPC calls fail "
                   "with error 400 (cudaErrorInvalidResourceHandle) or 801 (cudaErrorNotSupported), "
                   "this older runtime is a plausible cause.";
        }
#else
        note = "CUDA IPC capability probe skipped: this plugin was compiled against CUDART_VERSION " +
               std::to_string(CUDART_VERSION) +
               ", which predates cudaDevAttrIpcEventSupport (added in CUDA 12000). If IPC calls "
               "fail with error 400 (cudaErrorInvalidResourceHandle) or 801 (cudaErrorNotSupported), "
               "this build's older CUDA toolkit is a plausible cause.";
#endif

        // Stream creation is checked: an unchecked failure here would leave 'stream' null,
        // which CUDA silently treats as the default stream (0) instead of surfacing an error.
        cudaError_t st = cudaErrorUnknown;
        if (highPriorityStream) {
            // Per the CUDA docs this is only a scheduling *hint* for pending work -- it
            // cannot preempt work already running and "may not be respected for memory
            // transfers". Falls back to the plain non-priority creation if the range query
            // fails (older/limited drivers), so this can never turn a working device into a
            // non-functional one.
            int leastPriority = 0;
            int greatestPriority = 0;
            if (cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority) == cudaSuccess) {
                st = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
            }
        }
        if (st != cudaSuccess) {
            st = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        }
        if (st != cudaSuccess) {
            error = std::string("cudaStreamCreateWithFlags failed: ") + cudaGetErrorString(st);
            fatal = true;
        }
    }
};

} // namespace cudalink::common
