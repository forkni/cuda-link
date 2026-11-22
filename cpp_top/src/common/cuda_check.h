// CUDA_CHECK macros -- shared error-handling policy for both TOPs: any CUDA call failure
// latches a descriptive error string and turns the current cook into a no-op; no exception
// ever crosses the plugin ABI.
// CUDA-dependent, so this lives in src/common/ (shared by in_top and out_top) rather than
// src/core/, which stays free of any CUDA/TD dependency so it can be unit-tested standalone.

#pragma once

#include <cuda_runtime.h>

#include <string>

// Every latched string carries file+line, matching NVIDIA's own checkCudaErrors pattern --
// otherwise a failure surfaced only via myError/myLastError can't be traced back to which of
// several CUDA_CHECK call sites in the same function fired.
#define CUDALINK_CUDA_CHECK_BOOL(call, err_var)                                                              \
    do {                                                                                                     \
        cudaError_t _cudalink_status = (call);                                                               \
        if (_cudalink_status != cudaSuccess) {                                                               \
            (err_var) = std::string(#call) + " failed: " + cudaGetErrorString(_cudalink_status) + " (" +     \
                        __FILE__ + ":" + std::to_string(__LINE__) + ")";                                     \
            cudaGetLastError(); /* clear the slot so a later call does not report this error */              \
            return false;                                                                                    \
        }                                                                                                    \
    } while (0)

namespace cudalink::common {

// True for the errors the CUDA Runtime API reference describes as leaving "the process in an
// inconsistent state": every later CUDA call returns the same error, so a retry cannot work.
// cudaErrorUnknown is counted too, so an error of unknown kind keeps the old latch behaviour.
// Every other error (cudaErrorInvalidValue from a bad argument, a failed launch config, ...)
// is not sticky: the next cook can succeed once its cause is gone.
inline bool isStickyCudaError(cudaError_t status) {
    switch (status) {
        case cudaErrorIllegalAddress:      // 700
        case cudaErrorLaunchTimeout:       // 702
        case cudaErrorAssert:              // 710
        case cudaErrorHardwareStackError:  // 714
        case cudaErrorIllegalInstruction:  // 715
        case cudaErrorMisalignedAddress:   // 716
        case cudaErrorInvalidAddressSpace: // 717
        case cudaErrorInvalidPc:           // 718
        case cudaErrorLaunchFailure:       // 719
        case cudaErrorUnknown:             // 999
            return true;
        // Same "inconsistent state" text in the CUDA 12.8 driver_types.h. Numbers, not names:
        // CI builds with CUDA 12.6, whose headers may not have every name.
        case static_cast<cudaError_t>(226): // cudaErrorContained
        case static_cast<cudaError_t>(721): // cudaErrorTensorMemoryLeak
        case static_cast<cudaError_t>(911): // cudaErrorExternalDevice
            return true;
        default:
            return false;
    }
}

} // namespace cudalink::common

namespace cudalink::common {

// Adds a warning to 'warning', after any warning already set this cook ("; " between them).
inline void appendWarning(std::string& warning, const std::string& text) {
    if (!warning.empty()) warning += "; ";
    warning += text;
}

} // namespace cudalink::common

// Hot-path variant -- for CUDA calls on the per-frame path. A sticky error (see
// isStickyCudaError) latches fatal_var so the caller's execute() stops retrying CUDA calls
// against a broken context. Any other error is reported for this cook only: the per-thread
// last-error slot is cleared (so neither the next call nor TD's own CUDA code picks it up)
// and the next cook retries. Before 2026-11-21 every error latched, so a plain bad argument
// (cudaErrorInvalidValue from the Common page convert) killed the node for good.
#define CUDALINK_CUDA_CHECK_FATAL(call, err_var, fatal_var)                                                  \
    do {                                                                                                     \
        cudaError_t _cudalink_status = (call);                                                               \
        if (_cudalink_status != cudaSuccess) {                                                               \
            (err_var) = std::string(#call) + " failed: " + cudaGetErrorString(_cudalink_status) + " (" +     \
                        __FILE__ + ":" + std::to_string(__LINE__) + ")";                                     \
            if (cudalink::common::isStickyCudaError(_cudalink_status)) {                                     \
                (fatal_var) = true;                                                                          \
            } else {                                                                                         \
                cudaGetLastError();                                                                          \
            }                                                                                                \
            return;                                                                                          \
        }                                                                                                    \
    } while (0)
