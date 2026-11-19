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
            return false;                                                                                    \
        }                                                                                                    \
    } while (0)

// Fatal variant -- for CUDA calls on the per-frame hot path where a failure likely means the
// stream/context is now corrupted for the rest of the process. Latches fatal_var so the
// caller's execute() can short-circuit future cooks instead of retrying the same doomed CUDA
// calls forever against an already-broken context. Not used on cold paths (IPC open,
// reallocate) where a failure is often transient/recoverable (e.g. producer not ready yet) --
// those keep retrying via the plain macros above.
#define CUDALINK_CUDA_CHECK_FATAL(call, err_var, fatal_var)                                                  \
    do {                                                                                                     \
        cudaError_t _cudalink_status = (call);                                                               \
        if (_cudalink_status != cudaSuccess) {                                                               \
            (err_var) = std::string(#call) + " failed: " + cudaGetErrorString(_cudalink_status) + " (" +     \
                        __FILE__ + ":" + std::to_string(__LINE__) + ")";                                     \
            (fatal_var) = true;                                                                              \
            return;                                                                                          \
        }                                                                                                    \
    } while (0)
