// CUDA_CHECK macro -- D7's error policy: every CUDA call latches an error string and
// turns the cook into a no-op; no exception ever crosses the ABI.
// CUDA-dependent, so lives here (src/common/, shared by in_top and out_top) rather than
// src/core/, which D4 requires to stay CUDA-free.

#pragma once

#include <cuda_runtime.h>

#include <string>

#define CUDALINK_CUDA_CHECK(call, err_var)                                                       \
    do {                                                                                         \
        cudaError_t _cudalink_status = (call);                                                   \
        if (_cudalink_status != cudaSuccess) {                                                   \
            (err_var) = std::string(#call) + " failed: " + cudaGetErrorString(_cudalink_status); \
            return;                                                                              \
        }                                                                                         \
    } while (0)

#define CUDALINK_CUDA_CHECK_BOOL(call, err_var)                                                  \
    do {                                                                                         \
        cudaError_t _cudalink_status = (call);                                                   \
        if (_cudalink_status != cudaSuccess) {                                                   \
            (err_var) = std::string(#call) + " failed: " + cudaGetErrorString(_cudalink_status); \
            return false;                                                                        \
        }                                                                                         \
    } while (0)
