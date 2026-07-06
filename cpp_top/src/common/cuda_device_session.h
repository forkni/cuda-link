// CudaDeviceSession -- shared one-shot device/stream setup for both TOPs' constructors.
//
// CudaLinkInTOP and CudaLinkOutTOP each need to, before creating/importing any IPC handles:
// align the CUDA runtime's current device with TD's own CUDA device selection, verify the
// device supports CUDA IPC, and create a non-blocking stream. This was previously duplicated
// verbatim in both constructors -- only the stream-creation policy differed: the sender
// requests the greatest scheduling priority via cudaStreamCreateWithPriority (parity with the
// Python exporter's high_priority_stream policy), falling back to a plain non-priority stream
// if the priority-range query fails; the receiver always uses the plain non-priority path.
// 'highPriorityStream' selects between the two.
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

        // CUDA Runtime API docs: "Users can test their device for IPC functionality by
        // calling cudaDeviceGetAttribute with cudaDevAttrIpcEventSupport."
        int current = 0;
        if (cudaGetDevice(&current) == cudaSuccess) {
            int ipcSupport = 0;
            if (cudaDeviceGetAttribute(&ipcSupport, cudaDevAttrIpcEventSupport, current) == cudaSuccess &&
                ipcSupport == 0) {
                error = "device " + std::to_string(current) +
                        " does not support CUDA IPC (cudaDevAttrIpcEventSupport == 0)";
                fatal = true;
                return;
            }
        }

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
