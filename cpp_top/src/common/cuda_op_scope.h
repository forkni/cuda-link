// CudaOpScope -- RAII guard for TD::OP_Context's begin/endCUDAOperations() bracket.
//
// CPlusPlus_Common.h documents that all CUDA operations must occur on the main thread and
// between matching begin/endCUDAOperations() calls. A plain call pair around the CUDA work
// is fragile: CUDALINK_CUDA_CHECK (see cuda_check.h) does a bare `return` on any CUDA error,
// and execute()'s own try/catch(...) doesn't know to close a bracket that was left open --
// either path could skip the matching endCUDAOperations() call. Tying end to this guard's
// destructor makes every exit path -- normal return, a CUDALINK_CUDA_CHECK early return, or
// an exception unwinding through catch(...) -- call endCUDAOperations() exactly once, and
// only when begin actually succeeded.

#pragma once

#include "TOP_CPlusPlusBase.h"

namespace cudalink::common {

class CudaOpScope {
public:
    explicit CudaOpScope(TD::OP_Context* context)
        : myContext(context), myOk(context->beginCUDAOperations(nullptr)) {}

    ~CudaOpScope() {
        if (myOk) {
            myContext->endCUDAOperations(nullptr);
        }
    }

    CudaOpScope(const CudaOpScope&) = delete;
    CudaOpScope& operator=(const CudaOpScope&) = delete;
    CudaOpScope(CudaOpScope&&) = delete;
    CudaOpScope& operator=(CudaOpScope&&) = delete;

    // True if beginCUDAOperations() succeeded and the bracket is open.
    explicit operator bool() const { return myOk; }

private:
    TD::OP_Context* myContext;
    bool myOk;
};

} // namespace cudalink::common
