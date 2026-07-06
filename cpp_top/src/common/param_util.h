// param_util.h -- shared OP_ParAppendResult-check helper for both TOPs' Parameters::setup().
//
// setup() uses assert() nowhere on purpose: assert() compiles out entirely under NDEBUG (i.e.
// every Release build, which is all TD ever loads) -- so a parameter-registration failure
// would be silently ignored in production with zero observability. setup() itself is void and
// runs before any TOP instance exists, so there's no myError/status field to latch into;
// OutputDebugStringA is the cheapest guard that's actually observable (via a debugger or
// DebugView) in every build configuration, not just Debug.
//
// Both TOPs' Parameters.cpp previously carried an identical copy of this, differing only in
// the hardcoded "CudaLinkInTOP"/"CudaLinkOutTOP" message prefix -- 'topName' parameterizes
// that so the message stays unchanged per call site.

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <string>

#include "CPlusPlus_Common.h"

namespace cudalink::common {

inline void checkParamAppend(TD::OP_ParAppendResult res, const char* paramName, const char* topName) {
    if (res != TD::OP_ParAppendResult::Success) {
        OutputDebugStringA(
            (std::string(topName) + ": failed to register parameter '" + paramName + "'\n").c_str());
    }
}

} // namespace cudalink::common
