#include "Parameters.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <array>
#include <cstddef>
#include <string>

#include "CPlusPlus_Common.h"

namespace {
// C++ Guidelines audit LOW/cosmetic item: setup() previously used assert() to check each
// appendString/appendToggle/appendMenu result, but assert() compiles out entirely under
// NDEBUG (i.e. every Release build, which is all TD ever loads) -- so a parameter-registration
// failure here was silently ignored in production with zero observability. setup() itself is
// void and runs before any TOP instance exists, so there's no myError/status field to latch
// into; OutputDebugStringA is the cheapest guard that's actually observable (via a debugger or
// DebugView) in every build configuration, not just Debug.
void checkParamAppend(TD::OP_ParAppendResult res, const char* paramName) {
    if (res != TD::OP_ParAppendResult::Success) {
        OutputDebugStringA((std::string("CudaLinkOutTOP: failed to register parameter '") + paramName + "'\n").c_str());
    }
}
} // namespace

const char* Parameters::evalIpcmemname(const TD::OP_Inputs* inputs) {
    return inputs->getParString(IpcmemnameName);
}

bool Parameters::evalActive(const TD::OP_Inputs* inputs) {
    return inputs->getParInt(ActiveName) != 0;
}

bool Parameters::evalDebug(const TD::OP_Inputs* inputs) {
    return inputs->getParInt(DebugName) != 0;
}

int Parameters::evalNumslots(const TD::OP_Inputs* inputs) {
    const int index = inputs->getParInt(NumslotsName);
    constexpr int kNumOptions = static_cast<int>(sizeof(kNumslotsOptions) / sizeof(kNumslotsOptions[0]));
    if (index < 0 || index >= kNumOptions) {
        return kNumslotsOptions[kNumslotsDefaultIndex]; // defensive fallback, should not happen
    }
    return kNumslotsOptions[index];
}

void Parameters::setNumslotsEnabled(const TD::OP_Inputs* inputs, bool enabled) {
    inputs->enablePar(NumslotsName, enabled);
}

void Parameters::setup(TD::OP_ParameterManager* manager) {
    {
        TD::OP_StringParameter p;
        p.name = IpcmemnameName;
        p.label = IpcmemnameLabel;
        p.page = "CUDA Link";
        // Matches the default direction the Python example scripts exercise: TD as
        // sender, a separate Python process as receiver (td_exporter/
        // example_receiver_python.py's default CUDALINK_RECEIVER_SHM_NAME).
        p.defaultValue = "cudalink_ipc_TD>>Python";

        TD::OP_ParAppendResult res = manager->appendString(p);
        checkParamAppend(res, IpcmemnameName);
    }

    {
        TD::OP_NumericParameter p;
        p.name = ActiveName;
        p.label = ActiveLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 1.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        checkParamAppend(res, ActiveName);
    }

    {
        TD::OP_NumericParameter p;
        p.name = DebugName;
        p.label = DebugLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 0.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        checkParamAppend(res, DebugName);
    }

    {
        TD::OP_StringParameter p;
        p.name = NumslotsName;
        p.label = NumslotsLabel;
        p.page = "CUDA Link";
        p.defaultValue = "3";
        std::array<const char*, 3> names = {"2", "3", "4"};
        std::array<const char*, 3> labels = {"2", "3", "4"};

        TD::OP_ParAppendResult res =
            manager->appendMenu(p, static_cast<int32_t>(names.size()), names.data(), labels.data());
        checkParamAppend(res, NumslotsName);
    }
}
