#include "Parameters.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <string>

#include "CPlusPlus_Common.h"
#include "../common/param_util.h"

using cudalink::common::checkParamAppend;

const char* Parameters::evalIpcmemname(const TD::OP_Inputs* inputs) {
    return inputs->getParString(IpcmemnameName);
}

bool Parameters::evalActive(const TD::OP_Inputs* inputs) {
    return inputs->getParInt(ActiveName) != 0;
}

bool Parameters::evalDebug(const TD::OP_Inputs* inputs) {
    return inputs->getParInt(DebugName) != 0;
}

int Parameters::evalFramewaitms(const TD::OP_Inputs* inputs) {
    // Re-clamped here despite the UI-level clamp in setup(): getParInt() can return
    // out-of-range values via expressions/exports, and this value becomes a
    // WaitForSingleObject timeout on the cook thread.
    const int value = inputs->getParInt(FramewaitName);
    if (value < kFramewaitMin) return kFramewaitMin;
    if (value > kFramewaitMax) return kFramewaitMax;
    return value;
}

void Parameters::setup(TD::OP_ParameterManager* manager) {
    {
        TD::OP_StringParameter p;
        p.name = IpcmemnameName;
        p.label = IpcmemnameLabel;
        p.page = "CUDA Link";
        p.defaultValue = "cudalink_ipc_Python>>TD";

        TD::OP_ParAppendResult res = manager->appendString(p);
        checkParamAppend(res, IpcmemnameName, "CudaLinkInTOP");
    }

    {
        TD::OP_NumericParameter p;
        p.name = ActiveName;
        p.label = ActiveLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 1.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        checkParamAppend(res, ActiveName, "CudaLinkInTOP");
    }

    {
        TD::OP_NumericParameter p;
        p.name = DebugName;
        p.label = DebugLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 0.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        checkParamAppend(res, DebugName, "CudaLinkInTOP");
    }

    {
        // Opt-in doorbell wait (default 0 = off): on a cook where no new frame has been
        // published yet, block up to this many ms on the producer's publish doorbell
        // instead of instantly re-displaying the previous texture. Converts phase-aliasing
        // repeat frames (publish landing just after the read moment) into slightly-later
        // fresh frames. See CudaLinkInTOP::waitForFreshFrame() for the caveats.
        TD::OP_NumericParameter p;
        p.name = FramewaitName;
        p.label = FramewaitLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 0.0;
        p.minValues[0] = static_cast<double>(kFramewaitMin);
        p.maxValues[0] = static_cast<double>(kFramewaitMax);
        p.clampMins[0] = true;
        p.clampMaxes[0] = true;
        p.minSliders[0] = static_cast<double>(kFramewaitMin);
        p.maxSliders[0] = static_cast<double>(kFramewaitMax);

        TD::OP_ParAppendResult res = manager->appendInt(p);
        checkParamAppend(res, FramewaitName, "CudaLinkInTOP");
    }
}
