#include "Parameters.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <array>
#include <cstddef>
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

int Parameters::evalNumslots(const TD::OP_Inputs* inputs) {
    const int index = inputs->getParInt(NumslotsName);
    constexpr int kNumOptions = static_cast<int>(sizeof(kNumslotsOptions) / sizeof(kNumslotsOptions[0]));
    if (index < 0 || index >= kNumOptions) {
        return kNumslotsOptions[kNumslotsDefaultIndex]; // defensive fallback, should not happen
    }
    return kNumslotsOptions[index];
}

uint32_t Parameters::evalStreamid(const TD::OP_Inputs* inputs) {
    const int value = inputs->getParInt(StreamidName);
    return value > 0 ? static_cast<uint32_t>(value) : 0U;
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
        // Same default as CudaLinkInTOP, so a new Out/In pair connects with no typing. Not
        // the example .tox Sender's name (cudalink_ipc_TD>>Python): a new Out TOP used to
        // publish into that example stream the moment it was created.
        p.defaultValue = "cudalink_ipc_CPP";

        TD::OP_ParAppendResult res = manager->appendString(p);
        checkParamAppend(res, IpcmemnameName, "CudaLinkOutTOP");
    }

    {
        TD::OP_NumericParameter p;
        p.name = ActiveName;
        p.label = ActiveLabel;
        p.page = "CUDA Link";
        // Off on a new node: it connects only after the user sets the IPC Mem Name and
        // turns it on (2026-11-21: a new node joined a live stream the moment it was created).
        p.defaultValues[0] = 0.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        checkParamAppend(res, ActiveName, "CudaLinkOutTOP");
    }

    {
        TD::OP_NumericParameter p;
        p.name = DebugName;
        p.label = DebugLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 0.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        checkParamAppend(res, DebugName, "CudaLinkOutTOP");
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
        checkParamAppend(res, NumslotsName, "CudaLinkOutTOP");
    }

    {
        TD::OP_NumericParameter p;
        p.name = StreamidName;
        p.label = StreamidLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 0.0;
        p.minValues[0] = 0.0;
        p.clampMins[0] = true;
        p.minSliders[0] = 0.0;
        p.maxSliders[0] = static_cast<double>(kStreamidSliderMax);

        TD::OP_ParAppendResult res = manager->appendInt(p);
        checkParamAppend(res, StreamidName, "CudaLinkOutTOP");
    }
}
