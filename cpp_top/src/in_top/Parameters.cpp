#include "Parameters.h"

#include "CPlusPlus_Common.h"

const char* Parameters::evalIpcmemname(const TD::OP_Inputs* inputs) {
    return inputs->getParString(IpcmemnameName);
}

bool Parameters::evalActive(const TD::OP_Inputs* inputs) {
    return inputs->getParInt(ActiveName) != 0;
}

bool Parameters::evalDebug(const TD::OP_Inputs* inputs) {
    return inputs->getParInt(DebugName) != 0;
}

void Parameters::setup(TD::OP_ParameterManager* manager) {
    {
        TD::OP_StringParameter p;
        p.name = IpcmemnameName;
        p.label = IpcmemnameLabel;
        p.page = "CUDA Link";
        p.defaultValue = "cudalink_ipc_Python>>TD";

        TD::OP_ParAppendResult res = manager->appendString(p);
        assert(res == TD::OP_ParAppendResult::Success);
    }

    {
        TD::OP_NumericParameter p;
        p.name = ActiveName;
        p.label = ActiveLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 1.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        assert(res == TD::OP_ParAppendResult::Success);
    }

    {
        TD::OP_NumericParameter p;
        p.name = DebugName;
        p.label = DebugLabel;
        p.page = "CUDA Link";
        p.defaultValues[0] = 0.0;

        TD::OP_ParAppendResult res = manager->appendToggle(p);
        assert(res == TD::OP_ParAppendResult::Success);
    }
}
