// CudaLinkOutTOP custom parameters — page "CUDA Link".
//
// Ipcmemname, Active, Debug are direct mirrors of the receiver's (and the real .tox's)
// parameters of the same name. Unlike the receiver, the sender is the side that actually
// owns Numslots on the wire ("editable on the sender only while Active=Off"), so it is a
// real, settable Integer Menu parameter here -- options {2, 3, 4}, matching the real
// .tox's Numslots menu (td_exporter/HELP_DOC.md) exactly. No Mode (the two-DLL split
// makes it unnecessary) and no Reconnect/Cudadevice (same reasoning as the receiver).

#pragma once

namespace TD {
class OP_Inputs;
class OP_ParameterManager;
} // namespace TD

constexpr static char IpcmemnameName[] = "Ipcmemname";
constexpr static char IpcmemnameLabel[] = "IPC Mem Name";

constexpr static char ActiveName[] = "Active";
constexpr static char ActiveLabel[] = "Active";

constexpr static char DebugName[] = "Debug";
constexpr static char DebugLabel[] = "Debug";

constexpr static char NumslotsName[] = "Numslots";
constexpr static char NumslotsLabel[] = "Num Slots";

// Menu index -> actual slot count. Menu entries are the literal option strings; getParInt
// returns the 0-based selected index (confirmed against CustomOperatorSamples/SpectrumTOP's
// appendMenu + getParInt pattern), not the slot count itself -- evalNumslots() below
// converts index -> count so callers never need to know the menu encoding.
constexpr static int kNumslotsOptions[] = {2, 3, 4};
constexpr static int kNumslotsDefaultIndex = 1; // "3" -- matches the real .tox's default

class Parameters {
public:
    static void setup(TD::OP_ParameterManager* manager);

    static const char* evalIpcmemname(const TD::OP_Inputs* inputs);
    static bool evalActive(const TD::OP_Inputs* inputs);
    static bool evalDebug(const TD::OP_Inputs* inputs);
    // Returns the actual slot count (2, 3, or 4), not the raw menu index.
    static int evalNumslots(const TD::OP_Inputs* inputs);

    // Numslots is only meaningful/editable while the sender is not actively holding an
    // open SHM segment (changing it mid-run would desync a running consumer without a
    // VERSION_CHANGED-style reinit, which the sender always performs anyway on Active
    // Off->On -- "editable on the sender only while Active=Off").
    static void setNumslotsEnabled(const TD::OP_Inputs* inputs, bool enabled);
};
