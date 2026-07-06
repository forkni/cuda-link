// CudaLinkInTOP custom parameters — page "CUDA Link".
//
// Reconciled against the real .tox parameter surface (td_exporter/HELP_DOC.md):
// Ipcmemname, Active, Debug are direct mirrors. There is no Mode (the two-DLL split
// makes it unnecessary), no Reconnect (doesn't exist in the real .tox -- Active
// Off->On is the real manual-recovery UX), and no Cudadevice (structurally
// inapplicable to a Custom TOP; single-GPU scope for now).
//
// Numslots is intentionally NOT a parameter here: the Custom TOP API has no mechanism
// for a plugin to write a parameter's displayed value (OP_Inputs is getter-only;
// OP_Inputs::enablePar() only toggles interactivity, never sets a value) -- confirmed by
// exhaustive grep of the vendored CPlusPlus_Common.h (no setPar/readOnly/setDefault
// method exists). The real .tox's "read-only Numslots display in Receiver mode" is
// therefore surfaced here via the Info CHOP channel (CudaLinkInTOP::getInfoCHOPChan),
// alongside the other stats -- not as an OP_ParameterManager parameter.

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

constexpr static char FramewaitName[] = "Framewaitms";
constexpr static char FramewaitLabel[] = "Frame Wait (ms)";

// Framewaitms bounds: 0 disables the doorbell wait entirely (default -- the classic
// return-a-repeat-frame path is untouched); 8 ms is roughly half a 60 fps frame, a hard
// ceiling on how long a NoFrame cook may block TD's cook thread waiting for the producer's
// publish doorbell. Values are clamped at both the UI and eval level.
constexpr static int kFramewaitMin = 0;
constexpr static int kFramewaitMax = 8;

class Parameters {
public:
    static void setup(TD::OP_ParameterManager* manager);

    static const char* evalIpcmemname(const TD::OP_Inputs* inputs);
    static bool evalActive(const TD::OP_Inputs* inputs);
    static bool evalDebug(const TD::OP_Inputs* inputs);
    static int evalFramewaitms(const TD::OP_Inputs* inputs);
};
