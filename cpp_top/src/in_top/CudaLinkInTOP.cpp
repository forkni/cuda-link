#include "CudaLinkInTOP.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstring>

#include "Parameters.h"
#include "pixel_format_map.h"
#include "../common/cuda_check.h"

namespace {
// Wall-clock seconds as a double -- used only for debug-log timestamps here (unlike the
// sender, the receiver never writes the wire timestamp field).
double now_seconds() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}
} // namespace

using cudalink::core::AcquireResult;
using cudalink::core::Metadata;
using cudalink::core::PROTOCOL_MAGIC;
using cudalink::core::SHMLayout;
using cudalink::core::SlotState;
using cudalink::core::acquire_slot;
using cudalink::core::read_magic;
using cudalink::core::read_num_slots;
using cudalink::core::read_version;

namespace {
HANDLE asHandle(void* h) { return static_cast<HANDLE>(h); }
} // namespace

extern "C" {

DLLEXPORT
void FillTOPPluginInfo(TD::TOP_PluginInfo* info) {
    if (!info->setAPIVersion(TD::TOPCPlusPlusAPIVersion)) return;

    info->executeMode = TD::TOP_ExecuteMode::CUDA;

    TD::OP_CustomOPInfo& customInfo = info->customOPInfo;
    customInfo.opType->setString("Cudalinkin");
    customInfo.opLabel->setString("CUDA Link In");
    customInfo.opIcon->setString("CLI");
    customInfo.authorName->setString("forkni");
    customInfo.authorEmail->setString("forkni@gmail.com");

    customInfo.minInputs = 0;
    customInfo.maxInputs = 0;

    // Required alongside cookEveryFrame=true (getGeneralInfo) or every-frame cooking
    // never kick-starts for a node with no inputs and nothing initially viewing it.
    customInfo.cookOnStart = true;
}

DLLEXPORT
TD::TOP_CPlusPlusBase* CreateTOPInstance(const TD::OP_NodeInfo* info, TD::TOP_Context* context) {
    return new CudaLinkInTOP(info, context);
}

DLLEXPORT
void DestroyTOPInstance(TD::TOP_CPlusPlusBase* instance, TD::TOP_Context* context) {
    delete (CudaLinkInTOP*)instance;
}

} // extern "C"

CudaLinkInTOP::CudaLinkInTOP(const TD::OP_NodeInfo*, TD::TOP_Context* context) : myContext(context) {
    cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
}

CudaLinkInTOP::~CudaLinkInTOP() {
    teardown();
    if (myStream) {
        cudaStreamDestroy(myStream);
    }
}

void CudaLinkInTOP::getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs*, void*) {
    // The producer is external to TD; nothing in TD's dependency graph (no inputs, no
    // changing parameters most cooks) would otherwise trigger a re-cook when a new
    // frame arrives. Must poll every frame while Active (D1).
    ginfo->cookEveryFrame = true;
}

void CudaLinkInTOP::setupParameters(TD::OP_ParameterManager* manager, void*) {
    Parameters::setup(manager);
}

// ---------------------------------------------------------------------------
// D6 step 0 -- parameter-change detection (no push notification exists, D7)
// ---------------------------------------------------------------------------

void CudaLinkInTOP::checkParameterChanges(const TD::OP_Inputs* inputs) {
    const bool active = Parameters::evalActive(inputs);
    const char* ipcmemnameRaw = Parameters::evalIpcmemname(inputs);
    const std::string ipcmemname = ipcmemnameRaw ? ipcmemnameRaw : "";

    // Cached once per cook so other methods can check it without needing OP_Inputs
    // threaded through (mirrors the same addition on the sender side).
    myDebugEnabled = Parameters::evalDebug(inputs);

    if (myFirstCook) {
        myCachedActive = active;
        myCachedIpcmemname = ipcmemname;
        myFirstCook = false;
        return;
    }

    // Active edge (either direction): full teardown. Off->On must behave as a fresh
    // connection, never resuming stale state (D6/D7); On->Off must free everything
    // immediately (matches the real .tox's documented Active=Off behavior).
    if (active != myCachedActive) {
        teardown();
    }
    // Ipcmemname changed while running: drop the current connection so step 1 attempts
    // the new name this very cook (mirrors request_immediate_reconnect() -- there is no
    // backoff timer here to reset, since a Custom TOP's execute() already retries every
    // cook, unlike Python's exponential-backoff retry state machine).
    else if (ipcmemname != myCachedIpcmemname) {
        teardown();
    }

    myCachedActive = active;
    myCachedIpcmemname = ipcmemname;
}

// ---------------------------------------------------------------------------
// Debug logging (opt-in, Debug parameter) -- live-test finding: the transient error
// badge alone wasn't enough to diagnose what was actually failing during the sender's
// resolution/format switch (VERSION_CHANGED window).
// ---------------------------------------------------------------------------

void CudaLinkInTOP::debugLog(const std::string& msg) {
    if (!myDebugEnabled) return;
    if (!myDebugLogFile.is_open()) {
        char tempPath[MAX_PATH] = {};
        GetTempPathA(MAX_PATH, tempPath);
        myDebugLogFile.open(std::string(tempPath) + "cudalink_in_top_debug.log", std::ios::app);
        if (!myDebugLogFile.is_open()) return;
    }
    // Milliseconds since epoch as an integer -- see CudaLinkOutTOP.cpp::debugLog for why
    // (printing the raw double lost all sub-second precision: every line showed the
    // identical "1.78324e+09").
    const int64_t ms = static_cast<int64_t>(now_seconds() * 1000.0);
    myDebugLogFile << "[t=" << ms << " frame=" << myFrameCount << "] " << msg << "\n";
    myDebugLogFile.flush();
}

// ---------------------------------------------------------------------------
// Teardown / handle lifecycle
// ---------------------------------------------------------------------------

void CudaLinkInTOP::closeHandles() {
    for (auto* devPtr : mySlotDevPtrs) {
        if (devPtr) {
            cudaIpcCloseMemHandle(devPtr);
        }
    }
    mySlotDevPtrs.clear();
    for (auto evt : mySlotEvents) {
        if (evt) {
            cudaEventDestroy(evt);
        }
    }
    mySlotEvents.clear();
    myHandlesOpen = false;
}

void CudaLinkInTOP::teardown() {
    debugLog("teardown: begin");
    closeHandles();
    if (myShmView) {
        UnmapViewOfFile(myShmView);
        myShmView = nullptr;
    }
    if (myShmHandle) {
        CloseHandle(asHandle(myShmHandle));
        myShmHandle = nullptr;
    }
    myLastVersion = 0;
    myLastWriteIdx = 0;
    myLayout = SHMLayout(0);
    myShmMappedSize = 0;
    myStatus = "Waiting for producer";
    debugLog("teardown: complete");
}

bool CudaLinkInTOP::openSHM(const char* name) {
    if (myShmView) {
        return true; // already open
    }
    // Naive byte-widening, not a general UTF-8 decoder -- correct for the ASCII SHM
    // names this protocol uses in practice (matches Python's CreateFileMapping tagname
    // verbatim-and-unprefixed contract, D5's SHM naming interop note).
    HANDLE h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, std::wstring(name, name + std::strlen(name)).c_str());
    if (!h) {
        myStatus = "Waiting for producer";
        return false;
    }
    // Map generously; the mapping size is not stored anywhere on the wire (SHM naming
    // interop note, D5), so Python attachers observe a page-rounded size and this side
    // does the same by mapping 0 bytes (maps the whole underlying section).
    void* view = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!view) {
        CloseHandle(h);
        myError = "MapViewOfFile failed";
        return false;
    }

    // Real .tox's own Troubleshooting doc documents this exact failure mode: "another
    // process is using the same Ipcmemname for a different purpose."
    if (read_magic(static_cast<const uint8_t*>(view)) != PROTOCOL_MAGIC) {
        UnmapViewOfFile(view);
        CloseHandle(h);
        myError = "Protocol magic mismatch -- Ipcmemname is in use by an unrelated process";
        return false;
    }

    // Query the mapping's actual accessible size (same technique D5's SHM-naming note
    // describes Python attachers using -- the mapping size is not stored anywhere on
    // the wire). Used to bounds-check a wire-sourced num_slots before trusting it
    // (validateNumSlots): named SHM is attachable by any process that knows the name,
    // so its contents -- including num_slots -- are treated as untrusted input.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(view, &mbi, sizeof(mbi)) == 0) {
        UnmapViewOfFile(view);
        CloseHandle(h);
        myError = "VirtualQuery failed";
        return false;
    }

    myShmHandle = h;
    myShmView = static_cast<uint8_t*>(view);
    myShmMappedSize = mbi.RegionSize;
    return true;
}

bool CudaLinkInTOP::validateNumSlots(uint32_t numSlots) {
    // SHMLayout's offset math is uint32_t; numSlots * SLOT_SIZE can silently wrap on a
    // pathological value, which would defeat a total_size()-based bounds check applied
    // after the fact. The realistic range is 2-4 (HELP_DOC.md's Numslots menu); a cap of
    // 64 leaves generous headroom while keeping numSlots * 128 (SLOT_SIZE) many orders
    // of magnitude below uint32_t's wraparound point.
    constexpr uint32_t kMaxReasonableSlots = 64;
    if (numSlots == 0 || numSlots > kMaxReasonableSlots) {
        myError = "invalid num_slots on wire (" + std::to_string(numSlots) + ")";
        return false;
    }
    const cudalink::core::SHMLayout candidate(numSlots);
    if (candidate.total_size() > myShmMappedSize) {
        // The claimed slot count would require reading past the actual mapped view --
        // reject rather than risk an out-of-bounds read into unmapped memory.
        myError = "num_slots (" + std::to_string(numSlots) + ") implies a layout larger than the mapped SHM region";
        return false;
    }
    return true;
}

bool CudaLinkInTOP::openSlotHandlesIfNeeded() {
    if (myHandlesOpen) {
        return true;
    }

    const uint32_t numSlots = read_num_slots(myShmView);
    if (!validateNumSlots(numSlots)) {
        return false;
    }
    myLayout = SHMLayout(numSlots);
    if (myLastVersion == 0) {
        // True first connect (never adopted a version yet, D6 step 3 discussion):
        // acquire_slot() deliberately does not report VERSION_CHANGED when
        // last_version==0, so this is the only place the initial version gets adopted.
        myLastVersion = read_version(myShmView);
    }

    myMetadata = Metadata::read_from(myShmView, myLayout);

    mySlotDevPtrs.assign(numSlots, nullptr);
    mySlotEvents.assign(numSlots, nullptr);

    for (uint32_t slot = 0; slot < numSlots; ++slot) {
        cudaIpcMemHandle_t memHandle;
        std::memcpy(&memHandle, myShmView + myLayout.mem_handle_offset(slot), sizeof(memHandle));
        CUDALINK_CUDA_CHECK_BOOL(cudaIpcOpenMemHandle(&mySlotDevPtrs[slot], memHandle, cudaIpcMemLazyEnablePeerAccess),
                                  myError);

        cudaIpcEventHandle_t eventHandle;
        std::memcpy(&eventHandle, myShmView + myLayout.event_handle_offset(slot), sizeof(eventHandle));
        CUDALINK_CUDA_CHECK_BOOL(cudaIpcOpenEventHandle(&mySlotEvents[slot], eventHandle), myError);
    }

    myHandlesOpen = true;
    return true;
}

// ---------------------------------------------------------------------------
// execute() -- D6
// ---------------------------------------------------------------------------

void CudaLinkInTOP::execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void*) {
    const auto cookStart = std::chrono::steady_clock::now();
    try {
        myError.clear();
        myWarning.clear();

        // Step 0: cached-diff parameter-change detection (no push notification exists).
        checkParameterChanges(inputs);

        // Step 1: Active gate + SHM open (retried every cook -- no backoff needed for a
        // per-cook-polled Custom TOP).
        if (!Parameters::evalActive(inputs)) {
            myStatus = "Idle";
            return;
        }
        if (!myShmView) {
            const char* name = Parameters::evalIpcmemname(inputs);
            if (!name || !openSHM(name)) {
                return; // status already set to "Waiting for producer"
            }
            // BUG FIX: myLayout must be built from the real num_slots *before* the first
            // acquire_slot() call below -- acquire_slot() reads shutdown_offset(), which
            // depends on num_slots. Left at its zero-initialized default (num_slots=0),
            // shutdown_offset() resolves to byte 20 -- inside slot 0's raw IPC mem-handle
            // bytes, not the real shutdown flag -- which is essentially random binary
            // data and was being misread as a nonzero (shutdown) flag on every cook,
            // causing a rapid open/teardown/reopen churn loop that looked like a TD hang.
            const uint32_t numSlots = read_num_slots(myShmView);
            if (!validateNumSlots(numSlots)) {
                teardown();
                return;
            }
            myLayout = SHMLayout(numSlots);
        }

        // Step 2: classify SHM state.
        AcquireResult result = acquire_slot(myShmView, myLayout, myLastWriteIdx, myLastVersion);

        switch (result.state) {
            case SlotState::NoFrame:
                return; // previous output persists (D6/R5 -- SpectrumTOP.cpp / PyTorchTOP.cpp precedent)
            case SlotState::Shutdown:
                teardown();
                myStatus = "Producer exited";
                return;
            case SlotState::VersionChanged:
                debugLog("VERSION_CHANGED: new_version=" + std::to_string(result.new_version));
                closeHandles(); // re-derive layout from a freshly-read num_slots (R8)
                myLastVersion = result.new_version;
                // A new producer session's write_idx also restarts at 0 (SHMLayout::
                // build_buffer's default); without this reset, a stale myLastWriteIdx
                // that happens to numerically match the new session's counter later
                // would cause acquire_slot() to silently report NO_FRAME and miss a
                // real frame.
                myLastWriteIdx = 0;
                break; // fall through to handle-opening below
            case SlotState::NewFrame:
                break;
        }

        // Step 3: open-once-per-version (covers both VERSION_CHANGED fall-through and
        // the true first-connect case uniformly).
        if (!openSlotHandlesIfNeeded()) {
            return; // myError already set
        }

        if (result.state == SlotState::VersionChanged) {
            // VERSION_CHANGED alone carries no frame to display this cook; wait for the
            // next NEW_FRAME classification (mirrors _refresh_on_version_change).
            return;
        }

        // Defensive bound check: result.slot is derived from wire-sourced write_idx and
        // num_slots (both untrusted -- see validateNumSlots()); a malformed or malicious
        // SHM segment could in principle desync this from the actual size of
        // mySlotDevPtrs/mySlotEvents. Never index with an unchecked value.
        if (result.slot >= mySlotDevPtrs.size()) {
            myError = "acquire_slot returned out-of-range slot index";
            return;
        }
        myReadSlot = result.slot;

        // Step 4: create the output texture from cached metadata.
        const TD::OP_PixelFormat pixelFormat = cudalink::in_top::mapToPixelFormat(myMetadata);
        if (pixelFormat == TD::OP_PixelFormat::Invalid) {
            myWarning = "unsupported pixel format on wire (kind=" + std::to_string(myMetadata.format_kind) +
                        " bits=" + std::to_string(myMetadata.bits_per_comp) +
                        " flags=" + std::to_string(myMetadata.flags) + ")";
            return;
        }

        TD::TOP_CUDAOutputInfo info;
        info.textureDesc.width = myMetadata.width;
        info.textureDesc.height = myMetadata.height;
        info.textureDesc.texDim = TD::OP_TexDim::e2D;
        info.textureDesc.pixelFormat = pixelFormat;
        info.stream = myStream;

        const TD::OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(info, nullptr);
        if (!outputInfo) {
            myError = "createCUDAArray failed";
            return;
        }

        // Step 5: GPU-side wait + D2D copy, no CPU block (R6 -- confirmed by
        // CPlusPlus_Common.h's documented begin/end bracket purpose, not just inferred).
        if (!myContext->beginCUDAOperations(nullptr)) {
            myError = "beginCUDAOperations failed";
            return;
        }

        CUDALINK_CUDA_CHECK(cudaStreamWaitEvent(myStream, mySlotEvents[myReadSlot], 0), myError);

        const auto copyStart = std::chrono::steady_clock::now();
        const size_t itemsize = myMetadata.bits_per_comp / 8;
        const size_t rowBytes = static_cast<size_t>(myMetadata.width) * myMetadata.num_comps * itemsize;
        CUDALINK_CUDA_CHECK(cudaMemcpy2DToArrayAsync(outputInfo->cudaArray, 0, 0, mySlotDevPtrs[myReadSlot], rowBytes,
                                                       rowBytes, myMetadata.height, cudaMemcpyDeviceToDevice, myStream),
                            myError);
        myContext->endCUDAOperations(nullptr);
        // CPU-side enqueue cost only (the copy itself is async on myStream) -- a true
        // GPU-side copy_us would need cudaEvent-based timing; deferred, not required for
        // correctness.
        myCopyUs = std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - copyStart).count();

        // Step 6.
        myLastWriteIdx = result.write_idx;
        ++myFrameCount;
        myStatus = std::to_string(myMetadata.width) + "x" + std::to_string(myMetadata.height);
    } catch (...) {
        // No exception ever crosses the ABI (D7 error policy).
        myError = "unexpected exception in execute()";
    }
    myCookUs = std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - cookStart).count();
}

// ---------------------------------------------------------------------------
// Info CHOP / DAT / status (D7)
// ---------------------------------------------------------------------------

int32_t CudaLinkInTOP::getNumInfoCHOPChans(void*) {
    return 6; // frames, cook_us, copy_us, write_idx, read_slot, num_slots
}

void CudaLinkInTOP::getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan* chan, void*) {
    switch (index) {
        case 0:
            chan->name->setString("frames");
            chan->value = static_cast<float>(myFrameCount);
            break;
        case 1:
            chan->name->setString("cook_us");
            chan->value = myCookUs;
            break;
        case 2:
            chan->name->setString("copy_us");
            chan->value = myCopyUs;
            break;
        case 3:
            chan->name->setString("write_idx");
            chan->value = static_cast<float>(myLastWriteIdx);
            break;
        case 4:
            chan->name->setString("read_slot");
            chan->value = static_cast<float>(myReadSlot);
            break;
        case 5:
            // The real .tox's read-only Numslots display, sourced from the wire header
            // (R2/R8) -- surfaced here rather than as a parameter (see CudaLinkInTOP.h).
            chan->name->setString("num_slots");
            chan->value = static_cast<float>(myLayout.num_slots());
            break;
        default:
            break;
    }
}

bool CudaLinkInTOP::getInfoDATSize(TD::OP_InfoDATSize* infoSize, void*) {
    infoSize->rows = 4; // ipc_version, status, last_error, last_error_frame
    infoSize->cols = 2;
    infoSize->byColumn = false;
    return true;
}

void CudaLinkInTOP::getInfoDATEntries(int32_t index, int32_t, TD::OP_InfoDATEntries* entries, void*) {
    if (index == 0) {
        entries->values[0]->setString("ipc_version");
        entries->values[1]->setString(std::to_string(myLastVersion).c_str());
    } else if (index == 1) {
        entries->values[0]->setString("status");
        entries->values[1]->setString(myStatus.c_str());
    } else if (index == 2) {
        // Sticky (never auto-cleared, only overwritten) -- see getErrorString(). Survives
        // across cooks, unlike the transient error badge (live-test finding: a real error
        // during the sender's VERSION_CHANGED window flashed and disappeared too fast to
        // read).
        entries->values[0]->setString("last_error");
        entries->values[1]->setString(myLastError.c_str());
    } else if (index == 3) {
        entries->values[0]->setString("last_error_frame");
        entries->values[1]->setString(std::to_string(myLastErrorFrame).c_str());
    }
}

void CudaLinkInTOP::getErrorString(TD::OP_String* error, void*) {
    if (!myError.empty()) {
        myLastError = myError;
        myLastErrorFrame = myFrameCount;
        debugLog("ERROR: " + myError);
    }
    error->setString(myError.c_str());
    myError.clear();
}

void CudaLinkInTOP::getWarningString(TD::OP_String* warning, void*) {
    if (!myWarning.empty()) {
        debugLog("WARNING: " + myWarning);
    }
    warning->setString(myWarning.c_str());
    myWarning.clear();
}

void CudaLinkInTOP::getInfoPopupString(TD::OP_String* info, void*) { info->setString(myStatus.c_str()); }
