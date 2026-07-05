#include "CudaLinkOutTOP.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstring>

#include "Parameters.h"
#include "../common/cuda_check.h"

using cudalink::core::Metadata;
using cudalink::core::SHMLayout;
using cudalink::core::clear_shutdown;
using cudalink::core::commit_version;
using cudalink::core::publish;
using cudalink::core::read_version;
using cudalink::core::set_shutdown;
using cudalink::core::write_init_header;
using cudalink::core::write_slot_handle;
using cudalink::out_top::WireFormat;
using cudalink::out_top::mapFromPixelFormat;

namespace {
HANDLE asHandle(void* h) { return static_cast<HANDLE>(h); }

// Wall-clock seconds as a double, matching Python's time.time() for the wire timestamp
// field (D5's publish() call uses the same units as shm_protocol.py::publish_frame).
double now_seconds() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

// Naive byte-widening, not a general UTF-8 decoder -- correct for the ASCII SHM names
// this protocol uses in practice (matches CudaLinkInTOP::openSHM's identical technique
// and the SHM naming interop note, D5: names are verbatim & unprefixed).
std::wstring widen(const char* s) { return std::wstring(s, s + std::strlen(s)); }
} // namespace

extern "C" {

DLLEXPORT
void FillTOPPluginInfo(TD::TOP_PluginInfo* info) {
    if (!info->setAPIVersion(TD::TOPCPlusPlusAPIVersion)) return;

    info->executeMode = TD::TOP_ExecuteMode::CUDA;

    TD::OP_CustomOPInfo& customInfo = info->customOPInfo;
    customInfo.opType->setString("Cudalinkout");
    customInfo.opLabel->setString("CUDA Link Out");
    customInfo.opIcon->setString("CLO");
    customInfo.authorName->setString("forkni");
    customInfo.authorEmail->setString("forkni@gmail.com");

    customInfo.minInputs = 1;
    customInfo.maxInputs = 1;

    // Mirrors CudaLinkInTOP: harmless with an input present, and guarantees a first cook
    // even if the upstream input isn't already cooking every frame on its own.
    customInfo.cookOnStart = true;
}

DLLEXPORT
TD::TOP_CPlusPlusBase* CreateTOPInstance(const TD::OP_NodeInfo* info, TD::TOP_Context* context) {
    return new CudaLinkOutTOP(info, context);
}

DLLEXPORT
void DestroyTOPInstance(TD::TOP_CPlusPlusBase* instance, TD::TOP_Context* context) {
    delete (CudaLinkOutTOP*)instance;
}

} // extern "C"

CudaLinkOutTOP::CudaLinkOutTOP(const TD::OP_NodeInfo*, TD::TOP_Context* context) : myContext(context) {
    cudaStreamCreateWithFlags(&myStream, cudaStreamNonBlocking);
}

CudaLinkOutTOP::~CudaLinkOutTOP() {
    teardown();
    if (myStream) {
        cudaStreamDestroy(myStream);
    }
}

void CudaLinkOutTOP::getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs*, void*) {
    // Publish every cook while Active, independent of whether the input's own contents
    // changed -- keeps write_idx advancing as a heartbeat and matches the receiver's
    // equally unconditional per-cook polling (D1/D5).
    ginfo->cookEveryFrame = true;
}

void CudaLinkOutTOP::setupParameters(TD::OP_ParameterManager* manager, void*) {
    Parameters::setup(manager);
}

// ---------------------------------------------------------------------------
// D7 -- parameter-change detection (no push notification exists; sender's own copy of
// the receiver's cached-diff pattern, flagged as follow-up work in D7 and implemented
// here)
// ---------------------------------------------------------------------------

void CudaLinkOutTOP::checkParameterChanges(const TD::OP_Inputs* inputs) {
    const bool active = Parameters::evalActive(inputs);
    const char* ipcmemnameRaw = Parameters::evalIpcmemname(inputs);
    const std::string ipcmemname = ipcmemnameRaw ? ipcmemnameRaw : "";
    const int numslots = Parameters::evalNumslots(inputs);

    // Cached once per cook so reallocate()/teardown() can check it without needing
    // OP_Inputs threaded through (they run from execute() and from the destructor).
    myDebugEnabled = Parameters::evalDebug(inputs);

    // D7: Numslots is only meant to be edited while Active=Off; keep it disabled while
    // running so a live change can't be attempted through the UI (it would otherwise
    // silently do nothing until the next Active toggle).
    Parameters::setNumslotsEnabled(inputs, !active);

    if (myFirstCook) {
        myCachedActive = active;
        myCachedIpcmemname = ipcmemname;
        myCachedNumslots = numslots;
        myFirstCook = false;
        return;
    }

    // Active edge (either direction), or Ipcmemname changed while running: full
    // teardown (publishes shutdown first if a session was open) so the next cook starts
    // a fresh connection under whatever the current parameters say (mirrors
    // CudaLinkInTOP::checkParameterChanges' identical two-branch structure).
    if (active != myCachedActive) {
        teardown();
    } else if (ipcmemname != myCachedIpcmemname) {
        teardown();
    }
    // Numslots changes are picked up naturally by execute()'s needsAlloc check (D5 step
    // 3) without a full teardown -- no producer-exit signal is warranted for a live
    // format/geometry/slot-count change (the version bump alone tells the receiver to
    // reopen, D6's VERSION_CHANGED path).

    myCachedActive = active;
    myCachedIpcmemname = ipcmemname;
    myCachedNumslots = numslots;
}

// ---------------------------------------------------------------------------
// Debug logging (opt-in, Debug parameter) -- live-test finding: the transient error
// badge alone wasn't enough to diagnose what was actually failing during a
// resolution/format switch.
// ---------------------------------------------------------------------------

void CudaLinkOutTOP::debugLog(const std::string& msg) {
    if (!myDebugEnabled) return;
    if (!myDebugLogFile.is_open()) {
        char tempPath[MAX_PATH] = {};
        GetTempPathA(MAX_PATH, tempPath);
        myDebugLogFile.open(std::string(tempPath) + "cudalink_out_top_debug.log", std::ios::app);
        if (!myDebugLogFile.is_open()) return;
    }
    // Milliseconds since epoch as an integer -- printing now_seconds() (a double) via
    // std::ofstream << used the default 6-significant-figure precision, which for a
    // ~1.78e9-second epoch value showed identical "1.78324e+09" on every line (no
    // sub-second resolution at all). An integer millisecond count has no such precision
    // loss and is trivial to diff between lines.
    const int64_t ms = static_cast<int64_t>(now_seconds() * 1000.0);
    myDebugLogFile << "[t=" << ms << " frame=" << myFrameCount << "] " << msg << "\n";
    myDebugLogFile.flush();
}

// ---------------------------------------------------------------------------
// Teardown -- full stop (producer exit signal, then free everything)
// ---------------------------------------------------------------------------

void CudaLinkOutTOP::teardown() {
    debugLog("teardown: begin (allocated=" + std::string(myAllocated ? "true" : "false") + ")");
    if (myAllocated && myShmView) {
        // Signal producer exit before freeing anything it references (D6's
        // teardown-ordering note: using an imported IPC event after the exporter
        // destroys the original is undefined behavior -- the shutdown flag must already
        // be visible to the receiver before that happens).
        set_shutdown(myShmView, myLayout);
        if (myDoorbellHandle) {
            SetEvent(asHandle(myDoorbellHandle));
        }
        // Zero the IPC handle bytes so a receiver that reconnects to a stale mapping
        // (name reuse, D5's CreateFileMappingW note) can't attempt to open handles this
        // process is about to invalidate. Mirrors exporter.py::_do_cleanup step 1.
        for (uint32_t slot = 0; slot < myLayout.num_slots(); ++slot) {
            std::memset(myShmView + myLayout.slot_offset(slot), 0, cudalink::core::SLOT_SIZE);
        }
        // Grace period for the receiver to close its imported IPC handles before this
        // process frees the buffers/events they reference -- same 100ms safety margin
        // exporter.py::_do_cleanup uses for the identical UB class (CUDA IPC docs: using
        // an imported handle after the exporter frees the original is undefined).
        Sleep(100);
    }

    for (auto evt : mySlotEvents) {
        if (evt) {
            cudaEventDestroy(evt);
        }
    }
    mySlotEvents.clear();
    for (auto* devPtr : mySlotDevPtrs) {
        if (devPtr) {
            cudaFree(devPtr);
        }
    }
    mySlotDevPtrs.clear();

    if (myShmView) {
        UnmapViewOfFile(myShmView);
        myShmView = nullptr;
    }
    if (myShmHandle) {
        CloseHandle(asHandle(myShmHandle));
        myShmHandle = nullptr;
    }
    if (myDoorbellHandle) {
        CloseHandle(asHandle(myDoorbellHandle));
        myDoorbellHandle = nullptr;
    }

    myLayout = SHMLayout(0);
    myAllocated = false;
    myWidth = 0;
    myHeight = 0;
    myWriteIdx = 0;
    myStatus = "Idle";
    debugLog("teardown: complete");
}

// ---------------------------------------------------------------------------
// (Re)allocation -- D5 step 3
// ---------------------------------------------------------------------------

bool CudaLinkOutTOP::reallocate(uint32_t width, uint32_t height, const WireFormat& fmt, int numSlots,
                                 const char* name) {
    const auto reallocStart = std::chrono::steady_clock::now();
    debugLog("reallocate: begin " + std::to_string(width) + "x" + std::to_string(height) +
              " numSlots=" + std::to_string(numSlots) + " formatKind=" + std::to_string(fmt.format_kind) +
              " bits=" + std::to_string(fmt.bits_per_comp) + " flags=" + std::to_string(fmt.flags));

    // Build the entire NEW session into LOCAL storage first -- the live mySlotDevPtrs/
    // mySlotEvents/myShmView/myShmHandle members are left completely untouched until the
    // new session is fully written and committed below. This fixes a real, live-observed
    // race: the previous implementation freed the OLD device resources and bumped
    // 'version' on the wire BEFORE the new IPC handles/metadata were written, so a
    // receiver polling in that window could see VERSION_CHANGED and try to open handles
    // to memory that was stale or already freed (transient red error badge that
    // self-healed one poll later). On any failure below, only these locals are cleaned
    // up and the TOP keeps running its previous (still-valid, untouched) session.
    std::vector<void*> newDevPtrs(numSlots, nullptr);
    std::vector<cudaEvent_t> newEvents(numSlots, nullptr);
    auto cleanupNew = [&]() {
        for (auto evt : newEvents) {
            if (evt) cudaEventDestroy(evt);
        }
        for (auto* p : newDevPtrs) {
            if (p) cudaFree(p);
        }
    };

    const SHMLayout layout(static_cast<uint32_t>(numSlots));
    const size_t itemsize = fmt.bits_per_comp / 8;
    const size_t bufferSize = static_cast<size_t>(width) * height * fmt.num_comps * itemsize;

    for (int slot = 0; slot < numSlots; ++slot) {
        cudaError_t st = cudaMalloc(&newDevPtrs[slot], bufferSize);
        if (st != cudaSuccess) {
            myError = std::string("cudaMalloc failed: ") + cudaGetErrorString(st);
            cleanupNew();
            return false;
        }
        st = cudaEventCreateWithFlags(&newEvents[slot], cudaEventDisableTiming | cudaEventInterprocess);
        if (st != cudaSuccess) {
            myError = std::string("cudaEventCreateWithFlags failed: ") + cudaGetErrorString(st);
            cleanupNew();
            return false;
        }
    }

    // Reuse the existing SHM mapping when one is already open -- total_size() depends
    // only on num_slots, and Numslots is disabled while Active (D7's
    // setNumslotsEnabled(!active) in checkParameterChanges), so a live reallocate (this
    // path) never actually needs a different-sized mapping; only a true first connect
    // needs to create one.
    const bool needNewMapping = (myShmView == nullptr);
    HANDLE mapping = needNewMapping ? nullptr : asHandle(myShmHandle);
    uint8_t* view = needNewMapping ? nullptr : myShmView;
    bool preExisting = false;

    if (needNewMapping) {
        const std::wstring wname = widen(name);
        LARGE_INTEGER sz;
        sz.QuadPart = static_cast<LONGLONG>(layout.total_size());
        mapping =
            CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, sz.HighPart, sz.LowPart, wname.c_str());
        if (!mapping) {
            myError = "CreateFileMappingW failed";
            cleanupNew();
            return false;
        }
        // CreateFileMappingW returns a handle to an *existing* section (rather than
        // failing) when a previous/crashed session left one under this name (D5's SHM
        // naming interop note). If so, adopt its version so our upcoming commit is still
        // guaranteed to look different to any receiver that had seen that stale session.
        preExisting = (GetLastError() == ERROR_ALREADY_EXISTS);

        void* v = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (!v) {
            CloseHandle(mapping);
            myError = "MapViewOfFile failed";
            cleanupNew();
            return false;
        }
        view = static_cast<uint8_t*>(v);
    }

    if (preExisting) {
        const uint64_t priorVersion = read_version(view);
        if (priorVersion > myVersion) {
            myVersion = priorVersion;
        }
    }

    // Write everything the new session needs BEFORE bumping version below. While this
    // executes, 'version' on the wire (when reusing the mapping) still names the OLD
    // session, so a concurrently-polling receiver correctly reports no change yet -- the
    // C3 fix (ring_writer::commit_version's release store is the only thing that makes
    // this visible).
    write_init_header(view, layout, static_cast<uint32_t>(numSlots));
    for (int slot = 0; slot < numSlots; ++slot) {
        cudaIpcMemHandle_t memHandle;
        cudaError_t st = cudaIpcGetMemHandle(&memHandle, newDevPtrs[slot]);
        if (st != cudaSuccess) {
            myError = std::string("cudaIpcGetMemHandle failed: ") + cudaGetErrorString(st);
            cleanupNew();
            if (needNewMapping) {
                UnmapViewOfFile(view);
                CloseHandle(mapping);
            }
            return false;
        }
        cudaIpcEventHandle_t eventHandle;
        st = cudaIpcGetEventHandle(&eventHandle, newEvents[slot]);
        if (st != cudaSuccess) {
            myError = std::string("cudaIpcGetEventHandle failed: ") + cudaGetErrorString(st);
            cleanupNew();
            if (needNewMapping) {
                UnmapViewOfFile(view);
                CloseHandle(mapping);
            }
            return false;
        }
        write_slot_handle(view, layout, static_cast<uint32_t>(slot), reinterpret_cast<const uint8_t*>(&memHandle),
                          reinterpret_cast<const uint8_t*>(&eventHandle));
    }

    Metadata newMetadata{width,
                         height,
                         fmt.num_comps,
                         fmt.format_kind,
                         fmt.bits_per_comp,
                         fmt.flags,
                         static_cast<uint32_t>(bufferSize)};
    newMetadata.pack_into(view, layout);
    clear_shutdown(view, layout);

    if (!myDoorbellHandle) {
        const std::wstring wDoorbell = L"Local\\cudalink_db_" + widen(name);
        HANDLE db = CreateEventW(nullptr, FALSE, FALSE, wDoorbell.c_str());
        if (!db) {
            myWarning = "doorbell CreateEventW failed; consumer will fall back to poll-sleep";
        }
        myDoorbellHandle = db; // may remain null on failure -- SetEvent calls check for null (best-effort)
    }

    // Commit: the moment a receiver can observe VERSION_CHANGED (ring_reader's atomic
    // acquire load pairing with this release store), everything written above is already
    // guaranteed durable and visible -- this is the fix's load-bearing line.
    debugLog("reallocate: writes complete, committing version=" + std::to_string(myVersion + 1));
    commit_version(view, layout, ++myVersion);
    if (myDoorbellHandle) {
        SetEvent(asHandle(myDoorbellHandle));
    }

    // Only now swap the new resources into the live members and let the OLD ones go --
    // after a grace period, mirroring teardown()'s identical rationale: give a receiver
    // that was mid-cook against the old session (hasn't yet polled since the commit
    // above) one poll cycle (~16ms, well under 100ms) to notice VERSION_CHANGED and
    // relinquish its old imported handles before this process destroys what they refer to.
    std::vector<void*> oldDevPtrs = std::move(mySlotDevPtrs);
    std::vector<cudaEvent_t> oldEvents = std::move(mySlotEvents);
    mySlotDevPtrs = std::move(newDevPtrs);
    mySlotEvents = std::move(newEvents);
    myShmView = view;
    myShmHandle = mapping;
    myLayout = layout;
    myMetadata = newMetadata;
    myWidth = width;
    myHeight = height;
    myWriteIdx = 0;
    myAllocated = true;

    Sleep(100);
    for (auto evt : oldEvents) {
        if (evt) cudaEventDestroy(evt);
    }
    for (auto* p : oldDevPtrs) {
        if (p) cudaFree(p);
    }

    debugLog("reallocate: complete, elapsed=" +
              std::to_string(
                  std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - reallocStart).count()) +
              "ms");
    return true;
}

// ---------------------------------------------------------------------------
// execute() -- D5
// ---------------------------------------------------------------------------

// Live-tested: confirmed the .tox's own Sender-mode Script TOP also produces no output of
// its own (side-effects into SHM via cudaMemory(), never writes back) -- so a Custom TOP
// that skips output->createCUDAArray() entirely is byte-faithful to the reference and
// shows black/blank without erroring, as expected. Per explicit request, this TOP ALSO
// creates a pass-through output below (mirroring its input) so it can sit inline in an
// operator chain instead of being a dead end -- a deliberate UX improvement over the
// .tox, not a correctness requirement.
void CudaLinkOutTOP::execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void*) {
    const auto cookStart = std::chrono::steady_clock::now();
    try {
        myError.clear();
        myWarning.clear();

        // Step 0/1: cached-diff parameter-change detection + Active gate.
        checkParameterChanges(inputs);
        if (!Parameters::evalActive(inputs)) {
            myStatus = "Idle";
            return;
        }

        // Step 2: read the input TOP + map its pixel format.
        const TD::OP_TOPInput* in = inputs->getInputTOP(0);
        if (!in) {
            myWarning = "no input connected";
            return;
        }
        const WireFormat fmt = mapFromPixelFormat(in->textureDesc.pixelFormat);
        if (!fmt.supported) {
            myWarning = "unsupported input pixel format (id=" + std::to_string(static_cast<int>(in->textureDesc.pixelFormat)) +
                        ")";
            return;
        }
        const uint32_t width = in->textureDesc.width;
        const uint32_t height = in->textureDesc.height;
        if (width == 0 || height == 0) {
            myWarning = "input has zero-sized geometry";
            return;
        }
        const int numSlots = Parameters::evalNumslots(inputs);

        // Step 3: (re)allocate on first cook / resolution / format / numslots change.
        const bool needsAlloc = !myAllocated || width != myWidth || height != myHeight ||
                                 numSlots != static_cast<int>(myLayout.num_slots()) ||
                                 fmt.format_kind != myMetadata.format_kind ||
                                 fmt.bits_per_comp != myMetadata.bits_per_comp || fmt.flags != myMetadata.flags ||
                                 fmt.num_comps != myMetadata.num_comps;
        if (needsAlloc) {
            const char* name = Parameters::evalIpcmemname(inputs);
            if (!name || !*name) {
                myError = "Ipcmemname is empty";
                return;
            }
            if (!reallocate(width, height, fmt, numSlots, name)) {
                return; // myError already set
            }
        }

        // Step 4: acquire the input's cudaArray (must be called before
        // beginCUDAOperations(); the returned pointer isn't valid until after).
        TD::OP_CUDAAcquireInfo acquireInfo;
        acquireInfo.stream = myStream;
        const TD::OP_CUDAArrayInfo* arr = in->getCUDAArray(acquireInfo, nullptr);
        if (!arr) {
            myError = "getCUDAArray failed";
            return;
        }

        // Pass-through output array must ALSO be created before beginCUDAOperations() --
        // same "cudaArray* null until beginCUDAOperations()" rule as the input side
        // (TOP_CPlusPlusBase.h:349-352; PLAN-001's gotchas checklist), confirmed by the
        // receiver's own working code (CudaLinkInTOP.cpp), which creates its output array
        // before its begin/end bracket too. An earlier version of this code created it
        // AFTER beginCUDAOperations() had already run, so selfOut->cudaArray was never
        // populated -- passing that null array handle into cudaMemcpy2DToArrayAsync
        // produced cudaErrorInvalidResourceHandle ("pass-through output copy failed:
        // invalid resource handle"), confirmed live.
        TD::TOP_CUDAOutputInfo outInfo;
        outInfo.textureDesc = in->textureDesc;
        outInfo.stream = myStream;
        const TD::OP_CUDAArrayInfo* selfOut = output->createCUDAArray(outInfo, nullptr);

        // Step 5: GPU-side copy + blocking sync (source-lifetime safety, D5), inside the
        // begin/end CUDA-operations bracket.
        if (!myContext->beginCUDAOperations(nullptr)) {
            myError = "beginCUDAOperations failed";
            return;
        }

        const auto copyStart = std::chrono::steady_clock::now();
        const uint32_t slot = myWriteIdx % myLayout.num_slots(); // next slot to (over)write
        const size_t itemsize = myMetadata.bits_per_comp / 8;
        const size_t rowBytes = static_cast<size_t>(myMetadata.width) * myMetadata.num_comps * itemsize;
        CUDALINK_CUDA_CHECK(cudaMemcpy2DFromArrayAsync(mySlotDevPtrs[slot], rowBytes, arr->cudaArray, 0, 0, rowBytes,
                                                        myMetadata.height, cudaMemcpyDeviceToDevice, myStream),
                            myError);
        // Record the IPC-ready event right after the IPC-slot copy, before the
        // pass-through copy below, so the receiver's cudaStreamWaitEvent isn't delayed by
        // work this TOP does purely for its own on-screen display.
        CUDALINK_CUDA_CHECK(cudaEventRecord(mySlotEvents[slot], myStream), myError);

        // Pass-through output (this TOP's own display, per explicit request -- see the
        // comment above execute()). Non-fatal on failure -- the IPC export above is this
        // TOP's real job and must not be aborted just because the optional display copy
        // failed. 'selfOut' was created above, before beginCUDAOperations() (see that
        // comment for why the ordering matters).
        //
        // Deliberately does NOT copy directly from 'arr->cudaArray' (source) to
        // 'selfOut->cudaArray' (dest) -- that's an array-to-array copy, which requires
        // cudaMemcpy3DAsync with srcArray/dstArray and returned cudaErrorInvalidValue in
        // live testing (CUDA docs, cuda_runtime_api.h: array-to-array copies require
        // matching "element size" and array-specific depth/extent semantics that are easy
        // to get subtly wrong for a plain 2D array; cudaMemcpy2DArrayToArray would avoid
        // that ambiguity but has no stream parameter, forcing the legacy default stream --
        // a real perf risk here given the <=90us budget, since the default stream
        // implicitly synchronizes with other streams unless per-thread-default-stream
        // compilation is used, which this project doesn't do).
        //
        // Instead, reuse 'mySlotDevPtrs[slot]' as the source: it already holds a
        // byte-identical LINEAR copy of this frame from the IPC-export copy just above,
        // so this is the exact same cudaMemcpy2DToArrayAsync(dst_array, ..., linear_src,
        // ...) shape the receiver (CudaLinkInTOP.cpp) already uses successfully in
        // production. Same-stream ordering guarantees the IPC-slot write above completes
        // before this read -- no extra event/wait needed.
        if (selfOut) {
            cudaError_t passThroughStatus = cudaMemcpy2DToArrayAsync(
                selfOut->cudaArray, 0, 0, mySlotDevPtrs[slot], rowBytes, rowBytes, myMetadata.height,
                cudaMemcpyDeviceToDevice, myStream);
            if (passThroughStatus != cudaSuccess) {
                myWarning = std::string("pass-through output copy failed: ") + cudaGetErrorString(passThroughStatus);
            }
        } else {
            myWarning = "createCUDAArray failed (pass-through output unavailable this cook)";
        }

        // Blocking by design (D5 "Blocking by default"): the SDK guarantees the input
        // array's validity only until execute() returns, and nothing documents TD
        // synchronizing the user stream at endCUDAOperations() -- the same
        // source-lifetime race that forced the Python TD sender to a blocking export in
        // v1.10.1 (CUDA 719). Correctness first; both copies together are well inside
        // budget.
        CUDALINK_CUDA_CHECK(cudaStreamSynchronize(myStream), myError);
        myContext->endCUDAOperations(nullptr);
        myCopyUs = std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - copyStart).count();

        // Step 6: publish + doorbell.
        ++myWriteIdx;
        publish(myShmView, myLayout, myWriteIdx, now_seconds());
        if (myDoorbellHandle) {
            SetEvent(asHandle(myDoorbellHandle));
        }

        // Step 7 (stats recorded below, shared with the error path's cookUs update).
        ++myFrameCount;
        myStatus = std::to_string(width) + "x" + std::to_string(height);
    } catch (...) {
        // No exception ever crosses the ABI (D7 error policy).
        myError = "unexpected exception in execute()";
    }
    myCookUs = std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - cookStart).count();

    // Periodic bench log (Debug-gated, every 97 frames -- matches exporter.py's
    // FrameProfile cadence): gives an offline avg cook_us/copy_us number to compare
    // against PLAN-001's baseline table without needing to eyeball the Info CHOP live.
    if (myDebugEnabled) {
        mySumCookUs += myCookUs;
        mySumCopyUs += myCopyUs;
        if (++myBenchSamples >= 97) {
            debugLog("bench: avg_cook_us=" + std::to_string(mySumCookUs / myBenchSamples) +
                      " avg_copy_us=" + std::to_string(mySumCopyUs / myBenchSamples) +
                      " samples=" + std::to_string(myBenchSamples) + " frames=" + std::to_string(myFrameCount));
            mySumCookUs = 0.0f;
            mySumCopyUs = 0.0f;
            myBenchSamples = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Info CHOP / DAT / status (D7)
// ---------------------------------------------------------------------------

int32_t CudaLinkOutTOP::getNumInfoCHOPChans(void*) {
    return 5; // frames, cook_us, copy_us, write_idx, num_slots
}

void CudaLinkOutTOP::getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan* chan, void*) {
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
            chan->value = static_cast<float>(myWriteIdx);
            break;
        case 4:
            chan->name->setString("num_slots");
            chan->value = static_cast<float>(myLayout.num_slots());
            break;
        default:
            break;
    }
}

bool CudaLinkOutTOP::getInfoDATSize(TD::OP_InfoDATSize* infoSize, void*) {
    infoSize->rows = 4; // ipc_version, status, last_error, last_error_frame
    infoSize->cols = 2;
    infoSize->byColumn = false;
    return true;
}

void CudaLinkOutTOP::getInfoDATEntries(int32_t index, int32_t, TD::OP_InfoDATEntries* entries, void*) {
    if (index == 0) {
        entries->values[0]->setString("ipc_version");
        entries->values[1]->setString(std::to_string(myVersion).c_str());
    } else if (index == 1) {
        entries->values[0]->setString("status");
        entries->values[1]->setString(myStatus.c_str());
    } else if (index == 2) {
        // Sticky (never auto-cleared, only overwritten) -- see getErrorString(). Survives
        // across cooks, unlike the transient error badge (live-test finding).
        entries->values[0]->setString("last_error");
        entries->values[1]->setString(myLastError.c_str());
    } else if (index == 3) {
        entries->values[0]->setString("last_error_frame");
        entries->values[1]->setString(std::to_string(myLastErrorFrame).c_str());
    }
}

void CudaLinkOutTOP::getErrorString(TD::OP_String* error, void*) {
    if (!myError.empty()) {
        myLastError = myError;
        myLastErrorFrame = myFrameCount;
        debugLog("ERROR: " + myError);
    }
    error->setString(myError.c_str());
    myError.clear();
}

void CudaLinkOutTOP::getWarningString(TD::OP_String* warning, void*) {
    if (!myWarning.empty()) {
        debugLog("WARNING: " + myWarning);
    }
    warning->setString(myWarning.c_str());
    myWarning.clear();
}

void CudaLinkOutTOP::getInfoPopupString(TD::OP_String* info, void*) { info->setString(myStatus.c_str()); }
