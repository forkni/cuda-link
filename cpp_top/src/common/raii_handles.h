// Move-only RAII guards for the OS/CUDA resources CudaLinkInTOP/CudaLinkOutTOP own outside the
// begin/endCUDAOperations() bracket (device buffers, IPC events, the Win32 SHM mapping+view).
//
// reallocate() (CudaLinkOutTOP.cpp) and openSHM()/openSlotHandlesIfNeeded() (CudaLinkInTOP.cpp)
// build these resources across several fallible steps; without RAII, every early-return site in
// those functions has to hand-duplicate the matching cudaFree/cudaEventDestroy/
// UnmapViewOfFile+CloseHandle cleanup. These guards make that cleanup automatic
// (destructor-driven) and make partial-failure leaks (e.g. a mid-loop CUDA error leaving
// earlier-opened slots' resources dangling) structurally impossible instead of dependent on
// every call site remembering to clean up. release()/reset() let a guard hand ownership over to
// a long-lived class member once a session is fully built and committed -- callers are expected
// to call release() exactly once, on the success path.

#pragma once

#include <cuda_runtime.h>
#include <windows.h>

namespace cudalink::common {

// Owns a single cudaMalloc'd device buffer; frees it with cudaFree() unless released. For an
// imported IPC pointer (cudaIpcOpenMemHandle), use CudaIpcMemGuard below instead -- the two
// teardown calls are not interchangeable.
class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    ~CudaDeviceBuffer() { reset(); }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept : myPtr(other.release()) {}
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    // Takes ownership of a pointer obtained from a CUDA allocation call (e.g. cudaMalloc),
    // freeing whatever this guard previously held.
    void reset(void* ptr = nullptr) {
        if (myPtr) {
            cudaFree(myPtr);
        }
        myPtr = ptr;
    }

    // Releases ownership without freeing -- for handing the pointer to a long-lived member
    // (e.g. mySlotDevPtrs) once the session it belongs to has been committed.
    [[nodiscard]] void* release() noexcept {
        void* p = myPtr;
        myPtr = nullptr;
        return p;
    }

    [[nodiscard]] void* get() const noexcept { return myPtr; }
    explicit operator bool() const noexcept { return myPtr != nullptr; }

private:
    void* myPtr = nullptr;
};

// Owns a single imported IPC device pointer (from cudaIpcOpenMemHandle); closes it with
// cudaIpcCloseMemHandle() unless released. NOT interchangeable with CudaDeviceBuffer above --
// an imported pointer must be torn down with cudaIpcCloseMemHandle, never cudaFree (the two
// calls are not equivalent; CudaLinkInTOP::closeHandles() already does this correctly for the
// long-lived member vector, which this guard mirrors for the local build-up phase).
class CudaIpcMemGuard {
public:
    CudaIpcMemGuard() = default;
    ~CudaIpcMemGuard() { reset(); }

    CudaIpcMemGuard(const CudaIpcMemGuard&) = delete;
    CudaIpcMemGuard& operator=(const CudaIpcMemGuard&) = delete;
    CudaIpcMemGuard(CudaIpcMemGuard&& other) noexcept : myPtr(other.release()) {}
    CudaIpcMemGuard& operator=(CudaIpcMemGuard&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    void reset(void* ptr = nullptr) {
        if (myPtr) {
            cudaIpcCloseMemHandle(myPtr);
        }
        myPtr = ptr;
    }

    [[nodiscard]] void* release() noexcept {
        void* p = myPtr;
        myPtr = nullptr;
        return p;
    }

    [[nodiscard]] void* get() const noexcept { return myPtr; }
    explicit operator bool() const noexcept { return myPtr != nullptr; }

private:
    void* myPtr = nullptr;
};

// Owns a single CUDA event (created via cudaEventCreateWithFlags/cudaIpcOpenEventHandle);
// destroys it with cudaEventDestroy() unless released.
class CudaEventGuard {
public:
    CudaEventGuard() = default;
    ~CudaEventGuard() { reset(); }

    CudaEventGuard(const CudaEventGuard&) = delete;
    CudaEventGuard& operator=(const CudaEventGuard&) = delete;
    CudaEventGuard(CudaEventGuard&& other) noexcept : myEvent(other.release()) {}
    CudaEventGuard& operator=(CudaEventGuard&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    void reset(cudaEvent_t evt = nullptr) {
        if (myEvent) {
            cudaEventDestroy(myEvent);
        }
        myEvent = evt;
    }

    [[nodiscard]] cudaEvent_t release() noexcept {
        cudaEvent_t e = myEvent;
        myEvent = nullptr;
        return e;
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return myEvent; }
    explicit operator bool() const noexcept { return myEvent != nullptr; }

private:
    cudaEvent_t myEvent = nullptr;
};

// Owns a single instantiated CUDA graph exec (from cudaGraphInstantiate); destroys it with
// cudaGraphExecDestroy() unless released. Destroying an exec only frees the exec's own
// graph/node bookkeeping -- it never dereferences the memory operands baked into its nodes,
// so a guard may safely outlive (and destroy after) the buffers/arrays its graph copies
// between.
class CudaGraphExecGuard {
public:
    CudaGraphExecGuard() = default;
    ~CudaGraphExecGuard() { reset(); }

    CudaGraphExecGuard(const CudaGraphExecGuard&) = delete;
    CudaGraphExecGuard& operator=(const CudaGraphExecGuard&) = delete;
    CudaGraphExecGuard(CudaGraphExecGuard&& other) noexcept : myExec(other.release()) {}
    CudaGraphExecGuard& operator=(CudaGraphExecGuard&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    void reset(cudaGraphExec_t exec = nullptr) {
        if (myExec) {
            cudaGraphExecDestroy(myExec);
        }
        myExec = exec;
    }

    [[nodiscard]] cudaGraphExec_t release() noexcept {
        cudaGraphExec_t e = myExec;
        myExec = nullptr;
        return e;
    }

    [[nodiscard]] cudaGraphExec_t get() const noexcept { return myExec; }
    explicit operator bool() const noexcept { return myExec != nullptr; }

private:
    cudaGraphExec_t myExec = nullptr;
};

// Owns a paired Win32 file-mapping HANDLE + its MapViewOfFile() pointer. Both halves are always
// torn down together (UnmapViewOfFile before CloseHandle) since a mapping is useless once its
// view is gone and vice versa -- callers never need one without the other.
class ShmViewGuard {
public:
    ShmViewGuard() = default;
    ShmViewGuard(HANDLE mapping, uint8_t* view) : myMapping(mapping), myView(view) {}
    ~ShmViewGuard() { reset(); }

    ShmViewGuard(const ShmViewGuard&) = delete;
    ShmViewGuard& operator=(const ShmViewGuard&) = delete;
    ShmViewGuard(ShmViewGuard&& other) noexcept : myMapping(other.myMapping), myView(other.myView) {
        other.myMapping = nullptr;
        other.myView = nullptr;
    }
    ShmViewGuard& operator=(ShmViewGuard&& other) noexcept {
        if (this != &other) {
            reset();
            myMapping = other.myMapping;
            myView = other.myView;
            other.myMapping = nullptr;
            other.myView = nullptr;
        }
        return *this;
    }

    void reset() {
        if (myView) {
            UnmapViewOfFile(myView);
        }
        if (myMapping) {
            CloseHandle(myMapping);
        }
        myMapping = nullptr;
        myView = nullptr;
    }

    // Releases ownership of both halves together without tearing them down -- for handing the
    // pair to myShmHandle/myShmView once the session is committed.
    void release() noexcept {
        myMapping = nullptr;
        myView = nullptr;
    }

    [[nodiscard]] HANDLE mapping() const noexcept { return myMapping; }
    [[nodiscard]] uint8_t* view() const noexcept { return myView; }
    explicit operator bool() const noexcept { return myView != nullptr; }

private:
    HANDLE myMapping = nullptr;
    uint8_t* myView = nullptr;
};

} // namespace cudalink::common
