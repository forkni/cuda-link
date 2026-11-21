// common_page_td.h -- the TD-side glue both TOPs share for the Common page: pixel-format
// names for warnings and errors, the Input Smoothness read, and a grow-only scratch buffer
// for the resize/convert path (resize_convert.h).
#pragma once

#include <cstddef>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "CPlusPlus_Common.h"
#include "raii_handles.h"

namespace cudalink::common {

// The Common page's own menu names (getParString("format")), for messages a user can match
// against the Pixel Format menu.
inline const char* pixelFormatName(TD::OP_PixelFormat f) {
    switch (f) {
        case TD::OP_PixelFormat::BGRA8Fixed:
        case TD::OP_PixelFormat::RGBA8Fixed:
            return "rgba8fixed";
        case TD::OP_PixelFormat::RGBA16Fixed:
            return "rgba16fixed";
        case TD::OP_PixelFormat::RGBA16Float:
            return "rgba16float";
        case TD::OP_PixelFormat::RGBA32Float:
            return "rgba32float";
        case TD::OP_PixelFormat::Mono8Fixed:
            return "mono8fixed";
        case TD::OP_PixelFormat::Mono16Fixed:
            return "mono16fixed";
        case TD::OP_PixelFormat::Mono16Float:
            return "mono16float";
        case TD::OP_PixelFormat::Mono32Float:
            return "mono32float";
        case TD::OP_PixelFormat::RG8Fixed:
            return "rg8fixed";
        case TD::OP_PixelFormat::RG16Fixed:
            return "rg16fixed";
        case TD::OP_PixelFormat::RG16Float:
            return "rg16float";
        case TD::OP_PixelFormat::RG32Float:
            return "rg32float";
        case TD::OP_PixelFormat::A8Fixed:
            return "a8fixed";
        case TD::OP_PixelFormat::A16Fixed:
            return "a16fixed";
        case TD::OP_PixelFormat::A16Float:
            return "a16float";
        case TD::OP_PixelFormat::A32Float:
            return "a32float";
        case TD::OP_PixelFormat::MonoA8Fixed:
            return "monoalpha8fixed";
        case TD::OP_PixelFormat::MonoA16Fixed:
            return "monoalpha16fixed";
        case TD::OP_PixelFormat::MonoA16Float:
            return "monoalpha16float";
        case TD::OP_PixelFormat::MonoA32Float:
            return "monoalpha32float";
        case TD::OP_PixelFormat::RGB10A2Fixed:
            return "rgb10a2fixed";
        case TD::OP_PixelFormat::RGB11Float:
            return "rgba11float";
        default:
            return "unknown";
    }
}

// The alpha-only formats (A8 .. A32) share the wire tuple of mono; the codec needs the
// TD format to carry alpha, not red, in their one component.
inline bool isAlphaOnly(TD::OP_PixelFormat f) {
    return f == TD::OP_PixelFormat::A8Fixed || f == TD::OP_PixelFormat::A16Fixed ||
           f == TD::OP_PixelFormat::A16Float || f == TD::OP_PixelFormat::A32Float;
}

// Input Smoothness: nearest, or linear (mipmap is treated as linear: no mip chain exists
// for a linear IPC buffer).
inline bool commonBilinear(const TD::OP_Inputs* inputs) {
    const char* v = inputs->getParString("inputfiltertype");
    return v == nullptr || std::strcmp(v, "nearest") != 0;
}

// The warning both TOPs show when the Common Pixel Format converts the stream.
inline std::string formatConvertWarning(const char* what, TD::OP_PixelFormat from, TD::OP_PixelFormat to) {
    return std::string("Common page Pixel Format converts the ") + what + " from " + pixelFormatName(from) +
           " to " + pixelFormatName(to) + " (set Pixel Format to Use Input to keep " + pixelFormatName(from) +
           ")";
}

// Device scratch for the resize/convert path. Grows, never shrinks, until release().
class ScratchBuffer {
public:
    // Returns nullptr (and 'status' set) when the allocation fails.
    void* ensure(size_t bytes, cudaError_t* status) {
        *status = cudaSuccess;
        if (bytes <= myBytes && myBuf.get() != nullptr) return myBuf.get();
        myBuf.reset();
        myBytes = 0;
        void* p = nullptr;
        *status = cudaMalloc(&p, bytes);
        if (*status != cudaSuccess) {
            cudaGetLastError(); // clear the slot: the caller reports *status, a later call must not
            return nullptr;
        }
        myBuf.reset(p);
        myBytes = bytes;
        return p;
    }
    void release() {
        myBuf.reset();
        myBytes = 0;
    }

private:
    CudaDeviceBuffer myBuf;
    size_t myBytes = 0;
};

} // namespace cudalink::common
