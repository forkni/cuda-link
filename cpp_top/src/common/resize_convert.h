// resize_convert.h -- GPU resize + pixel-format conversion between two linear (pitched)
// device buffers, for the Common page (Output Resolution, Pixel Format, Input Smoothness).
//
// One thread per destination texel: map the texel centre into the source, sample the source
// (nearest, or bilinear with clamp-to-edge) after decoding it to float4 with the shared
// pixel codec (core/pixel_codec.h), then encode to the destination format. Equal sizes with
// bilinear sampling give a pure per-texel conversion (the sample lands on texel centres).
//
// Both TOPs use it only off the fast path: when the Common page asks for the input's (Out)
// or the wire's (In) own size and format, they keep their plain copy.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "pixel_codec.h"

namespace cudalink::common {

struct LinearImage {
    const void* data = nullptr; // source; for the destination see LinearImageMut
    size_t pitch = 0;           // bytes per row
    uint32_t width = 0;
    uint32_t height = 0;
    cudalink::core::PixelCodec codec;
};

struct LinearImageMut {
    void* data = nullptr;
    size_t pitch = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    cudalink::core::PixelCodec codec;
};

// Enqueues the kernel on 'stream'. Returns cudaErrorInvalidValue for an invalid codec or an
// empty image, else the launch status (cudaGetLastError right after the launch).
cudaError_t launchResizeConvert(const LinearImage& src, const LinearImageMut& dst, bool bilinear,
                                cudaStream_t stream);

} // namespace cudalink::common
