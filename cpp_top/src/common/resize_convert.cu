// resize_convert.cu -- see resize_convert.h.
#include "resize_convert.h"

namespace cudalink::common {

namespace {

using cudalink::core::decode_texel;
using cudalink::core::encode_texel;
using cudalink::core::Float4;
using cudalink::core::lerp4;
using cudalink::core::PixelCodec;

__device__ inline Float4 fetch(const uint8_t* base, size_t pitch, uint32_t bpt, const PixelCodec& c, int x,
                               int y) {
    return decode_texel(base + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * bpt, c);
}

__global__ void resizeConvertKernel(const uint8_t* src, size_t srcPitch, uint32_t sw, uint32_t sh,
                                    PixelCodec sc, uint8_t* dst, size_t dstPitch, uint32_t dw, uint32_t dh,
                                    PixelCodec dc, bool bilinear) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;

    const uint32_t sbpt = sc.bytes_per_texel();
    const float sx = (static_cast<float>(x) + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw);
    const float sy = (static_cast<float>(y) + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh);
    const int maxX = static_cast<int>(sw) - 1;
    const int maxY = static_cast<int>(sh) - 1;

    Float4 v;
    if (!bilinear) {
        // ceil - 1 equals floor except on an exact texel boundary, where it takes the
        // lower texel: TD's own nearest sampling does that for most texels (live oracle,
        // docs/CPP_TOP.md); the rest of its tie picks follow GPU rasterizer rounding.
        const int xi = max(0, min(static_cast<int>(ceilf(sx)) - 1, maxX));
        const int yi = max(0, min(static_cast<int>(ceilf(sy)) - 1, maxY));
        v = fetch(src, srcPitch, sbpt, sc, xi, yi);
    } else {
        const float u = sx - 0.5f;
        const float w = sy - 0.5f;
        const float fx = floorf(u);
        const float fy = floorf(w);
        const float tx = u - fx;
        const float ty = w - fy;
        const int x0 = max(0, min(static_cast<int>(fx), maxX));
        const int y0 = max(0, min(static_cast<int>(fy), maxY));
        const int x1 = max(0, min(static_cast<int>(fx) + 1, maxX));
        const int y1 = max(0, min(static_cast<int>(fy) + 1, maxY));
        const Float4 top =
            lerp4(fetch(src, srcPitch, sbpt, sc, x0, y0), fetch(src, srcPitch, sbpt, sc, x1, y0), tx);
        const Float4 bot =
            lerp4(fetch(src, srcPitch, sbpt, sc, x0, y1), fetch(src, srcPitch, sbpt, sc, x1, y1), tx);
        v = lerp4(top, bot, ty);
    }
    encode_texel(dst + static_cast<size_t>(y) * dstPitch + static_cast<size_t>(x) * dc.bytes_per_texel(), dc,
                 v);
}

} // namespace

cudaError_t launchResizeConvert(const LinearImage& src, const LinearImageMut& dst, bool bilinear,
                                cudaStream_t stream) {
    if (!src.codec.valid || !dst.codec.valid || src.data == nullptr || dst.data == nullptr ||
        src.width == 0 || src.height == 0 || dst.width == 0 || dst.height == 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(16, 16);
    const dim3 grid((dst.width + block.x - 1) / block.x, (dst.height + block.y - 1) / block.y);
    resizeConvertKernel<<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(src.data), src.pitch, src.width, src.height, src.codec,
        static_cast<uint8_t*>(dst.data), dst.pitch, dst.width, dst.height, dst.codec, bilinear);
    return cudaGetLastError();
}

} // namespace cudalink::common
