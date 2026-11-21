// resize_convert_tests -- GPU check of launchResizeConvert against a host reference.
//
// The host reference runs the same pixel codec (core/pixel_codec.h) and the same texel-centre
// mapping in double precision. Fixed-format outputs must match within 1 LSB, float outputs
// within a small relative error (the kernel works in float). Needs a CUDA GPU; ctest marks the
// test skipped (exit 77) when none is present.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.h"
#include "pixel_codec.h"
#include "resize_convert.h"

using namespace cudalink::core;   // NOLINT(google-build-using-namespace)
using namespace cudalink::common; // NOLINT(google-build-using-namespace)

namespace {

// Runs one CUDALINK_CUDA_CHECK_FATAL on a given status, the way execute() does.
void check_fatal(cudaError_t status, std::string& err, bool& fatal) {
    CUDALINK_CUDA_CHECK_FATAL(status, err, fatal);
}

// The hot-path latch: only process-corrupting errors latch (2026-11-21 regression: a plain
// cudaErrorInvalidValue from the Common page convert latched a new In TOP for good).
int check_latch_policy() {
    int failures = 0;
    for (cudaError_t sticky :
         {cudaErrorIllegalAddress, cudaErrorLaunchTimeout, cudaErrorAssert, cudaErrorHardwareStackError,
          cudaErrorIllegalInstruction, cudaErrorMisalignedAddress, cudaErrorInvalidAddressSpace,
          cudaErrorInvalidPc, cudaErrorLaunchFailure, cudaErrorUnknown}) {
        if (!isStickyCudaError(sticky)) ++failures;
    }
    for (cudaError_t plain : {cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidConfiguration,
                              cudaErrorInvalidResourceHandle, cudaErrorNotReady, cudaErrorMemoryAllocation}) {
        if (isStickyCudaError(plain)) ++failures;
    }

    std::string err;
    bool fatal = false;
    // A real failing runtime call, so the per-thread last-error slot is set (an enum value
    // alone never sets it). Baseline first: the call must set the slot, or the check below
    // would pass without the macro's cudaGetLastError().
    (void)cudaGetLastError();
    if (cudaMemset(nullptr, 0, 16) == cudaSuccess || cudaPeekAtLastError() == cudaSuccess) ++failures;
    (void)cudaGetLastError();
    check_fatal(cudaMemset(nullptr, 0, 16), err, fatal);
    if (fatal || err.empty() || cudaPeekAtLastError() != cudaSuccess) ++failures;
    (void)cudaGetLastError();

    err.clear();
    check_fatal(cudaErrorIllegalAddress, err, fatal);
    if (!fatal || err.find("illegal memory access") == std::string::npos) ++failures;

    err.clear();
    fatal = false;
    check_fatal(cudaSuccess, err, fatal);
    if (fatal || !err.empty()) ++failures;

    if (failures != 0) std::fprintf(stderr, "latch policy: %d failures\n", failures);
    return failures;
}

int g_failures = 0;
int g_cases = 0;

struct Fmt {
    const char* name;
    uint8_t kind, bits;
    uint16_t flags;
    uint32_t comps;
};

const Fmt kFormats[] = {
    {"bgra8", FORMAT_KIND_UNSIGNED, 8, FLAGS_BGRA, 4},
    {"rgba8", FORMAT_KIND_UNSIGNED, 8, 0, 4},
    {"mono8", FORMAT_KIND_UNSIGNED, 8, 0, 1},
    {"rgba16", FORMAT_KIND_UNSIGNED, 16, 0, 4},
    {"rgba16f", FORMAT_KIND_FLOAT, 16, 0, 4},
    {"rgba32f", FORMAT_KIND_FLOAT, 32, 0, 4},
    {"mono32f", FORMAT_KIND_FLOAT, 32, 0, 1},
    {"rg32f", FORMAT_KIND_FLOAT, 32, 0, 2},
    {"monoa16f", FORMAT_KIND_FLOAT, 16, FLAGS_MONO_ALPHA, 2},
};

Float4 host_sample(const std::vector<uint8_t>& src, size_t pitch, uint32_t sw, uint32_t sh,
                   const PixelCodec& c, uint32_t dw, uint32_t dh, uint32_t x, uint32_t y, bool bilinear) {
    const uint32_t bpt = c.bytes_per_texel();
    auto at = [&](int xi, int yi) { return decode_texel(src.data() + yi * pitch + xi * bpt, c); };
    const float sx = (static_cast<float>(x) + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw);
    const float sy = (static_cast<float>(y) + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh);
    const int maxX = static_cast<int>(sw) - 1, maxY = static_cast<int>(sh) - 1;
    if (!bilinear) {
        return at(std::max(0, std::min(static_cast<int>(std::ceil(sx)) - 1, maxX)),
                  std::max(0, std::min(static_cast<int>(std::ceil(sy)) - 1, maxY)));
    }
    const float u = sx - 0.5f, w = sy - 0.5f, fx = std::floor(u), fy = std::floor(w);
    const float tx = u - fx, ty = w - fy;
    auto cl = [](int v, int hi) { return std::max(0, std::min(v, hi)); };
    const int x0 = cl(static_cast<int>(fx), maxX), x1 = cl(static_cast<int>(fx) + 1, maxX);
    const int y0 = cl(static_cast<int>(fy), maxY), y1 = cl(static_cast<int>(fy) + 1, maxY);
    return lerp4(lerp4(at(x0, y0), at(x1, y0), tx), lerp4(at(x0, y1), at(x1, y1), tx), ty);
}

bool run_case(const Fmt& sf, const Fmt& df, uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh, bool bilinear,
              std::mt19937& rng) {
    ++g_cases;
    const PixelCodec sc = make_codec(sf.kind, sf.bits, sf.flags, sf.comps);
    const PixelCodec dc = make_codec(df.kind, df.bits, df.flags, df.comps);
    const size_t sp = static_cast<size_t>(sw) * sc.bytes_per_texel() + 12; // odd pitch on purpose
    const size_t dp = static_cast<size_t>(dw) * dc.bytes_per_texel() + 4;
    std::vector<uint8_t> src(sp * sh);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t y = 0; y < sh; ++y) {
        for (uint32_t x = 0; x < sw; ++x) {
            encode_texel(src.data() + y * sp + x * sc.bytes_per_texel(), sc,
                         Float4{dist(rng), dist(rng), dist(rng), dist(rng)});
        }
    }
    void *dSrc = nullptr, *dDst = nullptr;
    if (cudaMalloc(&dSrc, src.size()) != cudaSuccess || cudaMalloc(&dDst, dp * dh) != cudaSuccess)
        return false;
    cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice);
    cudaMemset(dDst, 0, dp * dh);
    const cudaError_t st = launchResizeConvert(LinearImage{dSrc, sp, sw, sh, sc},
                                               LinearImageMut{dDst, dp, dw, dh, dc}, bilinear, nullptr);
    std::vector<uint8_t> out(dp * dh);
    cudaMemcpy(out.data(), dDst, out.size(), cudaMemcpyDeviceToHost);
    cudaFree(dSrc);
    cudaFree(dDst);
    const std::string label = std::string(sf.name) + "->" + df.name + " " + std::to_string(sw) + "x" +
                              std::to_string(sh) + "->" + std::to_string(dw) + "x" + std::to_string(dh) +
                              (bilinear ? " linear" : " nearest");
    if (st != cudaSuccess) {
        std::fprintf(stderr, "%s: launch failed: %s\n", label.c_str(), cudaGetErrorString(st));
        ++g_failures;
        return true;
    }
    const float lsb = dc.kind == FORMAT_KIND_UNSIGNED ? 1.0f / ((1U << dc.bits) - 1U) : 0.0f;
    int bad = 0;
    for (uint32_t y = 0; y < dh && bad < 5; ++y) {
        for (uint32_t x = 0; x < dw && bad < 5; ++x) {
            uint8_t ref[16] = {};
            encode_texel(ref, dc, host_sample(src, sp, sw, sh, sc, dw, dh, x, y, bilinear));
            const Float4 r = decode_texel(ref, dc);
            const Float4 g = decode_texel(out.data() + y * dp + x * dc.bytes_per_texel(), dc);
            const float tol = dc.kind == FORMAT_KIND_UNSIGNED ? lsb * 1.01f : (dc.bits == 16 ? 2e-3f : 1e-5f);
            const float d = std::max(std::max(std::fabs(r.r - g.r), std::fabs(r.g - g.g)),
                                     std::max(std::fabs(r.b - g.b), std::fabs(r.a - g.a)));
            if (!(d <= tol)) {
                if (bad == 0) std::fprintf(stderr, "%s: mismatch at %u,%u diff %g\n", label.c_str(), x, y, d);
                ++bad;
            }
        }
    }
    if (bad) ++g_failures;
    return true;
}

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("resize_convert_tests: no CUDA device, skipped\n");
        return 77;
    }
    std::mt19937 rng(1234);
    const struct {
        uint32_t sw, sh, dw, dh;
    } sizes[] = {{64, 48, 64, 48}, {64, 48, 32, 24}, {64, 48, 8, 6}, {37, 21, 74, 42}, {64, 48, 36, 36}};
    for (const Fmt& sf : kFormats) {
        for (const Fmt& df : kFormats) {
            for (const auto& s : sizes) {
                for (bool bilinear : {false, true}) {
                    if (!run_case(sf, df, s.sw, s.sh, s.dw, s.dh, bilinear, rng)) {
                        std::fprintf(stderr, "cudaMalloc failed\n");
                        return 1;
                    }
                }
            }
        }
    }
    // Invalid input is rejected before any launch.
    LinearImage bad{};
    if (launchResizeConvert(bad, LinearImageMut{}, true, nullptr) != cudaErrorInvalidValue) ++g_failures;
    g_failures += check_latch_policy();
    std::printf("resize_convert_tests: %d cases, %d failures\n", g_cases, g_failures);
    return g_failures == 0 ? 0 : 1;
}
