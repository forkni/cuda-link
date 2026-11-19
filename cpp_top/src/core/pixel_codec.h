// pixel_codec.h -- decode one texel of any wire format to float4, encode float4 to any wire
// format. Header-only and TD-free, so the same code runs in the CUDA resize/convert kernel
// (compiled by nvcc, CUDALINK_HD = __host__ __device__) and in topcore_tests on the host.
//
// A format is the wire tuple (format_kind, bits_per_comp, flags, num_comps) from
// shm_layout.h. Channel rules (the live oracle in docs/CPP_TOP.md checks them against TD):
//   decode  4 comps: RGBA (BGRA when FLAGS_BGRA)       encode  4 comps: RGBA (or BGRA)
//           2 comps: RG -> (r, g, 0, 1)                         2 comps: RG <- (r, g)
//                    mono+alpha -> (m, m, m, a)                          mono+alpha <- (r, a)
//           1 comp:  mono -> (m, m, m, 1)                       1 comp:  mono <- r
//                    alpha -> (0, 0, 0, a)                               alpha <- a
// alpha_only is not on the wire (A* and Mono* share num_comps = 1); the TD side sets it from
// the OP_PixelFormat of a texture it reads or writes.
// Fixed formats are unsigned normalized: decode v / max, encode round(saturate(x) * max).
// 16-bit float uses IEEE half with round-to-nearest-even. bfloat16 is not supported.
#pragma once

#include <cstdint>
#include <cstring> // memcpy: unqualified, so nvcc device code can call it too

#include "shm_layout.h"

#if defined(__CUDACC__)
#define CUDALINK_HD __host__ __device__
#else
#define CUDALINK_HD
#endif

namespace cudalink::core {

struct Float4 {
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;
};

struct PixelCodec {
    uint8_t kind = FORMAT_KIND_FLOAT;
    uint8_t bits = 0;        // 8, 16 or 32
    uint8_t comps = 0;       // 1, 2 or 4
    bool bgra = false;       // 4-comp 8-bit BGRA order
    bool mono_alpha = false; // 2-comp mono + alpha, not RG
    bool alpha_only = false; // 1-comp alpha (A8 .. A32), not mono; set by the caller
    bool valid = false;

    CUDALINK_HD uint32_t bytes_per_texel() const { return static_cast<uint32_t>(comps) * (bits / 8U); }
};

inline PixelCodec make_codec(uint8_t format_kind, uint8_t bits_per_comp, uint16_t flags, uint32_t num_comps) {
    PixelCodec c;
    c.kind = format_kind;
    c.bits = bits_per_comp;
    c.comps = static_cast<uint8_t>(num_comps);
    c.bgra = (flags & FLAGS_BGRA) != 0;
    c.mono_alpha = (flags & FLAGS_MONO_ALPHA) != 0;
    const bool comps_ok = num_comps == 1 || num_comps == 2 || num_comps == 4;
    const bool kind_ok =
        (format_kind == FORMAT_KIND_UNSIGNED && (bits_per_comp == 8 || bits_per_comp == 16)) ||
        (format_kind == FORMAT_KIND_FLOAT && (bits_per_comp == 16 || bits_per_comp == 32));
    const bool flags_ok = (flags & FLAGS_BFLOAT16) == 0 &&
                          (!c.bgra || (num_comps == 4 && bits_per_comp == 8)) &&
                          (!c.mono_alpha || num_comps == 2);
    c.valid = comps_ok && kind_ok && flags_ok;
    return c;
}

inline bool same_codec(const PixelCodec& a, const PixelCodec& b) {
    return a.kind == b.kind && a.bits == b.bits && a.comps == b.comps && a.bgra == b.bgra &&
           a.mono_alpha == b.mono_alpha;
}

// --- IEEE half <-> float, integer-only so host and device give the same bits ---------------
CUDALINK_HD inline float half_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else { // subnormal: normalize
            exp = 127 - 15 + 1;
            while ((mant & 0x400U) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFU;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000U | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

CUDALINK_HD inline uint16_t float_to_half_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000U;
    const uint32_t absx = x & 0x7FFFFFFFU;
    if (absx >= 0x7F800000U) { // inf or nan
        return static_cast<uint16_t>(sign | 0x7C00U | (absx > 0x7F800000U ? 0x200U : 0U));
    }
    if (absx >= 0x477FF000U) { // rounds to >= 65520 -> inf
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (absx < 0x38800000U) { // below the smallest normal half: subnormal or zero
        if (absx < 0x33000000U) return static_cast<uint16_t>(sign); // < 2^-25 -> 0
        const uint32_t e = absx >> 23;
        const uint32_t m = (absx & 0x7FFFFFU) | 0x800000U;
        const uint32_t shift = 126U - e; // 14..24
        uint32_t half_m = m >> shift;
        const uint32_t rem = m & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1U);
        if (rem > halfway || (rem == halfway && (half_m & 1U))) ++half_m;
        return static_cast<uint16_t>(sign | half_m);
    }
    uint32_t h = ((absx >> 13) - ((127U - 15U) << 10));
    const uint32_t rem = absx & 0x1FFFU;
    if (rem > 0x1000U || (rem == 0x1000U && (h & 1U))) ++h;
    return static_cast<uint16_t>(sign | h);
}

// --- one component ------------------------------------------------------------------------
CUDALINK_HD inline float load_comp(const uint8_t* p, const PixelCodec& c) {
    if (c.kind == FORMAT_KIND_UNSIGNED) {
        if (c.bits == 8) return static_cast<float>(*p) / 255.0f;
        uint16_t v;
        memcpy(&v, p, 2);
        return static_cast<float>(v) / 65535.0f;
    }
    if (c.bits == 16) {
        uint16_t v;
        memcpy(&v, p, 2);
        return half_bits_to_float(v);
    }
    float v;
    memcpy(&v, p, 4);
    return v;
}

CUDALINK_HD inline void store_comp(uint8_t* p, const PixelCodec& c, float x) {
    if (c.kind == FORMAT_KIND_UNSIGNED) {
        // !(x > 0) also catches NaN -> 0.
        const float s = !(x > 0.0f) ? 0.0f : (x > 1.0f ? 1.0f : x);
        if (c.bits == 8) {
            *p = static_cast<uint8_t>(s * 255.0f + 0.5f);
        } else {
            const uint16_t v = static_cast<uint16_t>(s * 65535.0f + 0.5f);
            memcpy(p, &v, 2);
        }
        return;
    }
    if (c.bits == 16) {
        const uint16_t v = float_to_half_bits(x);
        memcpy(p, &v, 2);
        return;
    }
    memcpy(p, &x, 4);
}

// --- one texel ----------------------------------------------------------------------------
CUDALINK_HD inline Float4 decode_texel(const uint8_t* p, const PixelCodec& c) {
    const uint32_t step = c.bits / 8U;
    Float4 o;
    if (c.comps == 4) {
        const float c0 = load_comp(p, c), c1 = load_comp(p + step, c), c2 = load_comp(p + 2 * step, c);
        o.a = load_comp(p + 3 * step, c);
        o.g = c1;
        o.r = c.bgra ? c2 : c0;
        o.b = c.bgra ? c0 : c2;
    } else if (c.comps == 2) {
        const float c0 = load_comp(p, c), c1 = load_comp(p + step, c);
        if (c.mono_alpha) {
            o.r = o.g = o.b = c0;
            o.a = c1;
        } else {
            o.r = c0;
            o.g = c1;
            o.b = 0.0f;
        }
    } else if (c.alpha_only) {
        o.r = o.g = o.b = 0.0f;
        o.a = load_comp(p, c);
    } else {
        o.r = o.g = o.b = load_comp(p, c);
    }
    return o;
}

CUDALINK_HD inline void encode_texel(uint8_t* p, const PixelCodec& c, const Float4& v) {
    const uint32_t step = c.bits / 8U;
    if (c.comps == 4) {
        store_comp(p, c, c.bgra ? v.b : v.r);
        store_comp(p + step, c, v.g);
        store_comp(p + 2 * step, c, c.bgra ? v.r : v.b);
        store_comp(p + 3 * step, c, v.a);
    } else if (c.comps == 2) {
        store_comp(p, c, v.r);
        store_comp(p + step, c, c.mono_alpha ? v.a : v.g);
    } else {
        store_comp(p, c, c.alpha_only ? v.a : v.r);
    }
}

CUDALINK_HD inline Float4 lerp4(const Float4& a, const Float4& b, float t) {
    return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t};
}

} // namespace cudalink::core
