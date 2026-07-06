// Wire (format_kind, bits_per_comp, flags, num_comps) -> OP_PixelFormat for the receiver's
// output texture. Inverse of the sender-side OP_PixelFormat -> dtype table
// (out_top/pixel_format_map.h).
//
// Channel order for 8-bit 4-channel data is recovered from FLAGS_BGRA: if set, this
// resolves to BGRA8Fixed (TD's preferred/default 4-channel 8-bit format) instead of
// assuming RGBA order. Float/uint16 4-channel formats never set the flag and always
// map to their RGBA-order format.
// bfloat16 (FLAGS_BFLOAT16) has no OP_PixelFormat representation -- out of v1 scope,
// same as the sender table; returns Invalid.
// Alpha-only (A*Fixed/A*Float) vs Mono (Mono*Fixed/Float) are also indistinguishable on
// the wire (both are num_comps=1); this always resolves to Mono*, the same ambiguity
// the sender direction has.

#pragma once

#include "CPlusPlus_Common.h"
#include "../core/shm_layout.h"

namespace cudalink::in_top {

inline TD::OP_PixelFormat mapToPixelFormat(const cudalink::core::Metadata& meta) {
    using cudalink::core::FLAGS_BFLOAT16;
    using cudalink::core::FLAGS_BGRA;
    using cudalink::core::FLAGS_MONO_ALPHA;
    using cudalink::core::FORMAT_KIND_FLOAT;
    using cudalink::core::FORMAT_KIND_UNSIGNED;

    if ((meta.flags & FLAGS_BFLOAT16) != 0) {
        return TD::OP_PixelFormat::Invalid;
    }
    const bool monoAlpha = (meta.flags & FLAGS_MONO_ALPHA) != 0;

    if (meta.format_kind == FORMAT_KIND_FLOAT && meta.bits_per_comp == 32) {
        if (monoAlpha) return TD::OP_PixelFormat::MonoA32Float;
        switch (meta.num_comps) {
            case 1:
                return TD::OP_PixelFormat::Mono32Float;
            case 2:
                return TD::OP_PixelFormat::RG32Float;
            case 4:
                return TD::OP_PixelFormat::RGBA32Float;
            default:
                return TD::OP_PixelFormat::Invalid;
        }
    }
    if (meta.format_kind == FORMAT_KIND_FLOAT && meta.bits_per_comp == 16) {
        if (monoAlpha) return TD::OP_PixelFormat::MonoA16Float;
        switch (meta.num_comps) {
            case 1:
                return TD::OP_PixelFormat::Mono16Float;
            case 2:
                return TD::OP_PixelFormat::RG16Float;
            case 4:
                return TD::OP_PixelFormat::RGBA16Float;
            default:
                return TD::OP_PixelFormat::Invalid;
        }
    }
    if (meta.format_kind == FORMAT_KIND_UNSIGNED && meta.bits_per_comp == 16) {
        if (monoAlpha) return TD::OP_PixelFormat::MonoA16Fixed;
        switch (meta.num_comps) {
            case 1:
                return TD::OP_PixelFormat::Mono16Fixed;
            case 2:
                return TD::OP_PixelFormat::RG16Fixed;
            case 4:
                return TD::OP_PixelFormat::RGBA16Fixed;
            default:
                return TD::OP_PixelFormat::Invalid;
        }
    }
    if (meta.format_kind == FORMAT_KIND_UNSIGNED && meta.bits_per_comp == 8) {
        if (monoAlpha) return TD::OP_PixelFormat::MonoA8Fixed;
        const bool isBgra = (meta.flags & FLAGS_BGRA) != 0;
        switch (meta.num_comps) {
            case 1:
                return TD::OP_PixelFormat::Mono8Fixed;
            case 2:
                return TD::OP_PixelFormat::RG8Fixed;
            case 4:
                return isBgra ? TD::OP_PixelFormat::BGRA8Fixed : TD::OP_PixelFormat::RGBA8Fixed;
            default:
                return TD::OP_PixelFormat::Invalid;
        }
    }

    // FORMAT_KIND_SIGNED (int8/int16) has no entry in the v1 pixel-format table either.
    return TD::OP_PixelFormat::Invalid;
}

} // namespace cudalink::in_top
