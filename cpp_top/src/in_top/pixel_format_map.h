// Wire (format_kind, bits_per_comp, flags, num_comps) -> OP_PixelFormat for the receiver's
// output texture. Inverse of PLAN-001 D5's sender-side OP_PixelFormat -> dtype table.
//
// Channel order (RGBA vs BGRA) is not carried by the protocol, so 4-channel data always
// maps to the RGBA-order format (matches existing Python-side / D5-documented behavior).
// bfloat16 (FLAGS_BFLOAT16) has no OP_PixelFormat representation -- out of v1 scope,
// same as D5's sender table; returns Invalid.
// Alpha-only (A*Fixed/A*Float) vs Mono (Mono*Fixed/Float) are also indistinguishable on
// the wire (both are num_comps=1); this always resolves to Mono*, the same ambiguity D5
// documents for the sender direction.

#pragma once

#include "CPlusPlus_Common.h"
#include "../core/shm_layout.hpp"

namespace cudalink::in_top {

inline TD::OP_PixelFormat mapToPixelFormat(const cudalink::core::Metadata& meta) {
    using cudalink::core::FLAGS_BFLOAT16;
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
            case 1: return TD::OP_PixelFormat::Mono32Float;
            case 2: return TD::OP_PixelFormat::RG32Float;
            case 4: return TD::OP_PixelFormat::RGBA32Float;
            default: return TD::OP_PixelFormat::Invalid;
        }
    }
    if (meta.format_kind == FORMAT_KIND_FLOAT && meta.bits_per_comp == 16) {
        if (monoAlpha) return TD::OP_PixelFormat::MonoA16Float;
        switch (meta.num_comps) {
            case 1: return TD::OP_PixelFormat::Mono16Float;
            case 2: return TD::OP_PixelFormat::RG16Float;
            case 4: return TD::OP_PixelFormat::RGBA16Float;
            default: return TD::OP_PixelFormat::Invalid;
        }
    }
    if (meta.format_kind == FORMAT_KIND_UNSIGNED && meta.bits_per_comp == 16) {
        if (monoAlpha) return TD::OP_PixelFormat::MonoA16Fixed;
        switch (meta.num_comps) {
            case 1: return TD::OP_PixelFormat::Mono16Fixed;
            case 2: return TD::OP_PixelFormat::RG16Fixed;
            case 4: return TD::OP_PixelFormat::RGBA16Fixed;
            default: return TD::OP_PixelFormat::Invalid;
        }
    }
    if (meta.format_kind == FORMAT_KIND_UNSIGNED && meta.bits_per_comp == 8) {
        if (monoAlpha) return TD::OP_PixelFormat::MonoA8Fixed;
        switch (meta.num_comps) {
            case 1: return TD::OP_PixelFormat::Mono8Fixed;
            case 2: return TD::OP_PixelFormat::RG8Fixed;
            case 4: return TD::OP_PixelFormat::RGBA8Fixed;
            default: return TD::OP_PixelFormat::Invalid;
        }
    }

    // FORMAT_KIND_SIGNED (int8/int16) has no entry in D5's v1 pixel-format table either.
    return TD::OP_PixelFormat::Invalid;
}

} // namespace cudalink::in_top
