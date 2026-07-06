// OP_PixelFormat -> wire (format_kind, bits_per_comp, flags, num_comps) for the sender's
// input texture. Inverse of the receiver-side wire -> OP_PixelFormat table
// (in_top/pixel_format_map.h); verified 1:1 against the same OP_PixelFormat enum values
// in the vendored CPlusPlus_Common.h.
//
// Channel order for 8-bit 4-channel data is carried via FLAGS_BGRA: BGRA8Fixed (TD's
// preferred/default 4-channel 8-bit format) sets the flag so the receiver reconstructs
// BGRA8Fixed instead of assuming RGBA order. Float/uint16 4-channel formats are always
// RGBA on both sides and never set the flag.
// Alpha-only (A*Fixed/A*Float) and Mono (Mono*Fixed/Float) are indistinguishable on the
// wire (both are num_comps=1); this is an accepted ambiguity, same as the receiver's
// inverse table.
// RGB10A2Fixed / RGB11Float (packed formats) and signed-integer formats have no wire
// representation -- rejected (kResult.supported=false); execute() must skip the frame
// with a warning badge (mirrors TDSender.py::_is_unsupported_format).

#pragma once

#include "CPlusPlus_Common.h"
#include "../core/shm_layout.h"

namespace cudalink::out_top {

struct WireFormat {
    bool supported = false;
    uint8_t format_kind = cudalink::core::FORMAT_KIND_FLOAT;
    uint8_t bits_per_comp = 0;
    uint16_t flags = 0;
    uint32_t num_comps = 0;
};

inline WireFormat mapFromPixelFormat(TD::OP_PixelFormat fmt) {
    using cudalink::core::FLAGS_BGRA;
    using cudalink::core::FLAGS_MONO_ALPHA;
    using cudalink::core::FORMAT_KIND_FLOAT;
    using cudalink::core::FORMAT_KIND_UNSIGNED;

    WireFormat w;
    switch (fmt) {
        // --- float32 ---
        case TD::OP_PixelFormat::RGBA32Float:
            w = {true, FORMAT_KIND_FLOAT, 32, 0, 4};
            break;
        case TD::OP_PixelFormat::RG32Float:
            w = {true, FORMAT_KIND_FLOAT, 32, 0, 2};
            break;
        case TD::OP_PixelFormat::Mono32Float:
            w = {true, FORMAT_KIND_FLOAT, 32, 0, 1};
            break;
        case TD::OP_PixelFormat::A32Float:
            w = {true, FORMAT_KIND_FLOAT, 32, 0, 1};
            break;
        case TD::OP_PixelFormat::MonoA32Float:
            w = {true, FORMAT_KIND_FLOAT, 32, FLAGS_MONO_ALPHA, 2};
            break;

        // --- float16 (new capability -- rejected by Python's cudaMemory() path) ---
        case TD::OP_PixelFormat::RGBA16Float:
            w = {true, FORMAT_KIND_FLOAT, 16, 0, 4};
            break;
        case TD::OP_PixelFormat::RG16Float:
            w = {true, FORMAT_KIND_FLOAT, 16, 0, 2};
            break;
        case TD::OP_PixelFormat::Mono16Float:
            w = {true, FORMAT_KIND_FLOAT, 16, 0, 1};
            break;
        case TD::OP_PixelFormat::A16Float:
            w = {true, FORMAT_KIND_FLOAT, 16, 0, 1};
            break;
        case TD::OP_PixelFormat::MonoA16Float:
            w = {true, FORMAT_KIND_FLOAT, 16, FLAGS_MONO_ALPHA, 2};
            break;

        // --- uint16 ---
        case TD::OP_PixelFormat::RGBA16Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 16, 0, 4};
            break;
        case TD::OP_PixelFormat::RG16Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 16, 0, 2};
            break;
        case TD::OP_PixelFormat::Mono16Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 16, 0, 1};
            break;
        case TD::OP_PixelFormat::A16Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 16, 0, 1};
            break;
        case TD::OP_PixelFormat::MonoA16Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 16, FLAGS_MONO_ALPHA, 2};
            break;

        // --- uint8 ---
        case TD::OP_PixelFormat::RGBA8Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 8, 0, 4};
            break;
        case TD::OP_PixelFormat::BGRA8Fixed: // TD's preferred 4-channel 8-bit format
            w = {true, FORMAT_KIND_UNSIGNED, 8, FLAGS_BGRA, 4};
            break;
        case TD::OP_PixelFormat::RG8Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 8, 0, 2};
            break;
        case TD::OP_PixelFormat::Mono8Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 8, 0, 1};
            break;
        case TD::OP_PixelFormat::A8Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 8, 0, 1};
            break;
        case TD::OP_PixelFormat::MonoA8Fixed:
            w = {true, FORMAT_KIND_UNSIGNED, 8, FLAGS_MONO_ALPHA, 2};
            break;

        // --- rejected: packed / no wire representation (v1 scope) ---
        case TD::OP_PixelFormat::RGB10A2Fixed:
        case TD::OP_PixelFormat::RGB11Float:
        default:
            w = WireFormat{}; // supported=false
            break;
    }
    return w;
}

} // namespace cudalink::out_top
