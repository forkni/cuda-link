// common_page.h -- TD-free resolver for the size and aspect that TD's Common page asks for.
//
// The Out TOP does not need this: TOP_Output::getSuggestedOutputDesc already resolves every
// mode against its TOP input. The In TOP has no TOP input, so TD resolves "Use Input" and the
// relative modes against a 127x127 placeholder (Phase 0 probe, 2026-11-18). The In TOP
// therefore resolves those modes here against the wire frame, and takes TD's suggested size
// only for Custom and Parameter Panel, which do not depend on an input.
//
// Rounding follows the probe: relative modes floor (127 eighth -> 15, half -> 63, 2x -> 254).
// Fit and Limit scale the base to fit inside the Resolution box, keeping the base aspect.
// The live oracle in docs/CPP_TOP.md compares these sizes with the Out TOP's suggested desc.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace cudalink::core {

enum class ResMode : uint8_t { UseInput, Eighth, Quarter, Half, X2, X4, X8, Fit, Limit, Custom, ParPanel };
enum class AspectMode : uint8_t { UseInput, Resolution, Custom };

struct Size2 {
    uint32_t w = 0;
    uint32_t h = 0;
    bool operator==(const Size2& o) const { return w == o.w && h == o.h; }
};

// Pixel aspect of an output, as OP_TextureDesc carries it (0/0 = square pixels: w/h).
struct Aspect2 {
    float x = 0.0f;
    float y = 0.0f;
};

// Menu name (OP_Inputs::getParString("outputresolution")) -> mode. Unknown -> UseInput.
inline ResMode parse_res_mode(const char* name) {
    struct Entry {
        const char* name;
        ResMode mode;
    };
    static constexpr Entry kEntries[] = {
        {"useinput", ResMode::UseInput},
        {"eighth", ResMode::Eighth},
        {"quarter", ResMode::Quarter},
        {"half", ResMode::Half},
        {"2x", ResMode::X2},
        {"4x", ResMode::X4},
        {"8x", ResMode::X8},
        {"fit", ResMode::Fit},
        {"limit", ResMode::Limit},
        {"custom", ResMode::Custom},
        {"parpanel", ResMode::ParPanel},
    };
    if (name != nullptr) {
        for (const Entry& e : kEntries) {
            if (std::strcmp(name, e.name) == 0) return e.mode;
        }
    }
    return ResMode::UseInput;
}

// Menu name (getParString("outputaspect")) -> mode. Unknown -> Resolution (TD's default).
inline AspectMode parse_aspect_mode(const char* name) {
    if (name != nullptr) {
        if (std::strcmp(name, "useinput") == 0) return AspectMode::UseInput;
        if (std::strcmp(name, "custom") == 0) return AspectMode::Custom;
    }
    return AspectMode::Resolution;
}

namespace detail {
inline uint32_t at_least_one(double v) {
    const double f = std::floor(v);
    return f < 1.0 ? 1U : static_cast<uint32_t>(f);
}
inline uint32_t at_least_one_round(double v) {
    const double r = std::floor(v + 0.5);
    return r < 1.0 ? 1U : static_cast<uint32_t>(r);
}
} // namespace detail

// base: the size "Use Input" means (In TOP: the wire frame).
// box: the Resolution pars (Fit / Limit target).
// suggested: TD's suggested size (used for Custom and Parameter Panel only).
inline Size2 resolve_size(ResMode mode, Size2 base, Size2 box, Size2 suggested) {
    using detail::at_least_one;
    switch (mode) {
        case ResMode::UseInput:
            return base;
        case ResMode::Eighth:
            return {at_least_one(base.w / 8.0), at_least_one(base.h / 8.0)};
        case ResMode::Quarter:
            return {at_least_one(base.w / 4.0), at_least_one(base.h / 4.0)};
        case ResMode::Half:
            return {at_least_one(base.w / 2.0), at_least_one(base.h / 2.0)};
        case ResMode::X2:
            return {base.w * 2U, base.h * 2U};
        case ResMode::X4:
            return {base.w * 4U, base.h * 4U};
        case ResMode::X8:
            return {base.w * 8U, base.h * 8U};
        case ResMode::Fit:
        case ResMode::Limit: {
            if (base.w == 0 || base.h == 0 || box.w == 0 || box.h == 0) return base;
            double scale = std::min(static_cast<double>(box.w) / base.w, static_cast<double>(box.h) / base.h);
            if (mode == ResMode::Limit && scale >= 1.0) return base;
            return {detail::at_least_one_round(base.w * scale), detail::at_least_one_round(base.h * scale)};
        }
        case ResMode::Custom:
        case ResMode::ParPanel:
            return suggested;
    }
    return base;
}

// base_aspect: the aspect "Use Input" means (In TOP: the wire frame, 0/0 = square pixels).
// custom: the Aspect pars (TD's suggested aspect when the mode is Custom).
inline Aspect2 resolve_aspect(AspectMode mode, Size2 out, Aspect2 base_aspect, Aspect2 custom) {
    switch (mode) {
        case AspectMode::UseInput:
            return base_aspect;
        case AspectMode::Custom:
            return custom;
        case AspectMode::Resolution:
            break;
    }
    return {static_cast<float>(out.w), static_cast<float>(out.h)};
}

} // namespace cudalink::core
