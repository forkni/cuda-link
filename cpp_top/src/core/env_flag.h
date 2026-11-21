// env_flag.h -- the one rule both TOPs use to read an on/off environment variable.
//
// On: the variable is set, is not empty, and does not start with '0'. So "1", "true" and
// "yes" are on; unset, "" and "0" are off. The Python side (src/cuda_link/_env.py env_bool)
// is stricter: only "1" is on. Use "1" and "0" and the two sides agree.

#pragma once

#include <cstdlib>

namespace cudalink::core {

// The rule on a value already read (nullptr = the variable is not set). Pure, so
// topcore_tests covers it without touching the process environment.
constexpr bool flag_value_on(const char* value) noexcept {
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

// Reads `name` from the process environment now. Callers read once, at construction.
inline bool env_flag(const char* name) noexcept {
    return flag_value_on(std::getenv(name));
}

} // namespace cudalink::core
