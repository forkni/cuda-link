// win_util.h -- shared Win32 helpers for both TOPs.
//
// asHandle(): both TOPs store Win32 HANDLEs as void* class members (to avoid pulling
// <windows.h> into their headers) and cast back to HANDLE only at the .cpp call sites that
// actually touch the Win32 API.
//
// widen(): naive byte-widening, not a general UTF-8 decoder -- correct for the ASCII SHM/
// doorbell-event names this protocol uses in practice (both sides' CreateFileMapping/
// CreateEvent/OpenFileMapping names are verbatim and unprefixed ASCII).

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstring>
#include <string>

namespace cudalink::common {

inline HANDLE asHandle(void* h) {
    return static_cast<HANDLE>(h);
}

inline std::wstring widen(const char* s) {
    return {s, s + std::strlen(s)};
}

} // namespace cudalink::common
