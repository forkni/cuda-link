// doorbell_name.h -- the Win32 name of the publish doorbell (a named auto-reset event).
//
// The Out TOP creates the event and the In TOP opens it, so both must build the same name.
// The Python side builds it in src/cuda_link/_doorbell.py (doorbell_event_name, _PREFIX);
// tests/core/test_cpp_protocol_parity.py compares the two prefixes.

#pragma once

#include <cstring>
#include <string>

namespace cudalink::core {

constexpr wchar_t kDoorbellPrefix[] = L"Local\\cudalink_db_";

// kDoorbellPrefix + shm_name. The name is widened byte by byte, which is correct for the
// ASCII Ipcmemname values this protocol uses (same contract as common/win_util.h widen()).
inline std::wstring doorbell_event_name(const char* shm_name) {
    std::wstring name(kDoorbellPrefix);
    name.append(shm_name, shm_name + std::strlen(shm_name));
    return name;
}

} // namespace cudalink::core
