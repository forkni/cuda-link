// DebugLogger -- shared opt-in diagnostic trace for both Custom TOPs.
//
// Both CudaLinkOutTOP and CudaLinkInTOP need the same thing: a per-instance log file that
// stays completely inert (no file created, no I/O) until the node's Debug toggle is turned
// on, and then appends timestamped lines for the rest of the session. This class holds that
// one implementation so each TOP just owns a DebugLogger instead of carrying its own copy of
// the lazy-open/format/flush logic.
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>

namespace cudalink::common {

class DebugLogger {
public:
    // 'fileName' is the leaf name written under the OS temp directory (e.g.
    // "cudalink_out_top_debug.log"), so each TOP logs to its own file.
    explicit DebugLogger(std::string fileName) : myFileName(std::move(fileName)) {}

    // Mirrors the node's Debug parameter; cheap to call every cook.
    void setEnabled(bool enabled) { myEnabled = enabled; }
    bool enabled() const { return myEnabled; }

    // No-op when disabled, so callers can log unconditionally without checking enabled()
    // themselves. Lazily opens the file on first real use, appends one line, and flushes
    // immediately so the log is current if TD is killed rather than closed cleanly.
    void log(const std::string& msg, uint64_t frameCount) {
        if (!myEnabled) return;
        if (!myFile.is_open()) {
            char tempPath[MAX_PATH] = {};
            GetTempPathA(MAX_PATH, tempPath);
            myFile.open(std::string(tempPath) + myFileName, std::ios::app);
            if (!myFile.is_open()) return;
        }
        // Milliseconds since epoch as an integer, not a raw double: streaming a
        // ~1.78e9-second epoch value through std::ofstream's default float precision
        // rounds every line to the same "1.78324e+09", losing all sub-second resolution.
        const int64_t ms = static_cast<int64_t>(
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() * 1000.0);
        myFile << "[t=" << ms << " frame=" << frameCount << "] " << msg << "\n";
        myFile.flush();
    }

private:
    std::string myFileName;
    bool myEnabled = false;
    std::ofstream myFile;
};

} // namespace cudalink::common
