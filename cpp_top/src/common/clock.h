// clock.h -- the shared monotonic clock both TOPs use for slot meta host_timestamp.
//
// The Out TOP writes now_seconds() into the slot meta host_timestamp at publish; the In TOP
// (and the Python importer) subtracts it from its own now to get latency_us. steady_clock,
// NOT system_clock/wall-clock: on Windows both MSVC's steady_clock and CPython's
// time.perf_counter() resolve to QueryPerformanceCounter(), sharing the same boot-relative
// zero point, so this value is directly comparable to the perf_counter() the Python
// exporter publishes and the Python importer reads back against. A wall-clock/epoch
// timestamp would still decode fine (same 8-byte double on the wire) but would be off by
// ~ the epoch-vs-boot offset when read against perf_counter() -- the multi-hundred-
// billion-ms latency bug this replaced. Also avoids NTP/DST step corruption landing in the
// 1-30 ms signal band that a wall-clock reading would risk.

#pragma once

#include <chrono>

namespace cudalink::common {

inline double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

} // namespace cudalink::common
