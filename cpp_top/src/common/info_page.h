// info_page.h -- the "Info" parameter page both Custom TOPs share.
//
// Mirrors the .tox's Info page (Semverauthor / Semvershowhelp, same names so a script that
// reads one also reads the other). The package version and the build (git sha) are in the page
// header label and in the Info DAT rows "version" and "build", not in string parameters: TD
// saves a custom string par's value in the .toe, and a saved value beats the default of a newer
// DLL, so a Version or Build row showed the build the file was saved with, not the loaded one
// (live, 2026-11-19: row bea0b58, DLL ee63ce0). Labels and Info DAT rows come from the loaded
// DLL every time. The SDK has no read-only flag for custom pars, so the Author row is greyed with
// enablePar(..., false) on every cook (greyInfoPage). "Hide built-in Parameters" from the .tox
// has no SDK equivalent: the Common page cannot be hidden.
//
// CUDALINK_VERSION and CUDALINK_BUILD_SHA come from CMake (pyproject.toml version, and
// `git rev-parse --short HEAD` at configure time, "-dirty" for uncommitted edits). CMake
// reconfigures by itself when HEAD moves (cpp_top/CMakeLists.txt).
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <shellapi.h>

#include <cstdint>
#include <cstring>

#include "CPlusPlus_Common.h"
#include "param_util.h"

#ifndef CUDALINK_VERSION
#define CUDALINK_VERSION "unknown"
#endif
#ifndef CUDALINK_BUILD_SHA
#define CUDALINK_BUILD_SHA "unknown"
#endif

namespace cudalink::common {

constexpr char kInfoPage[] = "Info";
constexpr char kInfoHeaderName[] = "Cudalinkheader2";
constexpr char kInfoHeaderLabel[] = "CUDA-Link " CUDALINK_VERSION " (build " CUDALINK_BUILD_SHA
                                    ")   |   Zero-copy GPU texture sharing via CUDA IPC";
constexpr char kInfoAuthorName[] = "Semverauthor";
constexpr char kInfoHelpName[] = "Semvershowhelp";
constexpr char kInfoAuthor[] = "forkni (forkni@gmail.com)";
// The development branch for now: docs/CPP_TOP.md is not on master before the 2.0.0 release.
// Switch to blob/master at release time.
constexpr char kInfoHelpUrl[] = "https://github.com/forkni/cuda-link/blob/development/docs/CPP_TOP.md";

// Appends the Info page after the TOP's own page. Call last in setupParameters.
inline void appendInfoPage(TD::OP_ParameterManager* manager, const char* topName) {
    {
        TD::OP_StringParameter p(kInfoHeaderName);
        p.label = kInfoHeaderLabel;
        p.page = kInfoPage;
        checkParamAppend(manager->appendHeader(p), kInfoHeaderName, topName);
    }
    {
        TD::OP_StringParameter p(kInfoAuthorName);
        p.label = "Author";
        p.page = kInfoPage;
        p.defaultValue = kInfoAuthor;
        checkParamAppend(manager->appendString(p), kInfoAuthorName, topName);
    }
    {
        TD::OP_NumericParameter p(kInfoHelpName);
        p.label = "Show Help";
        p.page = kInfoPage;
        checkParamAppend(manager->appendPulse(p), kInfoHelpName, topName);
    }
}

// Greys the read-only Author row. Call every cook (enablePar state is per cook).
inline void greyInfoPage(const TD::OP_Inputs* inputs) {
    inputs->enablePar(kInfoAuthorName, false);
}

// Info DAT rows "version" and "build", appended after each TOP's own rows.
constexpr int32_t kNumBuildInfoDATRows = 2;

// Fills Info DAT row 'which' (0 = version, 1 = build) of the kNumBuildInfoDATRows rows.
inline void fillBuildInfoDATRow(int32_t which, TD::OP_InfoDATEntries* entries) {
    if (which == 0) {
        entries->values[0]->setString("version");
        entries->values[1]->setString(CUDALINK_VERSION);
    } else if (which == 1) {
        entries->values[0]->setString("build");
        entries->values[1]->setString(CUDALINK_BUILD_SHA);
    }
}

// Handles the Info page pulse; returns true when 'name' was an Info page par.
inline bool handleInfoPulse(const char* name) {
    if (name == nullptr || std::strcmp(name, kInfoHelpName) != 0) return false;
    ShellExecuteA(nullptr, "open", kInfoHelpUrl, nullptr, nullptr, SW_SHOWNORMAL);
    return true;
}

} // namespace cudalink::common
