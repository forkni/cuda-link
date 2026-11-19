// info_page.h -- the "Info" parameter page both Custom TOPs share.
//
// Mirrors the .tox's Info page (Semverversion / Semverauthor / Semvershowhelp, same names so a
// script that reads one also reads the other), plus a Build row the .tox does not need: the
// git sha the DLL was configured from, since two DLLs of the same package version can differ.
// The SDK has no read-only flag for custom pars, so the read-only rows are greyed with
// enablePar(..., false) on every cook (greyInfoPage). "Hide built-in Parameters" from the .tox
// has no SDK equivalent: the Common page cannot be hidden.
//
// CUDALINK_VERSION and CUDALINK_BUILD_SHA come from CMake (pyproject.toml version, and
// `git rev-parse --short HEAD` at configure time). Reconfigure to refresh the sha.
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <shellapi.h>

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
constexpr char kInfoHeaderLabel[] = "CUDA-Link   |   Zero-copy GPU texture sharing via CUDA IPC";
constexpr char kInfoVersionName[] = "Semverversion";
constexpr char kInfoBuildName[] = "Semverbuild";
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
    const struct {
        const char* name;
        const char* label;
        const char* value;
    } rows[] = {
        {kInfoVersionName, "Version", CUDALINK_VERSION},
        {kInfoBuildName, "Build", CUDALINK_BUILD_SHA},
        {kInfoAuthorName, "Author", kInfoAuthor},
    };
    for (const auto& row : rows) {
        TD::OP_StringParameter p(row.name);
        p.label = row.label;
        p.page = kInfoPage;
        p.defaultValue = row.value;
        checkParamAppend(manager->appendString(p), row.name, topName);
    }
    {
        TD::OP_NumericParameter p(kInfoHelpName);
        p.label = "Show Help";
        p.page = kInfoPage;
        checkParamAppend(manager->appendPulse(p), kInfoHelpName, topName);
    }
}

// Greys the read-only rows. Call every cook (enablePar state is per cook).
inline void greyInfoPage(const TD::OP_Inputs* inputs) {
    inputs->enablePar(kInfoVersionName, false);
    inputs->enablePar(kInfoBuildName, false);
    inputs->enablePar(kInfoAuthorName, false);
}

// Handles the Info page pulse; returns true when 'name' was an Info page par.
inline bool handleInfoPulse(const char* name) {
    if (name == nullptr || std::strcmp(name, kInfoHelpName) != 0) return false;
    ShellExecuteA(nullptr, "open", kInfoHelpUrl, nullptr, nullptr, SW_SHOWNORMAL);
    return true;
}

} // namespace cudalink::common
