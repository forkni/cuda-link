# cpp_top -- C++ Custom TOPs for cuda-link (PLAN-001).
#
# Targets:
#   cudalink_topcore  protocol core (ring_reader + ring_writer), TD-free and CUDA-free
#   topcore_tests     golden-byte parity tests for the core (CTest; fixtures from
#                     scripts/gen_golden_shm.py)
#   protocol_dump     read-only dump of a live SHM segment (Windows) or a saved image
#   CudaLinkInTOP / CudaLinkOutTOP  the two Custom TOP DLLs (Windows + CUDA only)
#
# CUDALINK_TOP_CORE_ONLY=ON builds only the first three: no Windows guard, no CUDA
# toolkit, no TD headers. CI uses it for the Linux leg (preset core-only).
#
# No .cu files exist yet (no on-GPU kernels needed for either TOP's D2D memcpy path), so
# this does not enable the CUDA *language* -- it links against the CUDA Runtime API
# (cudart) via CUDAToolkit as a plain C++ project. Enable LANGUAGES CUDA if/when a
# conversion kernel (e.g. a Y-flip surface-object kernel) is ever added.

cmake_minimum_required(VERSION 3.24)

# Must be set before project() or CMAKE_MSVC_RUNTIME_LIBRARY is silently ignored (D2).
cmake_policy(SET CMP0091 NEW)

project(cpp_top LANGUAGES CXX)

option(CUDALINK_TOP_CORE_ONLY "Build only the protocol core, its tests and protocol_dump" OFF)

# Drives clang-tidy (see repo-root .clang-tidy) and other tooling that needs include paths.
# The MSVC generator does not emit this file -- generate via the ninja-tooling preset
# (CMakePresets.json) for tooling; ship from either preset (msvc-release for the VS tree).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT WIN32 AND NOT CUDALINK_TOP_CORE_ONLY)
    message(FATAL_ERROR
        "cpp_top's TOPs are Windows-only (CUDA IPC handle sharing is Windows-specific). "
        "Configure with -DCUDALINK_TOP_CORE_ONLY=ON (preset core-only) to build the core tests.")
endif()

set(CMAKE_CXX_STANDARD 20) # required for std::atomic_ref (D4)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release dynamic CRT (/MD) in every configuration, including Debug, deliberately (D2):
# debug-CRT plugin DLLs are a recurring "failed to load" report against TD.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

# --- cudalink_topcore -- TD-free, CUDA-free protocol core (D4) ---
add_library(cudalink_topcore STATIC
    src/core/ring_reader.cpp
    src/core/ring_writer.cpp
)
target_include_directories(cudalink_topcore PUBLIC src/core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cudalink_topcore PRIVATE -Wall -Wextra)
endif()

# --- topcore_tests -- golden-byte parity suite (CTest) ---
enable_testing()
add_executable(topcore_tests tests/topcore_tests.cpp)
target_link_libraries(topcore_tests PRIVATE cudalink_topcore)
target_compile_definitions(topcore_tests PRIVATE
    CUDALINK_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
# No argument: the test reads the fixtures from CUDALINK_FIXTURE_DIR above.
add_test(NAME topcore_tests COMMAND topcore_tests)

# --- protocol_dump -- read-only SHM inspector (named segments need Windows) ---
add_executable(protocol_dump tools/protocol_dump.cpp)
target_link_libraries(protocol_dump PRIVATE cudalink_topcore)
add_test(NAME protocol_dump_fixture
    COMMAND protocol_dump --file "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ring_n3.bin")
set_tests_properties(protocol_dump_fixture PROPERTIES
    PASS_REGULAR_EXPRESSION "new reader  NewFrame slot 2")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(topcore_tests PRIVATE -Wall -Wextra)
    target_compile_options(protocol_dump PRIVATE -Wall -Wextra)
endif()

if(CUDALINK_TOP_CORE_ONLY)
    return()
endif()

set(CUDALINK_TOP_CUDA_MAJOR "12" CACHE STRING "CUDA major version this leg targets (12 or 11)")

if(NOT DEFINED CUDAToolkit_ROOT)
    # Primary dev target per PLAN-001: TD 2025.3x + CUDA 12.8. Override with
    # -DCUDAToolkit_ROOT=... to build against a different installed toolkit (e.g. the
    # 11.8 leg for TD 2023.1x, once that leg exists).
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT CUDAToolkit_VERSION_MAJOR STREQUAL CUDALINK_TOP_CUDA_MAJOR)
    message(WARNING
        "CUDAToolkit_VERSION_MAJOR (${CUDAToolkit_VERSION_MAJOR}) does not match "
        "CUDALINK_TOP_CUDA_MAJOR (${CUDALINK_TOP_CUDA_MAJOR}) -- check CUDAToolkit_ROOT.")
endif()

set(TD_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/td/2025")

# CUDA language for the Common page resize/convert kernel (resize_convert.cu), from the same
# toolkit as CUDA::cudart. Real SASS for Pascal..Ada (and Blackwell from CUDA 12.8, the first
# toolkit with sm_120), plus PTX of the newest one for later GPUs. CI uses CUDA 12.6.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
        set(CMAKE_CUDA_ARCHITECTURES "61-real;75-real;86-real;89-real;120")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "61-real;75-real;86-real;89")
    endif()
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- cudalink_topcommon -- shared, header-only helpers for both TOP DLLs ---
# TD/CUDA-dependent (CudaDeviceSession touches TD::TOP_Context and cuda_runtime.h), so this is
# a sibling to cudalink_topcore (protocol-core, TD/CUDA-free) rather than folded into it.
# INTERFACE, not STATIC: every helper here (cuda_check.h, cuda_op_scope.h, debug_log.h,
# raii_handles.h, cuda_device_session.h, bench_accumulator.h, param_util.h, win_util.h,
# info_page.h) is header-only -- there is no .cpp to compile. The one CUDA kernel lives in
# cudalink_kernels below.
add_library(cudalink_topcommon INTERFACE)
target_include_directories(cudalink_topcommon INTERFACE src/common)
# shell32: ShellExecuteA for the Info page's Show Help pulse (info_page.h).
target_link_libraries(cudalink_topcommon INTERFACE CUDA::cudart shell32)

# Info page values (info_page.h): the package version from __version__ in the repo's
# src/cuda_link/__init__.py (the single version source; pyproject.toml reads it too), and the
# git sha at configure time ("unknown" outside a git checkout). The sha gets a "-dirty" suffix
# when tracked files differ from HEAD at configure time. CMake reconfigures by itself when HEAD
# or the checked-out branch ref moves (a commit, checkout or reset), so a plain rebuild after a
# commit does not keep the old sha. Uncommitted edits made after configure are not seen: configure
# again for a dirty build.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../src/cuda_link/__init__.py" CUDALINK_INIT_VERSION_LINE
    REGEX "^__version__ = \"[^\"]+\"" LIMIT_COUNT 1)
string(REGEX REPLACE "^__version__ = \"([^\"]+)\".*" "\\1" CUDALINK_VERSION "${CUDALINK_INIT_VERSION_LINE}")
if(NOT CUDALINK_VERSION)
    set(CUDALINK_VERSION "unknown")
endif()
find_package(Git QUIET)
set(CUDALINK_BUILD_SHA "unknown")
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE CUDALINK_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CUDALINK_GIT_RESULT ERROR_QUIET)
    if(CUDALINK_GIT_RESULT EQUAL 0 AND CUDALINK_GIT_SHA)
        set(CUDALINK_BUILD_SHA "${CUDALINK_GIT_SHA}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" diff --quiet HEAD --
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE CUDALINK_GIT_DIRTY ERROR_QUIET)
        if(CUDALINK_GIT_DIRTY EQUAL 1)
            string(APPEND CUDALINK_BUILD_SHA "-dirty")
        endif()
        # Reconfigure when HEAD or the branch it points to moves. git-path resolves both for a
        # linked worktree too (HEAD in the worktree's git dir, refs/heads/* in the common dir).
        set(CUDALINK_GIT_WATCH HEAD)
        execute_process(COMMAND "${GIT_EXECUTABLE}" symbolic-ref -q HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE CUDALINK_GIT_REF OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE CUDALINK_GIT_REF_RESULT ERROR_QUIET)
        if(CUDALINK_GIT_REF_RESULT EQUAL 0 AND CUDALINK_GIT_REF)
            list(APPEND CUDALINK_GIT_WATCH "${CUDALINK_GIT_REF}")
        endif()
        foreach(CUDALINK_GIT_ITEM IN LISTS CUDALINK_GIT_WATCH)
            execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --git-path "${CUDALINK_GIT_ITEM}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE CUDALINK_GIT_FILE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if(CUDALINK_GIT_FILE)
                get_filename_component(CUDALINK_GIT_FILE "${CUDALINK_GIT_FILE}" ABSOLUTE
                    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
                if(EXISTS "${CUDALINK_GIT_FILE}")
                    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CUDALINK_GIT_FILE}")
                endif()
            endif()
        endforeach()
    endif()
endif()
message(STATUS "cudalink TOPs: version ${CUDALINK_VERSION}, build ${CUDALINK_BUILD_SHA}")
target_compile_definitions(cudalink_topcommon INTERFACE
    CUDALINK_VERSION="${CUDALINK_VERSION}" CUDALINK_BUILD_SHA="${CUDALINK_BUILD_SHA}")

# --- cudalink_kernels -- the Common page resize/convert kernel (resize_convert.h) ---
# Shared cudart, the same runtime the DLLs link (CUDA::cudart): a static cudart here would
# put a second CUDA runtime into each DLL.
add_library(cudalink_kernels STATIC src/common/resize_convert.cu)
target_include_directories(cudalink_kernels PUBLIC src/common src/core)
target_link_libraries(cudalink_kernels PUBLIC CUDA::cudart)
set_target_properties(cudalink_kernels PROPERTIES CUDA_RUNTIME_LIBRARY Shared POSITION_INDEPENDENT_CODE ON)
# sm_61 (Pascal) is still built on purpose; nvcc only warns that it will go away.
target_compile_options(cudalink_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

# --- resize_convert_tests -- GPU kernel vs host codec (needs a CUDA device; 77 = skipped) ---
add_executable(resize_convert_tests tests/resize_convert_tests.cu)
target_link_libraries(resize_convert_tests PRIVATE cudalink_kernels)
set_target_properties(resize_convert_tests PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
target_compile_options(resize_convert_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
add_test(NAME resize_convert_tests COMMAND resize_convert_tests)
set_tests_properties(resize_convert_tests PROPERTIES SKIP_RETURN_CODE 77)

# --- CudaLinkInTOP -- the receiver Custom TOP DLL (D6/D7) ---
add_library(CudaLinkInTOP MODULE
    src/in_top/CudaLinkInTOP.cpp
    src/in_top/Parameters.cpp
)
target_include_directories(CudaLinkInTOP PRIVATE ${TD_VENDOR_DIR} src/common)
target_link_libraries(CudaLinkInTOP PRIVATE cudalink_topcore cudalink_topcommon cudalink_kernels CUDA::cudart)
target_compile_definitions(CudaLinkInTOP PRIVATE _CRT_SECURE_NO_WARNINGS)

# TD loads Custom Operator DLLs from a Plugins folder with no "lib" prefix expected.
set_target_properties(CudaLinkInTOP PROPERTIES PREFIX "" SUFFIX ".dll")

# --- CudaLinkOutTOP -- the sender Custom TOP DLL (D5/D7) ---
add_library(CudaLinkOutTOP MODULE
    src/out_top/CudaLinkOutTOP.cpp
    src/out_top/Parameters.cpp
)
target_include_directories(CudaLinkOutTOP PRIVATE ${TD_VENDOR_DIR} src/common)
target_link_libraries(CudaLinkOutTOP PRIVATE cudalink_topcore cudalink_topcommon cudalink_kernels CUDA::cudart)
target_compile_definitions(CudaLinkOutTOP PRIVATE _CRT_SECURE_NO_WARNINGS)

set_target_properties(CudaLinkOutTOP PROPERTIES PREFIX "" SUFFIX ".dll")

# --- BUILD_INFO.txt -- the build sidecar next to the DLLs ---
# The root CMakeLists.txt ships it in the wheel next to the DLLs. The installers
# (cuda_link.td_plugin, for install_cpp_top.py and install_td_library.py) read it from the
# source folder for the install record (ADR-0016); it is not copied to the Plugins folder.
# Same values as the Info page.
# file(GENERATE) puts it in the DLLs' own folder for single- and multi-config generators.
file(STRINGS "${TD_VENDOR_DIR}/CPlusPlus_Common.h" CUDALINK_TD_API_LINE
     REGEX "^const int OP_CommonAPIVersion = [0-9]+;" LIMIT_COUNT 1)
# file(STRINGS) returns the line with its ";" escaped as "\;" (a list separator), so a REPLACE
# pattern that ends in ";" never matches and passes the whole line through: match the number.
string(REGEX MATCH "OP_CommonAPIVersion = ([0-9]+)" CUDALINK_TD_API_MATCH "${CUDALINK_TD_API_LINE}")
if(CUDALINK_TD_API_MATCH)
    set(CUDALINK_TD_API "${CMAKE_MATCH_1}")
else()
    set(CUDALINK_TD_API "unknown")
    message(WARNING "OP_CommonAPIVersion not found in ${TD_VENDOR_DIR}/CPlusPlus_Common.h: BUILD_INFO.txt says unknown")
endif()
file(GENERATE OUTPUT "$<TARGET_FILE_DIR:CudaLinkOutTOP>/BUILD_INFO.txt"
    CONTENT "cuda-link ${CUDALINK_VERSION}\nbuild ${CUDALINK_BUILD_SHA}\ncuda toolkit ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}\ntd header OP_CommonAPIVersion ${CUDALINK_TD_API}\n")
