# cpp_top -- C++ Custom TOPs for cuda-link (PLAN-001).
#
# This pass builds the receiver and sender: cudalink_topcore (protocol core, both
# ring_reader and ring_writer) + CudaLinkInTOP + CudaLinkOutTOP (the two Custom TOP
# DLLs). protocol_dump and topcore_tests (golden-byte parity infra) remain future work
# (out of scope for this implementation pass).
#
# No .cu files exist yet (no on-GPU kernels needed for either TOP's D2D memcpy path), so
# this does not enable the CUDA *language* -- it links against the CUDA Runtime API
# (cudart) via CUDAToolkit as a plain C++ project. Enable LANGUAGES CUDA if/when a
# conversion kernel (e.g. a Y-flip surface-object kernel) is ever added.

cmake_minimum_required(VERSION 3.24)

# Must be set before project() or CMAKE_MSVC_RUNTIME_LIBRARY is silently ignored (D2).
cmake_policy(SET CMP0091 NEW)

project(cpp_top LANGUAGES CXX)

# Drives clang-tidy (see repo-root .clang-tidy) and other tooling that needs include paths.
# The MSVC generator does not emit this file -- generate via the Ninja preset for tooling
# even though the shipping build stays MSVC.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT WIN32)
    message(FATAL_ERROR "cpp_top is Windows-only (CUDA IPC handle sharing is Windows-specific).")
endif()

set(CMAKE_CXX_STANDARD 20) # required for std::atomic_ref (D4)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release dynamic CRT (/MD) in every configuration, including Debug, deliberately (D2):
# debug-CRT plugin DLLs are a recurring "failed to load" report against TD.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

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

# --- cudalink_topcore -- TD-free, CUDA-free protocol core (D4) ---
add_library(cudalink_topcore STATIC
    src/core/ring_reader.cpp
    src/core/ring_writer.cpp
)
target_include_directories(cudalink_topcore PUBLIC src/core)

# --- cudalink_topcommon -- shared, header-only helpers for both TOP DLLs ---
# TD/CUDA-dependent (CudaDeviceSession touches TD::TOP_Context and cuda_runtime.h), so this is
# a sibling to cudalink_topcore (protocol-core, TD/CUDA-free) rather than folded into it.
# INTERFACE, not STATIC: every helper here (cuda_check.h, cuda_op_scope.h, debug_log.h,
# raii_handles.h, cuda_device_session.h, bench_accumulator.h, param_util.h, win_util.h) is
# header-only -- there is no .cpp to compile.
add_library(cudalink_topcommon INTERFACE)
target_include_directories(cudalink_topcommon INTERFACE src/common)
target_link_libraries(cudalink_topcommon INTERFACE CUDA::cudart)

# --- CudaLinkInTOP -- the receiver Custom TOP DLL (D6/D7) ---
add_library(CudaLinkInTOP MODULE
    src/in_top/CudaLinkInTOP.cpp
    src/in_top/Parameters.cpp
)
target_include_directories(CudaLinkInTOP PRIVATE ${TD_VENDOR_DIR} src/common)
target_link_libraries(CudaLinkInTOP PRIVATE cudalink_topcore cudalink_topcommon CUDA::cudart)
target_compile_definitions(CudaLinkInTOP PRIVATE _CRT_SECURE_NO_WARNINGS)

# TD loads Custom Operator DLLs from a Plugins folder with no "lib" prefix expected.
set_target_properties(CudaLinkInTOP PROPERTIES PREFIX "" SUFFIX ".dll")

# --- CudaLinkOutTOP -- the sender Custom TOP DLL (D5/D7) ---
add_library(CudaLinkOutTOP MODULE
    src/out_top/CudaLinkOutTOP.cpp
    src/out_top/Parameters.cpp
)
target_include_directories(CudaLinkOutTOP PRIVATE ${TD_VENDOR_DIR} src/common)
target_link_libraries(CudaLinkOutTOP PRIVATE cudalink_topcore cudalink_topcommon CUDA::cudart)
target_compile_definitions(CudaLinkOutTOP PRIVATE _CRT_SECURE_NO_WARNINGS)

set_target_properties(CudaLinkOutTOP PROPERTIES PREFIX "" SUFFIX ".dll")
