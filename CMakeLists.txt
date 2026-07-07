# CMake build for cuda-link's optional native wait-backend extension
# (_native_waiter). The native extension is Windows-only. Unlike
# cuda-link-spout, it requires NEITHER the CUDA Toolkit NOR any SDK at build
# time: the cudaEventQuery function pointer is handed in explicitly by Python
# (CUDARuntimeAPI's own resolved symbol, via set_cuda_event_query() — see
# src/cuda_link/_cpp/native_waiter.cpp) — this module never links, loads, or
# independently resolves cudart itself. Only a C++17 compiler + the Windows
# SDK (for synchapi.h) are needed.
#
# On any other platform, the native build is skipped and scikit-build-core
# packages a pure-Python wheel, so `pip install` succeeds everywhere. The
# pure-Python API runs against FakeWaitBackend without the extension; the
# native backend raises a clear error only if it is actually used without
# having been built.
cmake_minimum_required(VERSION 3.24)
project(cuda_link LANGUAGES CXX)

# Build the native extension only on Windows. Force with -DBUILD_NATIVE_WAITER=ON.
option(BUILD_NATIVE_WAITER "Build the native _native_waiter extension (Windows only)" ${WIN32})

if(NOT BUILD_NATIVE_WAITER)
  message(STATUS "cuda-link: skipping native _native_waiter; packaging a pure-Python wheel. "
                 "To build the extension, configure on Windows "
                 "(or force -DBUILD_NATIVE_WAITER=ON).")
  return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 2.12 CONFIG REQUIRED)

pybind11_add_module(_native_waiter
  src/cuda_link/_cpp/native_waiter.cpp
)

target_link_libraries(_native_waiter PRIVATE
  kernel32
  synchronization
)

# Install the compiled module next to the Python package so it imports as
# cuda_link._native_waiter.
install(TARGETS _native_waiter LIBRARY DESTINATION cuda_link)
