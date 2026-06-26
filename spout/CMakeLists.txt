# CMake build for the cuda-link-spout native module (_spout_bridge).
# The native extension is Windows-only: it requires the CUDA Toolkit (12.x or 13.x),
# the Windows SDK (D3D11/DXGI), and the Spout2 SDK. Driven by scikit-build-core.
#
# On any other platform — or when the Spout2 SDK path isn't supplied — the native
# build is skipped and scikit-build-core packages a pure-Python wheel, so
# `pip install` succeeds everywhere. The pure-Python API runs against
# FakeSpoutBackend without the extension; the native backend raises a clear error
# only if it is actually used without having been built.
cmake_minimum_required(VERSION 3.24)
project(cuda_link_spout LANGUAGES CXX)

# Build the native extension only on Windows with a Spout2 SDK present. Force with
# -DBUILD_SPOUT_NATIVE=ON (then -DSPOUT2_ROOT=... is required, validated below).
option(BUILD_SPOUT_NATIVE "Build the native _spout_bridge extension (Windows + CUDA + Spout2)" OFF)
if(WIN32 AND DEFINED SPOUT2_ROOT)
  set(BUILD_SPOUT_NATIVE ON)
endif()

if(NOT BUILD_SPOUT_NATIVE)
  message(STATUS "cuda-link-spout: skipping native _spout_bridge; packaging a pure-Python wheel. "
                 "To build the extension, configure on Windows with -DSPOUT2_ROOT=path/to/Spout2 "
                 "(or force -DBUILD_SPOUT_NATIVE=ON).")
  return()
endif()

# --- native build (Windows + CUDA + D3D11 + Spout2) -------------------------
# CUDA is reached purely through the runtime API + CUDA::cudart (no .cu sources),
# so the CUDA language is intentionally not enabled — only a C++ compiler is needed.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Spout2 SDK location: set -DSPOUT2_ROOT=path/to/Spout2 at configure time.
# Expected layout: ${SPOUT2_ROOT}/SPOUTSDK/SpoutDirectX/SpoutDX
if(NOT DEFINED SPOUT2_ROOT)
  message(FATAL_ERROR "BUILD_SPOUT_NATIVE is ON but SPOUT2_ROOT is unset; "
                      "set -DSPOUT2_ROOT to your Spout2 SDK checkout (github.com/leadedge/Spout2)")
endif()
set(SPOUTDX_DIR "${SPOUT2_ROOT}/SPOUTSDK/SpoutDirectX/SpoutDX")
set(SPOUTGL_DIR "${SPOUT2_ROOT}/SPOUTSDK/SpoutGL")

file(GLOB SPOUTDX_SOURCES
  "${SPOUTDX_DIR}/*.cpp"
  "${SPOUTGL_DIR}/*.cpp"
)

pybind11_add_module(_spout_bridge
  src/cuda_link_spout/_cpp/spout_bridge.cpp
  ${SPOUTDX_SOURCES}
)

target_include_directories(_spout_bridge PRIVATE
  ${SPOUTDX_DIR}
  ${SPOUTGL_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(_spout_bridge PRIVATE
  CUDA::cudart
  d3d11
  dxgi
)

# Install the compiled module next to the Python package so it imports as
# cuda_link_spout._spout_bridge.
install(TARGETS _spout_bridge LIBRARY DESTINATION cuda_link_spout)
