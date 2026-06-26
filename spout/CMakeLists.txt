# CMake build for the cuda-link-spout native module (_spout_bridge).
# Windows-only: requires CUDA Toolkit (12.x or 13.x), the Windows SDK (D3D11/DXGI),
# and the Spout2 SDK. Driven by scikit-build-core via pyproject.toml.
cmake_minimum_required(VERSION 3.24)
project(cuda_link_spout LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Native _spout_bridge needs Windows + CUDA Toolkit + Spout2 SDK. Default off elsewhere so
# scikit-build-core can still produce the documented pure-Python wheel (FakeSpoutBackend).
# Override with -DBUILD_SPOUT_NATIVE=ON/OFF (e.g. cmake.define.BUILD_SPOUT_NATIVE=OFF).
option(BUILD_SPOUT_NATIVE "Compile the _spout_bridge native module" ${WIN32})

if(NOT BUILD_SPOUT_NATIVE)
  message(STATUS "cuda-link-spout: native _spout_bridge disabled; building pure-Python wheel (FakeSpoutBackend).")
  return()
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Spout2 SDK location: set -DSPOUT2_ROOT=path/to/Spout2 at configure time.
# Expected layout: ${SPOUT2_ROOT}/SPOUTSDK/SpoutDirectX/SpoutDX
if(NOT DEFINED SPOUT2_ROOT)
  message(FATAL_ERROR "Set -DSPOUT2_ROOT to your Spout2 SDK checkout (github.com/leadedge/Spout2)")
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
