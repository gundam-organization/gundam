# Shared by GUNDAM and the standalone CUDA smoke test. No GPU is needed here.
set(GUNDAM_CUDA_ENABLED OFF)
string(TOUPPER "${WITH_CUDA_LIB}" gundamCudaMode)
if(NOT gundamCudaMode MATCHES "^(OFF|ON|AUTO)$")
  message(FATAL_ERROR "WITH_CUDA_LIB must be OFF, ON, or AUTO.")
endif()

if(NOT gundamCudaMode STREQUAL "OFF")
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR "CUDA support requires CMake 3.18 or newer.")
  endif()
  cmake_policy(SET CMP0104 NEW)
  # CheckLanguage trusts an explicitly supplied compiler and skips its probe.
  if(IS_ABSOLUTE "${CMAKE_CUDA_COMPILER}" AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
    message(FATAL_ERROR
      "CUDA compiler does not exist: ${CMAKE_CUDA_COMPILER}. "
      "Locate nvcc with 'command -v nvcc' after loading the toolkit, then "
      "configure with its actual path. Use a new build directory or remove "
      "the stale entry with cmake -U CMAKE_CUDA_COMPILER.")
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GUNDAM_CUDA_ENABLED ON)
    # Honor CMAKE_CUDA_ARCHITECTURES / CUDAARCHS, otherwise use the compiler
    # default. Never probe the GPU or impose an obsolete architecture.
    message(STATUS "CUDA ${CUDAToolkit_VERSION}: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  elseif(gundamCudaMode STREQUAL "ON")
    message(FATAL_ERROR "WITH_CUDA_LIB=ON requires a CUDA compiler. Set CMAKE_CUDA_COMPILER to nvcc, or use AUTO/OFF.")
  else()
    message(STATUS "CUDA compiler unavailable; continuing without CUDA (AUTO).")
  endif()
else()
  message(STATUS "CUDA support disabled (WITH_CUDA_LIB=OFF).")
endif()
