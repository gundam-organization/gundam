cmake_minimum_required(VERSION 3.18)

# Also usable without ROOT or any other GUNDAM dependency:
# cmake -S tests/cuda -B build-cuda-smoke -DCMAKE_CUDA_ARCHITECTURES=70
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(GundamCudaSmoke LANGUAGES CXX)
  set(WITH_CUDA_LIB ON)
  include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/cuda.cmake)
  enable_testing()
endif()

add_executable(gundamCudaSmokeTest CudaSmokeTest.cu)
set_target_properties(gundamCudaSmokeTest PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
target_link_libraries(gundamCudaSmokeTest PRIVATE CUDA::cudart)
add_test(NAME cuda-smoke COMMAND gundamCudaSmokeTest)
set_tests_properties(cuda-smoke PROPERTIES LABELS "cuda;device" TIMEOUT 60)
