set(BACKENDS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# CUDA availability has been resolved by dependencies.cmake at this point.
cmessage(STATUS "The following backends will be compiled:")
cmessage(STATUS "- CPU: default")
if(GUNDAM_CUDA_ENABLED)
  cmessage(STATUS "- CUDA: NVIDIA GPUs (${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION})")
endif()
if(APPLE)
  enable_language(OBJCXX)
  cmessage(STATUS "- MPS: Apple Metal GPUs (Objective-C++: ${CMAKE_OBJCXX_COMPILER_ID} ${CMAKE_OBJCXX_COMPILER_VERSION})")
endif()

add_subdirectory(Common/Formula)
add_subdirectory(Common/Binning)
add_subdirectory(Common/Model)
add_subdirectory(Common/Execution)

add_subdirectory(Implementations/CPU)
if(GUNDAM_CUDA_ENABLED)
  add_subdirectory(Implementations/CUDA)
endif()
if(APPLE)
  add_subdirectory(Implementations/MPS)
endif()

add_subdirectory(Common/Manager)
add_library(GundamBackends ALIAS GundamBackendsCommonManager)
