set(BACKENDS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

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
