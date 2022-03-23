set(SRCFILES
        )
set(HEADERS
        include/CacheEventWeights.h
        )

if(CMAKE_CUDA_COMPILER)
  cmessage(STATUS "CUDA utilities being compiled")
  set(SRCFILES ${SRCFILES} src/CacheEventWeights.cu)
else(CMAKE_CUDA_COMPILER)
  cmessage(STATUS "CUDA utilities are not being compiled")
  set(SRCFILES ${SRCFILES} src/CacheEventWeights.cpp)
endif(CMAKE_CUDA_COMPILER)

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamCache STATIC ${SRCFILES})
else()
  add_library(GundamCache SHARED ${SRCFILES})
endif()

target_link_libraries( GundamCache PUBLIC
        ${ROOT_LIBRARIES}
        )

set_target_properties(GundamCache PROPERTIES VERSION
  "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

install(TARGETS GundamCache DESTINATION lib)

set(MODULETargets ${MODULETargets} GundamCache PARENT_SCOPE)
