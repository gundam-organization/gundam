

set( SRCFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheAtomicCAS.c
)

set( HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CacheManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CacheParameters.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CacheWeights.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightNormalization.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightCompactSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightMonotonicSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightUniformSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightGeneralSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightGraph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CacheIndexedSums.h
)

set( SRC_FILE_EXT "cpp" )
if(CMAKE_CUDA_COMPILER)
  cmessage( STATUS "CUDA utilities being compiled.")
  set( SRC_FILE_EXT "cu" )
endif(CMAKE_CUDA_COMPILER)

list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightNormalization.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightNormalization.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightCompactSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightMonotonicSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightUniformSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightGeneralSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightGraph.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightBase.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheParameters.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheWeights.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheIndexedSums.${SRC_FILE_EXT} )

if( USE_STATIC_LINKS )
  add_library(GundamCache STATIC ${SRCFILES})
else()
  add_library(GundamCache SHARED ${SRCFILES})
endif()

# Make sure the current directories are available for the later
# compilation.
target_include_directories( GundamCache PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( GundamCache PUBLIC
  GundamDialDictionary
  GundamParametersManager
  GundamSamplesManager
  ${ROOT_LIBRARIES})

if(NOT CMAKE_CUDA_COMPILER)
  target_compile_definitions( GundamCache PUBLIC HEMI_CUDA_DISABLE)
endif(CMAKE_CUDA_COMPILER)

#  set_target_properties(GundamCache PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS GundamCache DESTINATION lib)
