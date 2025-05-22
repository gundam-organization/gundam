# Build the CacheManager code with or without CUDA.

# The headers are the same for both CUDA and C++
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightBilinear.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightBicubic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightTabulated.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightKriged.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/WeightBase.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CacheIndexedSums.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CacheRecursiveSums.h
)

# Pure C++ only source files.
set( SRCFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheAtomicCAS.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheSampleHistFiller.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheEventWeightFiller.cpp
)

set( SRC_FILE_EXT "cpp" )       # The C++ extension
if(CMAKE_CUDA_COMPILER)
  cmessage( STATUS "CUDA utilities being compiled.")
  set( SRC_FILE_EXT "cu" )      # Use this extension when using CUDA
endif(CMAKE_CUDA_COMPILER)

# Add the files that are compiled using C++ for CPU and with CUDA
# for GPU.
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightNormalization.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightNormalization.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightCompactSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightMonotonicSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightUniformSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightGeneralSpline.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightGraph.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightBilinear.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightBicubic.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightTabulated.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightKriged.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/WeightBase.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheParameters.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheWeights.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheIndexedSums.${SRC_FILE_EXT} )
list( APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/CacheRecursiveSums.${SRC_FILE_EXT} )

if( USE_STATIC_LINKS )
  add_library(GundamCacheManager STATIC ${SRCFILES})
else()
  add_library(GundamCacheManager SHARED ${SRCFILES})
endif( USE_STATIC_LINKS )

# Make sure the current directories are available for the later
# compilation.
target_include_directories( GundamCacheManager PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( GundamCacheManager PUBLIC
  GundamDialDictionary
  GundamParametersManager
  GundamSamplesManager
  ${ROOT_LIBRARIES})

# Add extra compilation flags.  The commented flags are as reminders
# for how to add specific debug flags.
if(NOT CMAKE_CUDA_COMPILER)
  # Specific definitions when not using the CUDA compiler, make sure
  # that HEMI_CUDA_DISABLE is set.  This is done since C++ can compile
  # everything except the actual kernel launches and so the CUDA
  # routines need to be explicitly marked as "do not call" during C++
  # compilation.
  target_compile_definitions( GundamCacheManager PUBLIC
    HEMI_CUDA_DISABLE)
else()
  # Specific definitions when using a CUDA compiler (e.g. NVCC)
  # Nothing yet!
endif(NOT CMAKE_CUDA_COMPILER)

install(TARGETS GundamCacheManager DESTINATION lib)
