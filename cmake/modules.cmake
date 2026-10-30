
message("")
cmessage( WARNING "Defining modules...")

set( MODULES Utils )

# Add the basic modules
list( APPEND MODULES DialDictionary)
list( APPEND MODULES ParametersManager)
list( APPEND MODULES SamplesManager)
list( APPEND MODULES DatasetManager)
if(WITH_BACKENDS)
  # CUDA availability has been resolved by dependencies.cmake at this point.
  set(gundamCompiledBackends "The following backends will be compiled:\n - CPU -- default")
  if(GUNDAM_CUDA_ENABLED)
    string(APPEND gundamCompiledBackends "\n - CUDA -- NVIDIA GPUs (${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION})")
  endif()
  if(APPLE)
    enable_language(OBJCXX)
    string(APPEND gundamCompiledBackends "\n - MPS -- Apple Metal GPUs (Objective-C++: ${CMAKE_OBJCXX_COMPILER_ID} ${CMAKE_OBJCXX_COMPILER_VERSION})")
  endif()
  cmessage(STATUS "${gundamCompiledBackends}")
  list( APPEND MODULES Backends)
else()
  cmessage(STATUS "Backends to compile: none (WITH_BACKENDS=OFF)")
endif()
list( APPEND MODULES Propagator)
list( APPEND MODULES StatisticalInference)
list( APPEND MODULES Fitter)

if(WITH_PYTHON_INTERFACE)
  list( APPEND MODULES PythonInterface)
endif()

if(WITH_CACHE_MANAGER)
  list( APPEND MODULES CacheManager )
endif()

foreach(mod ${MODULES})
  cmessage( STATUS "Adding cmake module: ${mod}" )
  add_subdirectory( ${CMAKE_SOURCE_DIR}/src/${mod} )
endforeach()
