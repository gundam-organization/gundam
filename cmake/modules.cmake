
message("")
cmessage( WARNING "Defining modules...")

set( MODULES Utils )

# Add the basic modules
list( APPEND MODULES DialDictionary)
list( APPEND MODULES ParametersManager)
list( APPEND MODULES SamplesManager)
list( APPEND MODULES DatasetManager)
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
