
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

if(WITH_BACKENDS)
  list( APPEND MODULES Backends)
endif()

if(WITH_CACHE_MANAGER)
  list( APPEND MODULES CacheManager )
endif()

cmessage( STATUS "The following modules will be included:" )
foreach(mod ${MODULES})
  cmessage( STATUS "- ${mod}" )
endforeach()

cmessage( STATUS "Reading module cmake config..." )
foreach(mod ${MODULES})
  add_subdirectory( ${CMAKE_SOURCE_DIR}/src/${mod} )
endforeach()
