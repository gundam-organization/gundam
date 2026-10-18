
message("")
cmessage( WARNING "Defining modules...")

if(WIN32)
  # Export the module APIs and generate the import .lib files used by MSVC.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

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
