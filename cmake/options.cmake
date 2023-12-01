# CMake options

message("")
cmessage(STATUS "Setting up options...")

option( WITH_GUNDAM_ROOT_APP "Build app gundamRoot" ON )
option( WITH_CACHE_MANAGER "Enable compiling of the cache manager used by the GPU" ON )
option( ENABLE_CUDA "Enable CUDA language check (Cache::Manager requires a GPU if CUDA is found)" OFF )

option( BUILD_DOC "Build documentation" OFF )
option( ENABLE_DEV_MODE "Enable specific dev related printouts" OFF  )
option( USE_STATIC_LINKS "Library build in static mod" OFF )
option( YAMLCPP_DIR "Set custom path to yaml-cpp lib" OFF )

option( COLOR_OUTPUT "Enable colored terminal output." ON )
option( TTYCHECK "Enable check if output is being sent to terminal/TTY." ON )
option( BATCH_MODE "Build to run in a batch queue (affects output)" OFF )

option( CXX_WARNINGS "Enable most C++ warning flags." ON )
option( CXX_MARCH_FLAG "Enable cpu architecture specific optimzations." OFF )
option( CMAKE_CXX_EXTENSIONS "Enable GNU extensions to C++ langauge (-std=gnu++14)." OFF )


# Set the default built type if it isn't already defined
if(NOT DEFINED CMAKE_BUILD_TYPE
    OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(BATCH_MODE)
  set(COLOR_OUTPUT NO)
  set(TTYCHECK NO)
  add_definitions( -D GUNDAM_BATCH )
endif(BATCH_MODE)

if (WITH_CACHE_MANAGER)
  cmessage( WARNING "Enabling cache manager..." )
  add_definitions( -D GUNDAM_USING_CACHE_MANAGER )

  # uncomment to enable the slow validations (NEVER during productions
  # or normal running).  These are whole code validations and are
  # extremely slow.
  if (CACHE_MANAGER_SLOW_VALIDATION)
    cmessage( STATUS "Using slow validation for debugging" )
    cmessage(WARNING "Using slow validation so runs will be very slow" )
    add_definitions( -DCACHE_MANAGER_SLOW_VALIDATION)
  endif (CACHE_MANAGER_SLOW_VALIDATION)

  cmessage( STATUS "GPU support is enabled (compiled, but only used when CUDA enabled)" )
else()
  cmessage( ALERT "GPU support is disabled" )
endif()


if( BUILD_DOC )
  # check if Doxygen is installed
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxygen.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started" )

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
  else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation" )
  endif (DOXYGEN_FOUND)
endif()
