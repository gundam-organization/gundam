# CMake options

message("")
cmessage( WARNING "Setting up options...")


# general options
option( ENABLE_COLOR_OUTPUT "Enable colored terminal output." ON )
option( ENABLE_TTY_CHECK "Enable check if output is being sent to terminal/TTY." ON )
option( ENABLE_BATCH_MODE "Build to run in a batch queue (affects output)." OFF )
option( ENABLE_DEV_MODE "Enable specific dev related printouts." OFF  )

# extensions
option( WITH_DOXYGEN "Build documentation with doxygen." OFF )
option( WITH_GUNDAM_ROOT_APP "Build app gundamRoot." ON )
option( WITH_CACHE_MANAGER "Enable compiling of the cache manager (required for GPU computing)." ON )
option( WITH_CUDA_LIB "Enable CUDA language check (Cache::Manager requires a GPU if CUDA is found)." OFF )
option( WITH_MINUIT2_MISSING "Allow MINUIT2 to be missing" OFF )

# compile helper
option( YAMLCPP_DIR "Set custom path to yaml-cpp lib." OFF )

# dev options
option( USE_STATIC_LINKS "Use static link of libraries and apps instead of shared." OFF )
option( DISABLE_ZLIB "Disable Zlib dependency." OFF )
option( CXX_WARNINGS "Enable most C++ warning flags." ON )
option( CXX_MARCH_FLAG "Enable cpu architecture specific optimisations." OFF )
option( CMAKE_CXX_EXTENSIONS "Enable GNU extensions to C++ language (-std=gnu++14)." OFF )
option( DISABLE_MANUAL_LOG_HEADER "Don't rely on the manually set logger user header string." ON )


# Reading options
##################

# make sure the build type is upper cased
string( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )

# Set the default built type if it isn't already defined
if( NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set( CMAKE_BUILD_TYPE "RELEASE" )
  cmessage( STATUS "Build type not set. Using default build type: RELEASE." )
elseif( CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
  cmessage( STATUS "Build type manually specified to: RELEASE." )
elseif( CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
  cmessage( STATUS "Build type manually specified to: DEBUG." )
else()
  cmessage( WARNING "Build type not recognised: ${CMAKE_BUILD_TYPE}. Using default build type: RELEASE." )
  set( CMAKE_BUILD_TYPE "RELEASE" )
endif()

cmessage( STATUS "Using build type: ${CMAKE_BUILD_TYPE}" )

if( ENABLE_BATCH_MODE )
  cmessage( STATUS "-D ENABLE_BATCH_MODE=ON: defining appropriate compile options..." )
  set( ENABLE_COLOR_OUTPUT NO )
  set( ENABLE_TTY_CHECK NO )
  add_definitions( -D GUNDAM_BATCH )
endif( ENABLE_BATCH_MODE )

if( WITH_CACHE_MANAGER )
  cmessage( STATUS "-D WITH_CACHE_MANAGER=ON: enabling cache manager library..." )
  add_definitions( -D GUNDAM_USING_CACHE_MANAGER )

  # uncomment to enable the slow validations (NEVER during productions
  # or normal running).  These are whole code validations and are
  # extremely slow.
  if( CACHE_MANAGER_SLOW_VALIDATION )
    cmessage( STATUS "  Using slow validation for debugging" )
    cmessage( STATUS "  Using slow validation so runs will be very slow" )
    add_definitions( -D CACHE_MANAGER_SLOW_VALIDATION )
  endif( CACHE_MANAGER_SLOW_VALIDATION )

  cmessage( STATUS "Cache manager is enabled. GPU support can be enabled using WITH_CUDA_LIB option." )
else()
  cmessage( STATUS "Cache manager is disabled. Use -D WITH_CACHE_MANAGER=ON if needed." )
endif()


if( WITH_DOXYGEN )
  cmessage( STATUS "-D WITH_DOXYGEN=ON: enabling doxygen build..." )
  # check if Doxygen is installed
  find_package(Doxygen)
  if( DOXYGEN_FOUND )
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/resources/doxygen/Doxygen.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file( ${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY )
    cmessage( STATUS "Doxygen build started." )

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
  else()
    cmessage( FATAL_ERROR "Doxygen need to be installed to generate the doxygen documentation." )
  endif()
endif()
