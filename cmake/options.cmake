# CMake options

message("")
cmessage( STATUS "Setting up options...")


# general options
option( ENABLE_COLOR_OUTPUT "Enable colored terminal output." ON )
option( ENABLE_TTY_CHECK "Enable check if output is being sent to terminal/TTY." ON )
option( ENABLE_BATCH_MODE "Build to run in a batch queue (affects output)." OFF )
option( ENABLE_DEV_MODE "Enable specific dev related printouts." OFF  )

# extensions
option( WITH_DOXYGEN "Build documentation with doxygen." OFF )
option( WITH_GUNDAM_ROOT_APP "Build app gundamRoot." ON )
option( WITH_GUNDAM_SANDBOX_APP "Build app gundamRoot." OFF )
option( WITH_CACHE_MANAGER "Enable compiling of the cache manager (required for GPU computing)." ON )
option( WITH_CUDA_LIB "Enable CUDA language check (Cache::Manager requires a GPU if CUDA is found)." OFF )
option( WITH_MINUIT2_MISSING "Allow MINUIT2 to be missing" OFF )
option( WITH_PYTHON_INTERFACE "Compile the python interface modules" OFF )
option( WITH_TESTS "Build CMake tests." ON )
option( WITH_GOOGLE_TEST "Enables GoogleTest unit tests." OFF )
option( WITH_PYTORCH "Enable using PyTorch to sample likelihood from pytorch model." OFF )

# compile helper
option( YAMLCPP_DIR "Set custom path to yaml-cpp lib." OFF )

# dev options
option( USE_STATIC_LINKS "Use static link of libraries and apps instead of shared." OFF )
option( CXX_WARNINGS "Enable most C++ warning flags." ON )


# Reading options
##################

# make sure the build type is upper cased
string( TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )

# Set the default built type if it isn't already defined
if( NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set( CMAKE_BUILD_TYPE "RELEASE" )
  cmessage( STATUS "Build type not set. Using default build type: RELEASE." )
elseif( CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
  cmessage( WARNING "Build type manually specified to: RELEASE." )
elseif( CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
  cmessage( WARNING "Build type manually specified to: DEBUG." )
  add_definitions( -D DEBUG_BUILD )
else()
  cmessage( WARNING "Build type not recognised: ${CMAKE_BUILD_TYPE}. Using default build type: RELEASE." )
  set( CMAKE_BUILD_TYPE "RELEASE" )
endif()

cmessage( STATUS "Using build type: ${CMAKE_BUILD_TYPE}" )

if( ENABLE_BATCH_MODE )
  cmessage( WARNING "-D ENABLE_BATCH_MODE=ON: defining appropriate compile options..." )
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
    cmessage( WARNING "  Using slow validation for debugging" )
    cmessage( WARNING "  Using slow validation so runs will be very slow" )
    add_definitions( -D CACHE_MANAGER_SLOW_VALIDATION )
  endif( CACHE_MANAGER_SLOW_VALIDATION )

  if( NOT WITH_CUDA_LIB )
    cmessage( WARNING "Cache manager on without GPU support.  Use -D WITH_CUDA_LIB=ON to enable." )
  endif( NOT WITH_CUDA_LIB )
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

# Ensure the default install prefix is set to the build directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  cmessage( STATUS "CMAKE_INSTALL_PREFIX not set, install prefix is set to the build dir." )
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Install path prefix" FORCE)
endif()

if(WITH_PYTHON_INTERFACE)
  set( PYBIND11_PYTHON_VERSION 3.11.6 )
  set( PYBIND11_FINDPYTHON ON )
  find_package( pybind11 REQUIRED )
endif()
