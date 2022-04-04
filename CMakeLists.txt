cmake_minimum_required (VERSION 3.12 FATAL_ERROR)

# Configure CMake project
project(gundam)
set (gundam_VERSION_MAJOR 1)
set (gundam_VERSION_MINOR 3)
set (gundam_VERSION_REVISION 5)
set (gundam_VERSION_TAG "f") # f means forward -> include commits after a given release
set (gundam_VERSION_STRING "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

option(CMAKE_CXX_EXTENSIONS "Enable GNU extensions to C++ langauge (-std=gnu++14)." OFF)
option(CXX_MARCH_FLAG "Enable cpu architecture specific optimzations." OFF)
option(CXX_WARNINGS "Enable most C++ warning flags." ON)
option(COLOR_OUTPUT "Enable colored terminal output." ON)
option(TTYCHECK "Enable check if output is being sent to terminal/TTY." ON)
option(WITH_OPENMP "Build with OpenMP libraries" OFF )
option(WITH_CUDA "Build with CUDA (if available)" ON)
option(YAMLCPP_DIR "Set custom path to yaml-cpp lib" OFF )
option(ENABLE_DEV_MODE "Enable specific dev related printouts" OFF )
option(BATCH_MODE "Build to run in a batch queue (affects output)" OFF)

set(VERBOSE TRUE)

# Check for the availability of CUDA
if(WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message("-- CUDA support enabled")
    enable_language(CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # The default is taken from the CUDAARCHS environment variable.  If it
      # isn't set, then set it to the earliest non-deprecated architecture
      #    2022: architectures before 52 are deprecated.
      if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.23)
        # After cmake 3.23, this can be set to all or all-major
        set(CMAKE_CUDA_ARCHITECTURES all)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 52)
      endif()
    endif()
    message("-- CUDA compilation architectures: \"${CMAKE_CUDA_ARCHITECTURES}\"")
  else()
    message("-- CUDA support requested, but not supported by this environment")
  endif(CMAKE_CUDA_COMPILER)
else()
  message("-- CUDA support disabled")
endif(WITH_CUDA)

include(${CMAKE_SOURCE_DIR}/cmake/cmessage.cmake)

#Changes default install path to be a subdirectory of the build dir.
#Should set the installation dir at configure time with
#-DCMAKE_INSTALL_PREFIX=/install/path
if(NOT DEFINED CMAKE_INSTALL_PREFIX
    OR CMAKE_INSTALL_PREFIX STREQUAL ""
    OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
endif()

cmessage(STATUS "CMAKE_INSTALL_PREFIX: \"${CMAKE_INSTALL_PREFIX}\"")

if(NOT DEFINED CMAKE_BUILD_TYPE
    OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Debug)
endif()
cmessage(STATUS "CMAKE_BUILD_TYPE: \"${CMAKE_BUILD_TYPE}\"")

if(NOT DEFINED USE_STATIC_LINKS)
    set(USE_STATIC_LINKS 0)
endif()

if(BATCH_MODE)
  set(COLOR_OUTPUT NO)
  set(TTYCHECK NO)
  add_definitions(-DGUNDAM_BATCH)
endif(BATCH_MODE)

################################################################################
#                            Check Dependencies
################################################################################

# ROOT
if(NOT DEFINED ENV{ROOTSYS} )
  cmessage(FATAL_ERROR "$ROOTSYS is not defined, please set up root first.")
else()
  cmessage(STATUS "Using ROOT installed at $ENV{ROOTSYS}")
  set(CMAKE_ROOTSYS $ENV{ROOTSYS})
endif()

#If you want to try an use the terminally buggy ROOT CMake scripts
find_package(ROOT
  REQUIRED COMPONENTS Geom Physics Matrix MathCore Tree RIO
  OPTIONAL_COMPONENTS Minuit2)
if(ROOT_FOUND)
  cmessage(STATUS "[ROOT]: ROOT found")
  include(${ROOT_USE_FILE})
  cmessage(STATUS "[ROOT]: ROOT package found ${ROOT_LIBRARIES}")
  if (NOT ROOT_minuit2_FOUND)
    # Minuit2 wasn't found, but make really sure before giving up.
    execute_process (COMMAND root-config --has-minuit2
      OUTPUT_VARIABLE ROOT_minuit2_FOUND
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif(NOT ROOT_minuit2_found)
else(ROOT_FOUND)
   cmessage(STATUS "Including local GENERATE_ROOT_DICTIONARY implementation.")
   include(${CMAKE_SOURCE_DIR}/cmake/GenROOTDictionary.cmake)
   execute_process(COMMAND root-config --cflags
     OUTPUT_VARIABLE ROOT_CXX_FLAGS
     OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND root-config --libs
     OUTPUT_VARIABLE ROOT_LIBRARIES
     OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND root-config --version
     OUTPUT_VARIABLE ROOT_VERSION
     OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process (COMMAND root-config --ldflags
     OUTPUT_VARIABLE ROOT_LINK_FLAGS
     OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process (COMMAND root-config --has-minuit2
     OUTPUT_VARIABLE ROOT_minuit2_FOUND
     OUTPUT_STRIP_TRAILING_WHITESPACE)
   cmessage(STATUS "[ROOT]: root-config --version: ${ROOT_VERSION}")
   cmessage(STATUS "[ROOT]: root-config --libs: ${ROOT_LIBRARIES}")
   cmessage(STATUS "[ROOT]: root-config --cflags: ${ROOT_CXX_FLAGS}")
   cmessage(STATUS "[ROOT]: root-config --ldflags: ${ROOT_LINK_FLAGS}")
   add_compile_options("SHELL:${ROOT_CXX_FLAGS}")
   add_link_options("SHELL:${ROOT_LINK_FLAGS}")
endif(ROOT_FOUND)

if (NOT ROOT_minuit2_FOUND)
  cmessage(STATUS "[ROOT]:Rebuild root using -Dminuit2=on in the cmake command")
  cmessage(FATAL_ERROR "[ROOT]: minuit2 is required")
endif(NOT ROOT_minuit2_FOUND)

# YAML-CPP
find_package( yaml-cpp REQUIRED HINTS ${YAMLCPP_DIR} )
if(NOT yaml-cpp_FOUND)
    cmessage(FATAL_ERROR "yaml-cpp library not found.")
endif()
include_directories( ${YAML_CPP_INCLUDE_DIR} )
cmessage(STATUS "Custom yaml-cpp installation: ${YAMLCPP_DIR}")
cmessage(STATUS "yaml-cpp include directory: ${YAML_CPP_INCLUDE_DIR}")
cmessage(STATUS "yaml-cpp lib: ${YAML_CPP_LIBRARIES}")
if( "${YAML_CPP_INCLUDE_DIR} " STREQUAL " ")
    # WORKAROUND FOR CCLYON (old cmake version/pkg)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    set(YAMLCPP_INSTALL_DIR ${YAMLCPP_DIR})
    find_package( YAMLCPP REQUIRED )
    if( NOT YAMLCPP_FOUND )
        cmessage(FATAL_ERROR "yaml-cpp library not found.")
    endif()
    cmessage(STATUS " - yaml-cpp include directory: ${YAMLCPP_INCLUDE_DIR}")
    cmessage(STATUS " - yaml-cpp lib: ${YAMLCPP_LIBRARY}")
    if( "${YAMLCPP_INCLUDE_DIR} " STREQUAL " ")
        cmessage(FATAL_ERROR "empty YAMLCPP_INCLUDE_DIR returned.")
    endif()
    include_directories( ${YAMLCPP_INCLUDE_DIR} )
    set(YAML_CPP_LIBRARIES ${YAMLCPP_LIBRARY})
#    list(APPEND LINK_LIBARY_LIST ${YAMLCPP_LIBRARY})
endif()

if (WITH_CUDA)
  add_definitions( -DGUNDAM_USING_CUDA )

  # uncomment to enable the slow validations (NEVER during productions
  # or normal running).  These are whole code validations and are
  # extremely slow.
  # add_definitions( -DCACHE_MANAGER_SLOW_VALIDATION)

  cmessage(STATUS "Enable GPU support (compiled, but only used if CUDA enabled)")
endif()

################################################################################
#                       SubModules
################################################################################

# SubModules: These are just adding the code directly, and not being compiled
# as stand-alone projects.  This has the danger that it's not fully configured,
# but should be OK

## Add the CmdLineParser
# Reproduce needed parts of the simple-cpp-cmd-line-parser CMakeLists.txt
include_directories(submodules/simple-cpp-cmd-line-parser/include)
if(yaml-cpp_FOUND)
    add_definitions( -DCMDLINEPARSER_YAML_CPP_ENABLED=1 )
endif()

## Add the GenericToolbox
# Reproduce needed parts of the cpp-generic-toolbox CMakeLists.txt
include_directories(submodules/cpp-generic-toolbox/include)
add_definitions( -D PROGRESS_BAR_FILL_TAG="\\\"T2K\#"\\\" )
if (COLOR_OUTPUT)
  add_definitions( -D PROGRESS_BAR_ENABLE_RAINBOW=1 )
else (COLOR_OUTPUT)
  # add_definitions( -D PROGRESS_BAR_ENABLE_RAINBOW=0 )
  add_definitions( -D CPP_GENERIC_TOOLBOX_NOCOLOR )
endif (COLOR_OUTPUT)
if (BATCH_MODE)
  add_definitions( -D CPP_GENERIC_TOOLBOX_BATCH )
endif(BATCH_MODE)

## Add the Logger
# Reproduce needed parts of the simple-cpp-logger CMakeLists.txt
include_directories(submodules/simple-cpp-logger/include)
if(${CMAKE_BUILD_TYPE} MATCHES Debug OR ${ENABLE_DEV_MODE})
    cmessage(STATUS "Logger set in dev mode.")
    add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
    add_definitions( -D LOGGER_PREFIX_LEVEL=3 )
    add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER} {FILELINE}"\\\" )
    if(NOT COLOR_OUTPUT)
      add_definitions( -D LOGGER_ENABLE_COLORS=0 )
      add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=0 )
    else()
      add_definitions( -D LOGGER_ENABLE_COLORS=1 )
      add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=1 )
    endif()
else()
    cmessage(STATUS "Logger set in release mode.")
    add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
    add_definitions( -D LOGGER_PREFIX_LEVEL=3 )
    add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER}"\\\" )
    if(NOT COLOR_OUTPUT)
      add_definitions( -D LOGGER_ENABLE_COLORS=0 )
      add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=0 )
    else()
      add_definitions( -D LOGGER_ENABLE_COLORS=1 )
      add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=1 )
    endif()
endif()

################################################################################
#                            Compiler Setup
################################################################################

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.5)
        cmessage(STATUS "Detected GCC version: ${CMAKE_CXX_COMPILER_VERSION}")
        cmessage(FATAL_ERROR "GCC version must be at least 4.8.5")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
        cmessage(STATUS "Detected Clang version: ${CMAKE_CXX_COMPILER_VERSION}")
        cmessage(FATAL_ERROR "Clang version must be at least 3.3")
    endif()
else()
  cmessage(WARNING "You are using an untested compiler.")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if( WITH_OPENMP )
    cmessage(STATUS "Compiling with OpenMP")
    include(FindOpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        cmessage(STATUS "OpenMP include directory: ${OpenMP_CXX_INCLUDE_DIR}")
        cmessage(STATUS "OpenMP libs: ${OpenMP_CXX_LIB_NAMES}")
    else()
        cmessage(ERROR "OpenMP not found.")
    endif()
else()
    cmessage(STATUS "Compiling without OpenMP, enable with \"-D WITH_OPENMP\"")
endif()

if(CXX_MARCH_FLAG)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

if(CXX_WARNINGS)
  cmessage(STATUS "Enable CXX warnings")
  add_compile_options(-Wall)
  add_compile_options(-Wno-unused-variable)
  add_compile_options(-Wno-sign-compare)
  add_compile_options(-Wno-unused-function)
  # add_compile_options(-Wformat=0)
  # add_compile_options(-Wno-reorder)
endif()

if(NOT COLOR_OUTPUT)
  add_definitions(-D NOCOLOR)
endif()

if(TTYCHECK)
  add_definitions(-D TTYCHECK)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -g")

cmessage (STATUS "C++ Compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
cmessage (STATUS "C++ Release flags : ${CMAKE_CXX_FLAGS_RELEASE}")
cmessage (STATUS "C++ Debug flags   : ${CMAKE_CXX_FLAGS_DEBUG}")

################################################################################
# CMake Generated
###############################################################################

configure_file( ${CMAKE_SOURCE_DIR}/cmake/versionConfig.h.in ${CMAKE_BINARY_DIR}/generated/versionConfig.h )
include_directories( ${CMAKE_BINARY_DIR}/generated/ )

################################################################################
#                            Specify Target Subdirs
################################################################################

set( MODULES
        Utils
        anaevents
        FitParameters
        FitSamples
        CacheManager
        ErrorPropagator
        Fitter
)

# Each module will add it's primary target to this and they will be
# automatically added to the executables.
set(MODULETargets "")

cmessage (STATUS "Setting up include directories...")
include_directories( ${CMAKE_SOURCE_DIR}/include )
foreach(mod ${MODULES})
    cmessage (STATUS "Including headers of module: ${mod}")
    include_directories( ${CMAKE_SOURCE_DIR}/src/${mod}/include )
endforeach()


cmessage (STATUS "Configuring modules...")
foreach(mod ${MODULES})
    cmessage (STATUS "Adding cmake module: ${mod}")
    add_subdirectory( ${CMAKE_SOURCE_DIR}/src/${mod} )
endforeach()

add_subdirectory( ${CMAKE_SOURCE_DIR}/src/Applications )

configure_file(cmake/build_setup.sh.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh" DESTINATION ${CMAKE_INSTALL_PREFIX})

cmessage( STATUS "Tagged Version ${gundam_VERSION_STRING}" )
