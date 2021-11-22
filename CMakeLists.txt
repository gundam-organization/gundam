cmake_minimum_required (VERSION 3.5 FATAL_ERROR)

# Configure CMake project
project(gundam)
set (gundam_VERSION_MAJOR 1)
set (gundam_VERSION_MINOR 2)
set (gundam_VERSION_REVISION 0)
set (gundam_VERSION_TAG f) # f means forward -> include commits after a given release
set (gundam_VERSION_STRING "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

option(CMAKE_CXX_EXTENSIONS "Enable GNU extensions to C++ langauge (-std=gnu++14)." OFF)
option(CXX_MARCH_FLAG "Enable cpu architecture specific optimzations." OFF)
option(CXX_WARNINGS "Enable most C++ warning flags." ON)
option(COLOR_OUTPUT "Enable colored terminal output." ON)
option(TTYCHECK "Enable check if output is being sent to terminal/TTY." ON)
option(WITH_OPENMP "Build with OpenMP libraries" OFF )
option(YAMLCPP_DIR "Set custom path to yaml-cpp lib" "" )

set(VERBOSE TRUE)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
else()
    set(CMAKE_MACOSX_RPATH 0)
    set(CMAKE_SKIP_BUILD_RPATH TRUE)
    set(CMAKE_SKIP_RPATH TRUE)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/cmessage.cmake)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#Changes default install path to be a subdirectory of the build dir.
#Can set build dir at configure time with -DCMAKE_INSTALL_PREFIX=/install/path
if(CMAKE_INSTALL_PREFIX STREQUAL "" OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
endif()

cmessage(STATUS "CMAKE_INSTALL_PREFIX: \"${CMAKE_INSTALL_PREFIX}\"")

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE DEBUG)
elseif(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE DEBUG)
endif()

if(NOT DEFINED USE_STATIC_LINKS)
    set(USE_STATIC_LINKS 0)
endif()

#if(NOT DEFINED ENV{XSLLHFITTER})
#  cmessage(FATAL_ERROR "$XSLLHFITTER is not defined, please source the package setup.sh before building.")
#endif()

cmessage(STATUS "CMAKE_BUILD_TYPE: \"${CMAKE_BUILD_TYPE}\"")


################################################################################
#                            Check Dependencies
################################################################################

set( LINK_LIBARY_LIST "" )

# ROOT
if ( NOT DEFINED ENV{ROOTSYS} )
  cmessage(FATAL_ERROR "$ROOTSYS is not defined, please set up root first.")
else()
  cmessage(STATUS "Using ROOT installed at $ENV{ROOTSYS}")
  set(CMAKE_ROOTSYS $ENV{ROOTSYS})
endif()

#If you want to try an use the terminally buggy ROOT CMake scripts
if (USEROOTCMAKE)
  include ($ENV{ROOTSYS}/etc/cmake/FindROOT.cmake)
  if ( NOT ROOT_FOUND )
    cmessage(FATAL_ERROR "[ROOT]: FindROOT.cmake could not be found, or did not execute as expected is ROOT installed in $ROOTSYS = $ENV{ROOTSYS}?")
  endif()
else()
  cmessage(STATUS "Including local GENERATE_ROOT_DICTIONARY implementation.")
  include(${CMAKE_SOURCE_DIR}/cmake/GenROOTDictionary.cmake)
endif()

execute_process (COMMAND root-config --cflags OUTPUT_VARIABLE ROOT_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND root-config --libs OUTPUT_VARIABLE ROOT_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND root-config --version OUTPUT_VARIABLE ROOT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
cmessage(STATUS "[ROOT]: root-config --version: ${ROOT_VERSION}")
cmessage(STATUS "[ROOT]: root-config --cflags: ${ROOT_CXX_FLAGS}")
cmessage(STATUS "[ROOT]: root-config --libs: ${ROOT_LD_FLAGS}")

# YAML-CPP
find_package( YAMLCPP REQUIRED )
if( NOT YAMLCPP_FOUND )
    cmessage(FATAL_ERROR "yaml-cpp library not found.")
endif()
cmessage(STATUS " - yaml-cpp include directory: ${YAMLCPP_INCLUDE_DIR}")
cmessage(STATUS " - yaml-cpp lib: ${YAMLCPP_LIBRARY}")
include_directories( ${YAMLCPP_INCLUDE_DIR} )
#    target_link_libraries( OptionParserExample ${YAMLCPP_LIBRARY} )
list(APPEND LINK_LIBARY_LIST ${YAMLCPP_LIBRARY})


################################################################################
#                       SubModules
################################################################################

# SubModules

## Add the CmdLineParser
include_directories(submodules/simple-cpp-cmd-line-parser/include)
if( YAMLCPP_FOUND )
    add_definitions( -D CMDLINEPARSER_YAML_CPP_ENABLED=1 )
endif()

## Add the GenericToolbox
include_directories(submodules/cpp-generic-toolbox/include)
add_definitions( -D PROGRESS_BAR_ENABLE_RAINBOW=1 )
add_definitions( -D PROGRESS_BAR_FILL_TAG="\\\"T2K\#"\\\" )

## Add the Logger
include_directories(submodules/simple-cpp-logger/include)


if(${CMAKE_BUILD_TYPE} MATCHES Release)
    add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
    add_definitions( -D LOGGER_PREFIX_LEVEL=3 )
    add_definitions( -D LOGGER_ENABLE_COLORS=1 )
    add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=1 )
    add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER}"\\\" )
else()
    add_definitions( -D LOGGER_MAX_LOG_LEVEL_PRINTED=6 )
    add_definitions( -D LOGGER_PREFIX_LEVEL=3 )
    add_definitions( -D LOGGER_ENABLE_COLORS=1 )
    add_definitions( -D LOGGER_ENABLE_COLORS_ON_USER_HEADER=1 )
    add_definitions( -D LOGGER_PREFIX_FORMAT="\\\"{TIME} {USER_HEADER} {FILELINE}"\\\" )
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
    #message(WARNING "You are using an untested compiler.")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if( WITH_OPENMP )
    cmessage(STATUS "Compiling with OpenMP")
    include(FindOpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        cmessage(STATUS " - OpenMP include directory: ${OpenMP_CXX_INCLUDE_DIR}")
        cmessage(STATUS " - OpenMP libs: ${OpenMP_CXX_LIB_NAMES}")
    else()
        cmessage(ERROR "OpenMP not found.")
    endif()
else()
    cmessage(STATUS "Compiling without OpenMP (to enable it, add \"-D WITH_OPENMP\" )")
endif()

if(CXX_MARCH_FLAG)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

if(CXX_WARNINGS)
#    set(CXX_WARNING_FLAGS "-Wall -Wno-unused-variable -Wno-sign-compare -Wno-unused-function -Wno-unused-but-set-variable -Wno-reorder")
    set(CXX_WARNING_FLAGS "-Wall -Wno-unused-variable -Wno-sign-compare -Wno-unused-function -Wformat=0 -Wno-reorder")
endif()

if(NOT COLOR_OUTPUT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOCOLOR")
endif()

if(TTYCHECK)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTTYCHECK")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${ROOT_CXX_FLAGS} ${CXX_WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC ${ROOT_CXX_FLAGS} ${CXX_WARNING_FLAGS}")

if(VERBOSE)
  cmessage (STATUS "C++ Compiler      : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  cmessage (STATUS "C++ Release flags : ${CMAKE_CXX_FLAGS_RELEASE}")
  cmessage (STATUS "C++ Debug flags   : ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

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
        ErrorPropagator
        Fitter
)
#set(MODULETargets "")

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
