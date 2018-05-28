cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (xsLLhFitter)

set (xsLLhFitter_VERSION_MAJOR 0)
set (xsLLhFitter_VERSION_MINOR 1) #The q+1'th letter of the alphabet
set (xsLLhFitter_VERSION_REVISION 0)

set (VERBOSE TRUE)

set (CMAKE_SKIP_BUILD_RPATH TRUE)

include(${CMAKE_SOURCE_DIR}/cmake/cmessage.cmake)

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

if(NOT DEFINED ENV{XSLLHFITTER})
  cmessage(FATAL_ERROR "$XSLLHFITTER is not defined, please source the package setup.sh before building.")
endif()

cmessage(STATUS "CMAKE_BUILD_TYPE: \"${CMAKE_BUILD_TYPE}\"")

################################################################################
#                            Check Dependencies
################################################################################

##################################  ROOT  ######################################
if ( NOT DEFINED ENV{ROOTSYS} )
  cmessage (FATAL_ERROR "$ROOTSYS is not defined, please set up root first.")
else()
  cmessage(STATUS "Using ROOT installed at $ENV{ROOTSYS}")
  set(CMAKE_ROOTSYS $ENV{ROOTSYS})
endif()

#If you want to try an use the terminally buggy ROOT CMake scripts
if (USEROOTCMAKE)
  include ($ENV{ROOTSYS}/etc/cmake/FindROOT.cmake)
  if ( NOT ROOT_FOUND )
    cmessage (FATAL_ERROR "[ROOT]: FindROOT.cmake could not be found, or did not execute as expected is ROOT installed in $ROOTSYS = $ENV{ROOTSYS}?")
  endif()
else()
  cmessage(STATUS "Including local GENERATE_ROOT_DICTIONARY implementation.")
  include(${CMAKE_SOURCE_DIR}/cmake/GenROOTDictionary.cmake)
endif()

execute_process (COMMAND root-config --cflags OUTPUT_VARIABLE ROOT_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND root-config --libs OUTPUT_VARIABLE ROOT_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process (COMMAND root-config --version OUTPUT_VARIABLE ROOT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
cmessage ( STATUS "[ROOT]: root-config --version: " ${ROOT_VERSION})
cmessage ( STATUS "[ROOT]: root-config --cflags: " ${ROOT_CXX_FLAGS} )
cmessage ( STATUS "[ROOT]: root-config --libs: " ${ROOT_LD_FLAGS} )

################################## COMPILER ####################################

set(CXX_WARNINGS "-Wall -Wno-unused-variable -Wno-sign-compare -Wno-unused-function -Wno-unused-but-set-variable -Wno-reorder")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG_MSG -fopenmp -g -O0 ${ROOT_CXX_FLAGS} ${CXX_WARNINGS} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -march=native -fPIC -O3 ${ROOT_CXX_FLAGS} ${CXX_WARNINGS} ")

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CURRENT_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CURRENT_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})
else()
  cmessage(FATAL_ERROR "[ERROR]: Unknown CMAKE_BUILD_TYPE (\"${CMAKE_BUILD_TYPE}\"): Should be \"DEBUG\" or \"RELEASE\".")
endif()

if (VERBOSE)
  cmessage (STATUS "C++ Compiler      : " ${CXX_COMPILER_NAME})
  cmessage (STATUS "    Release flags : " ${CMAKE_CXX_FLAGS_RELEASE})
  cmessage (STATUS "    Debug flags   : " ${CMAKE_CXX_FLAGS_DEBUG})
  cmessage (STATUS "    Link Flags    : " ${CMAKE_LINK_FLAGS})
endif()

################################################################################
#                            Specify Target Subdirs
################################################################################

set(MODULES
  anaevents
  fitparam
  xsecfit
  utils
  )
set(MODULETargets "")
foreach(mod ${MODULES})
  cmessage (STATUS "Configuring directory: ${mod}/src")
  add_subdirectory(${mod}/src)
endforeach()

add_subdirectory(src)

configure_file(cmake/build_setup.sh.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh" DESTINATION ${CMAKE_INSTALL_PREFIX})
