cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (NUWRO)

set (NUWRO_VERSION_MAJOR 11)
set (NUWRO_VERSION_MINOR 18) #The q+1'th letter of the alphabet
set (NUWRO_VERSION_REVISION 0)

enable_language(Fortran)

set (VERBOSE TRUE)

set (CMAKE_SKIP_BUILD_RPATH TRUE)

if(CMAKE_INSTALL_PREFIX STREQUAL "" OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX: \"${CMAKE_INSTALL_PREFIX}\"")


if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE DEBUG)
elseif(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE DEBUG)
endif()

message(STATUS "CMAKE_BUILD_TYPE: \"${CMAKE_BUILD_TYPE}\"")

if(NUWRO_BUILD_VALID STREQUAL "")
  set(NUWRO_BUILD_VALID TRUE)
elseif(NOT DEFINED NUWRO_BUILD_VALID)
  set(NUWRO_BUILD_VALID TRUE)
endif()

message(STATUS "[CONFIG]: NUWRO_BUILD_VALID=\"${NUWRO_BUILD_VALID}\"")

if(NUWRO_SINGLE_LIB STREQUAL "")
  set(NUWRO_SINGLE_LIB TRUE)
elseif(NOT DEFINED NUWRO_SINGLE_LIB)
  set(NUWRO_SINGLE_LIB TRUE)
endif()

message(STATUS "[CONFIG]: NUWRO_SINGLE_LIB=\"${NUWRO_SINGLE_LIB}\"")

if(NUWRO_USE_SPP_BKGSCALE STREQUAL "")
  set(NUWRO_USE_SPP_BKGSCALE TRUE)
elseif(NOT DEFINED NUWRO_USE_SPP_BKGSCALE)
  set(NUWRO_USE_SPP_BKGSCALE TRUE)
endif()

################################################################################
#                            Check Dependencies
################################################################################

##################################  ROOT  ######################################
if ( NOT DEFINED ENV{ROOTSYS} )
  message (FATAL_ERROR "$ROOTSYS is not defined, please set up root first.")
endif()

include ($ENV{ROOTSYS}/etc/cmake/FindROOT.cmake)

if ( NOT ROOT_FOUND )
  message (FATAL_ERROR "[ROOT]: FindROOT.cmake could not be found, or did not execute as expected is ROOT installed in $ROOTSYS = $ENV{ROOTSYS}?")
else()
  execute_process (COMMAND root-config --cflags OUTPUT_VARIABLE ROOT_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process (COMMAND root-config --libs OUTPUT_VARIABLE ROOT_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process (COMMAND root-config --version OUTPUT_VARIABLE ROOT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  message ( STATUS "[ROOT]: root-config --version: " ${ROOT_VERSION})
  message ( STATUS "[ROOT]: root-config --cflags: " ${ROOT_CXX_FLAGS} )
  message ( STATUS "[ROOT]: root-config --libs: " ${ROOT_LD_FLAGS} )
endif()

if(${ROOT_VERSION} MATCHES "6.*")
  set(ISROOT6 TRUE)
  message( STATUS "[ROOT]: Using ROOT 6.")
else()
  set(ISROOT6 FALSE)
endif()

################################## Pythia6 ####################################

if(DEFINED ENV{PYTHIA6})
  set(PYTHIA6LIBLOC "-L$ENV{PYTHIA6}")
else()
  set(PYTHIA6LIBLOC)
endif()

################################## COMPILER ####################################

set(NUWRO_CFLAGS "")
if(NUWRO_USE_SPP_BKGSCALE)
  set(NUWRO_CFLAGS "${NUWRO_CFLAGS} -DUSE_SPP_BKGSCALE")
endif()

set(CXX_WARNINGS "-Wall -Wno-unused-variable -Wno-sign-compare -Wno-unused-function -Wno-unused-but-set-variable -Wno-reorder")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 ${ROOT_CXX_FLAGS} ${CXX_WARNINGS} ${NUWRO_CFLAGS} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -O3 ${ROOT_CXX_FLAGS} ${CXX_WARNINGS} ")

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS_DEBUG} ${ROOT_LD_FLAGS} -lEG -lEGPythia6 -lGeom -lMinuit ${PYTHIA6LIBLOC} -lPythia6")
  set(CURRENT_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
elseif(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS_RELEASE} ${ROOT_LD_FLAGS} -lEG -lEGPythia6 -lGeom -lMinuit ${PYTHIA6LIBLOC} -lPythia6 -Wl,--no-as-needed")
  set(CURRENT_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})
else()
  message(FATAL_ERROR "[ERROR]: Unknown CMAKE_BUILD_TYPE (\"${CMAKE_BUILD_TYPE}\"): Should be \"DEBUG\" or \"RELEASE\".")
endif()

if (VERBOSE)
  message (STATUS "C++ Compiler      : " ${CXX_COMPILER_NAME})
  message (STATUS "    Release flags : " ${CMAKE_CXX_FLAGS_RELEASE})
  message (STATUS "    Debug flags   : " ${CMAKE_CXX_FLAGS_DEBUG})
  message (STATUS "    Link Flags    : " ${CMAKE_LINK_FLAGS})
endif()

################################################################################
#                            Specify Target Subdirs
################################################################################

add_subdirectory(src)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/setup.sh
  "source $ENV{ROOTSYS}/bin/thisroot.sh\n")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/setup.sh "export NUWRO=${PROJECT_SOURCE_DIR}\n")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/setup.sh "export PATH=${CMAKE_INSTALL_PREFIX}/bin:\${PATH}\n")
if(DEFINED ENV{PYTHIA6})
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/setup.sh "export PYTHIA6=$ENV{PYTHIA6}\n")
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/setup.sh "export LD_LIBRARY_PATH=\${PYTHIA6}:\${LD_LIBRARY_PATH}\n")
endif()
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/setup.sh "export LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:\${LD_LIBRARY_PATH}")
