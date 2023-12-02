#
# GUNDAM project CMake main file
#

# cmake_minimum_required() should be called prior to this top-level project()
# 3.5 minimum a priori. Taking a lower version as min will make recent CMake
# version complain about the deprecation version older than 3.5.
# Might require higher version for specific features.
cmake_minimum_required( VERSION 3.5 FATAL_ERROR )

# Define project
project( GUNDAM )


# necessary ?
#include( CTest )

# colored messages
include( ${CMAKE_SOURCE_DIR}/cmake/utils/cmessage.cmake )

# define cmake options
include( ${CMAKE_SOURCE_DIR}/cmake/options.cmake )

# git version checking
include( ${CMAKE_SOURCE_DIR}/cmake/version.cmake )

# checking dependencies
include( ${CMAKE_SOURCE_DIR}/cmake/dependencies.cmake )

# setup compiler options
include( ${CMAKE_SOURCE_DIR}/cmake/compiler.cmake )

# setting up submodule libraries
include( ${CMAKE_SOURCE_DIR}/cmake/submodules.cmake )

# defining GUNDAM modules
include( ${CMAKE_SOURCE_DIR}/cmake/modules.cmake )

# defining GUNDAM applications
include( ${CMAKE_SOURCE_DIR}/cmake/applications.cmake )


message("")
cmessage( WARNING "Identified GUNDAM version: ${GUNDAM_VERSION_STRING}+${GUNDAM_VERSION_POST_NB}-${GUNDAM_VERSION_POST_COMMIT}/${GUNDAM_VERSION_BRANCH}" )

# A command that works to compile gundam from any sub-directory.  You
# can get this aliased to "gundam-build" by sourcing
#
# . "$(git rev-parse --show-toplevel)/cmake/scripts/gundam-setup.sh"
#
# That runs:
# Local Variables:
# compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
# End:
