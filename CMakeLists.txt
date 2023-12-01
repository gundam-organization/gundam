cmake_minimum_required( VERSION 3.12 FATAL_ERROR )

project( GUNDAM )

# colored messages
include( ${CMAKE_SOURCE_DIR}/cmake/utils/cmessage.cmake )

cmessage(STATUS "Setting up CMake files...")

# define cmake options
include( ${CMAKE_SOURCE_DIR}/cmake/options.cmake )
include( ${CMAKE_SOURCE_DIR}/cmake/version.cmake )

# ?
include( CTest )


################################################################################
#                            Check Dependencies
################################################################################

include( ${CMAKE_SOURCE_DIR}/cmake/dependencies.cmake )
include( ${CMAKE_SOURCE_DIR}/cmake/compiler.cmake )
include( ${CMAKE_SOURCE_DIR}/cmake/submodules.cmake )

################################################################################
#                            Specify Target Subdirs
################################################################################

message("")
cmessage(STATUS "Defining modules...")

set( MODULES Utils )

# Add the basic modules
list( APPEND MODULES DialDictionary)
list( APPEND MODULES ParametersManager)
list( APPEND MODULES SamplesManager)
list( APPEND MODULES DatasetManager)
list( APPEND MODULES Propagator)
list( APPEND MODULES Fitter)

if(WITH_CACHE_MANAGER)
    list( APPEND MODULES CacheManager )
endif()

foreach(mod ${MODULES})
    cmessage( STATUS "Adding cmake module: ${mod}" )
    add_subdirectory( ${CMAKE_SOURCE_DIR}/src/${mod} )
endforeach()


message("")
cmessage(STATUS "Defining applications...")

add_subdirectory( ${CMAKE_SOURCE_DIR}/src/Applications )
add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )

configure_file(cmake/build_setup.sh.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh"
    DESTINATION ${CMAKE_INSTALL_PREFIX})

cmessage( STATUS "Identified GUNDAM version: ${GUNDAM_VERSION_STRING}" )

# A command that works to compile gundam from any sub-directory.  You
# can get this aliased to "gundam-build" by sourcing
#
# . "$(git rev-parse --show-toplevel)/cmake/gundam-setup.sh"
#
# That runs:
# Local Variables:
# compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
# End:
