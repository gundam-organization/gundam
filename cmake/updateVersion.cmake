
#####################
# Supposed to be ran at build time
#####################

include( ${CMAKE_SOURCE_DIR}/cmake/cmessage.cmake )
include( ${CMAKE_SOURCE_DIR}/cmake/versionCheck.cmake )

# Check last git tag
doVersionCheck()


#####################
# CMake Generated
#####################

cmessage( STATUS "Generating version config header: ${GENERATE_DIR_FOR_VERSION_CHECK}/generated/VersionConfig.h" )
configure_file( ${CMAKE_SOURCE_DIR}/cmake/VersionConfig.h.in ${GENERATE_DIR_FOR_VERSION_CHECK}/generated/VersionConfig.h )


