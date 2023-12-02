
message("")
cmessage( WARNING "Checking out git version...")

# Printout version & generate header
cmessage( STATUS "Pre-checking GUNDAM version" )
set(GENERATE_DIR_FOR_VERSION_CHECK ${CMAKE_BINARY_DIR} CACHE STRING "GENERATE_DIR_FOR_VERSION_CHECK")
add_custom_target( preBuildVersionCheck
    COMMAND ${CMAKE_COMMAND} -D GENERATE_DIR_FOR_VERSION_CHECK=${GENERATE_DIR_FOR_VERSION_CHECK} -P ${CMAKE_SOURCE_DIR}/cmake/utils/git-version.cmake
    COMMENT "Checking GUNDAM version with git files..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Run the command while building makefiles
include( ${CMAKE_SOURCE_DIR}/cmake/utils/git-version.cmake )
