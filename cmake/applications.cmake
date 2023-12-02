
message("")
cmessage( WARNING "Defining applications...")

add_subdirectory( ${CMAKE_SOURCE_DIR}/src/Applications )


include( CTest )
add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )

configure_file(cmake/generated/build_setup.sh.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/setup.sh"
    DESTINATION ${CMAKE_INSTALL_PREFIX})

