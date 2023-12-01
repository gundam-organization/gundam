
# Runs on cmake AND make

# reloading cmessage since
include( ${CMAKE_SOURCE_DIR}/cmake/utils/cmessage.cmake )

function( doVersionCheck )
  # Git version
  execute_process(
      COMMAND git describe --tags
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE VERSION
      RESULT_VARIABLE RETURN_VAL
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # default returns
  set( GUNDAM_VERSION_MAJOR "X" PARENT_SCOPE )
  set( GUNDAM_VERSION_MINOR "X" PARENT_SCOPE )
  set( GUNDAM_VERSION_MICRO "X" PARENT_SCOPE )
  set( GUNDAM_VERSION_TAG "" PARENT_SCOPE )
  set( GUNDAM_VERSION_POST_NB "" PARENT_SCOPE )
  set( GUNDAM_VERSION_POST_COMMIT "" PARENT_SCOPE )
  set( GUNDAM_VERSION_STRING "X.X.X" PARENT_SCOPE )
  set( GUNDAM_VERSION_BRANCH "" PARENT_SCOPE )

  if(RETURN_VAL EQUAL "0")
    cmessage( STATUS "Git version: ${VERSION}")

    string(REPLACE "-" ";" VERSION_SEP ${VERSION})
    list(LENGTH VERSION_SEP len)

    if( ${len} GREATER_EQUAL 2 )
      list(GET VERSION_SEP 0 VERSION_STR)         # VERSION_SEP[0] = X.X.X
      list(GET VERSION_SEP 1 VERSION_POST_NB)     # VERSION_SEP[1] = NB COMMITS AFTER TAG
      list(GET VERSION_SEP 2 VERSION_POST_COMMIT) # VERSION_COMMIT[2] = "g" + COMMIT HASH
      #    set(GUNDAM_VERSION_TAG "-${VERSION_POST_NB}-${VERSION_POST_COMMIT}")
      set( GUNDAM_VERSION_TAG "f" )
      set( GUNDAM_VERSION_POST_NB "${VERSION_POST_NB}" PARENT_SCOPE )
      set( GUNDAM_VERSION_POST_COMMIT "${VERSION_POST_COMMIT}" PARENT_SCOPE )

      # Git version
      execute_process(
          COMMAND git rev-parse --abbrev-ref HEAD
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
          OUTPUT_VARIABLE CURRENT_BRANCH_NAME
          RESULT_VARIABLE RETURN_VAL
          OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      if(RETURN_VAL EQUAL "0")
        set( GUNDAM_VERSION_BRANCH "${CURRENT_BRANCH_NAME}" PARENT_SCOPE )
        cmessage( STATUS "Git branch: ${CURRENT_BRANCH_NAME}")
      endif()


    else()
      list(GET VERSION_SEP 0 VERSION_STR)         # VERSION_SEP[0] = X.X.X
      set(GUNDAM_VERSION_TAG "")
    endif()

    # Parsing version number
    string(REPLACE "." ";" VERSION_STR_SEP ${VERSION_STR})
    list(GET VERSION_STR_SEP 0 VERSION_MAJOR)
    list(GET VERSION_STR_SEP 1 VERSION_MINOR)
    list(GET VERSION_STR_SEP 2 VERSION_MICRO)

    set( GUNDAM_VERSION_MAJOR "${VERSION_MAJOR}" PARENT_SCOPE )
    set( GUNDAM_VERSION_MINOR "${VERSION_MINOR}" PARENT_SCOPE )
    set( GUNDAM_VERSION_MICRO "${VERSION_MICRO}" PARENT_SCOPE )

    # Building the version string
    set( GUNDAM_VERSION_STRING
        "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_MICRO}${GUNDAM_VERSION_TAG}"
        PARENT_SCOPE
    )

  else()
    cmessage( WARNING "Could not find git version." )
  endif()

endfunction( doVersionCheck )

# Check last git tag
doVersionCheck()


#####################
# CMake Generated
#####################

cmessage( STATUS "Generating version config header: ${GENERATE_DIR_FOR_VERSION_CHECK}/generated/VersionConfig.h" )
configure_file( ${CMAKE_SOURCE_DIR}/cmake/VersionConfig.h.in ${GENERATE_DIR_FOR_VERSION_CHECK}/generated/VersionConfig.h )


