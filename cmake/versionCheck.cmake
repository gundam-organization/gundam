

function( doVersionCheck )
  # Git version
  execute_process(
      COMMAND git describe --tags
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      OUTPUT_VARIABLE VERSION
      RESULT_VARIABLE RETURN_VAL
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(RETURN_VAL EQUAL "0")
    cmessage(STATUS "Git version: ${VERSION}")

    string(REPLACE "-" ";" VERSION_SEP ${VERSION})
    list(LENGTH VERSION_SEP len)

    if( ${len} GREATER_EQUAL 2 )
      list(GET VERSION_SEP 0 VERSION_STR)         # VERSION_SEP[0] = X.X.X
      list(GET VERSION_SEP 1 VERSION_POST_NB)     # VERSION_SEP[1] = NB COMMITS AFTER TAG
      list(GET VERSION_SEP 2 VERSION_POST_COMMIT) # VERSION_COMMIT[2] = "g" + COMMIT HASH
      #    set(GUNDAM_VERSION_TAG "-${VERSION_POST_NB}-${VERSION_POST_COMMIT}")
      set(GUNDAM_VERSION_TAG "f")
    else()
      list(GET VERSION_SEP 0 VERSION_STR)         # VERSION_SEP[0] = X.X.X
      set(GUNDAM_VERSION_TAG "")
    endif()

    # Parsing version number
    string(REPLACE "." ";" VERSION_STR_SEP ${VERSION_STR})
    list(GET VERSION_STR_SEP 0 GUNDAM_VERSION_MAJOR)
    list(GET VERSION_STR_SEP 1 GUNDAM_VERSION_MINOR)
    list(GET VERSION_STR_SEP 2 GUNDAM_VERSION_REVISION)

    set(
        GUNDAM_VERSION_STRING
        "${GUNDAM_VERSION_MAJOR}.${GUNDAM_VERSION_MINOR}.${GUNDAM_VERSION_REVISION}${GUNDAM_VERSION_TAG}"
        PARENT_SCOPE
    )
  else()
    cmessage(WARNING "Bad exit status")
    set( GUNDAM_VERSION_STRING "X.X.X" PARENT_SCOPE )
  endif()

endfunction( doVersionCheck )
