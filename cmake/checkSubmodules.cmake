


function( checkSubmodule )

  list( GET ARGV 0 SELECTED_SUBMODULE )
  cmessage( WARNING "Checking submodule: ${SELECTED_SUBMODULE}" )

  file( GLOB FILES_IN_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/submodules/${SELECTED_SUBMODULE}/*")

  if( ${FILES_IN_DIRECTORY} )
    cmessage( STATUS "Git submodule is present" )
  else()
    cmessage( FATAL_ERROR "Git submodule is not present, please checkout: git submodule update --init --remote --recursive" )
  endif()

endfunction( checkSubmodule )
