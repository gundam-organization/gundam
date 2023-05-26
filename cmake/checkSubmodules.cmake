


function( checkSubmodule )
  list( GET ARGV 0 SELECTED_SUBMODULE )

  cmessage( WARNING "Checking submodule: ${SELECTED_SUBMODULE}" )

  execute_process(
      COMMAND git rev-parse --is-inside-work-tree
      RESULT_VARIABLE GIT_INSIDE_WORK_TREE
      OUTPUT_QUIET
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/submodules/${SELECTED_SUBMODULE}
  )

  if (GIT_INSIDE_WORK_TREE EQUAL 0)
    execute_process(
        COMMAND git submodule status --recursive --quiet
        RESULT_VARIABLE GIT_SUBMODULE_STATUS
        OUTPUT_QUIET
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/submodules/${SELECTED_SUBMODULE}
    )

    if (GIT_SUBMODULE_STATUS EQUAL 0)
      # Submodule is present
      cmessage( STATUS "Git submodule is present" )
      # Additional actions for when the submodule is present
    else ()
      # Submodule is not present
      cmessage( FATAL_ERROR "Git submodule is not present, please checkout: git submodule update --init --remote --recursive" )
      # Additional actions for when the submodule is not present
    endif ()
  else ()
    # Not inside a Git repository
    cmessage( ERROR "Not inside a Git repository")
    # Additional actions for when not inside a Git repository
  endif ()

endfunction( checkSubmodule )
