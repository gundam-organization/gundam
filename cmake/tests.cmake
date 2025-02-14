#######################################################################
# Handle the testing infrastructure in the "tests" directory.  By
# default, WITH_TESTS only enables the most basic testing.
#######################################################################


if( WITH_TESTS )
  message("")
  cmessage( WARNING "Defining tests...")
  include( CTest )
  add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )
endif()

