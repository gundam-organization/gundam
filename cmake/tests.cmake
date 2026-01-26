#######################################################################
# Handle the testing infrastructure in the "tests" directory.  By
# default, WITH_TESTS only enables the most basic testing.
#######################################################################


if( WITH_TESTS )
  message("")
  cmessage( STATUS "Defining tests...")
  include( CTest )
  add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )
else()
  message("")
  cmessage( WARNING "Skipping tests...")
endif()

