#######################################################################
# Handle the testing infrastructure in the "tests" directory.  By
# default, ENABLE_TESTS only enables the most basic testing.
#######################################################################

if( ENABLE_TESTS )
  cmessage( STATUS "Build GUNDAM test and validations")
  include( CTest )
  add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )
else()
  cmessage(WARNING "GUNDAM tests and validations are disabled.")
endif()

