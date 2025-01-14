

if( ENABLE_TESTS )
  message("")
  cmessage( WARNING "Defining tests...")
  include( CTest )
  add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )
endif()

