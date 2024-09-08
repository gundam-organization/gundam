
message("")
cmessage( WARNING "Defining tests...")

if( DISABLE_GOOGLE_TESTS )
  cmessage( ALERT "Google tests disabled (DISABLE_GOOGLE_TESTS=ON). Skipping...")
else()
  include( CTest )
  add_subdirectory( ${CMAKE_SOURCE_DIR}/tests )
endif()

