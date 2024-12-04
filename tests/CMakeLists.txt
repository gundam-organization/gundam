# Define tests.  These are run via `CTest', or by hand.  The "make
# test" target will also run ctest so the tests can be easily added to the build
#
#  This places the output of the test script in
#  ${CMAKE_BINARY_DIR}/Testing/Temporary/LastTest.log and the output
#  of the individual gundam tests in to
#  ${CMAKE_BINARY_DIR}/Testing/gundam-tests.<date>
#

cmessage( STATUS "Setting up tests..." )
enable_testing()

# Run the gundam-tests.sh script which runs scripted tests.  The
# COMMAND is a little complicated since the testing script needs to
# run in the source directory, but puts the output in a sub-directory
# of the binary directory.  The variable GUNDAM_TESTS_ARGUMENT may be
# set from the CMAKE (i.e. -DGUNDAM_TESTS_ARGUMENT="value") command
# line, or it will take a default value of "-f".  The useful values
# are "-f" for fast test, "-r" for regular tests, "-e" for extended
# tests, and "-s" for slow tests.
#
if(NOT DEFINED GUNDAM_TESTS_ARGUMENT OR GUNDAM_TESTS_ARGUMENT STREQUAL "")
  set(GUNDAM_TESTS_ARGUMENT "-f")
endif(NOT DEFINED GUNDAM_TESTS_ARGUMENT OR GUNDAM_TESTS_ARGUMENT STREQUAL "")

add_test(
    NAME gundam-tests
    COMMAND "sh" "-c" "(cd ${CMAKE_SOURCE_DIR}/tests && ./gundam-tests.sh ${GUNDAM_TESTS_ARGUMENT} -a ${CMAKE_BINARY_DIR}/Testing/gundam-tests.$(date +%Y-%m-%d-%H%M%S) )"
)

if( NOT SKIP_GOOGLE_TEST )
  ## Compiled unit tests using GoogleTest.  These are only run if GTest
  ## is found.
  find_package(GTest QUIET)
  if(GTEST_FOUND)
    if( WITH_CACHE_MANAGER )

      cmessage( STATUS "Compiling google tests..." )
      # Setup the hemi test suite for the host

      add_executable(gundamGTest_host.exe
          GTests/hemiArrayTest.cpp
          GTests/hemiExecutionPolicyTest.cpp
          GTests/cachedSumsTest.cpp
          GTests/hemiExternals.cpp)
      target_link_libraries(gundamGTest_host.exe GTest::gtest_main)
      target_link_libraries(gundamGTest_host.exe GundamCacheManager)
      target_compile_definitions(gundamGTest_host.exe PUBLIC HEMI_CUDA_DISABLE)
      gtest_discover_tests(gundamGTest_host.exe)

      if(CMAKE_CUDA_COMPILER)
        # Setup the hemi test suite for the device.
        add_executable(gundamGTest_device.exe
            GTests/hemiArrayTest.cu
            GTests/hemiExecutionPolicyTest.cu
            GTests/hemiLaunch.cu
            GTests/cachedSumsTest.cu
            GTests/hemiExternals.cpp)
        target_link_libraries(gundamGTest_device.exe GTest::gtest_main)
        target_link_libraries(gundamGTest_device.exe GundamCacheManager)
        gtest_discover_tests(gundamGTest_device.exe)
      endif(CMAKE_CUDA_COMPILER)

    else( WITH_CACHE_MANAGER )

      cmessage( WARNING "WITH_CACHE_MANAGER is set to false. Skipping Google test executables." )

    endif( WITH_CACHE_MANAGER )

  else(GTEST_FOUND)
    cmessage( FATAL_ERROR "Google test is not available. Use -D SKIP_GOOGLE_TEST=ON to skip it" )
  endif(GTEST_FOUND)

endif()
