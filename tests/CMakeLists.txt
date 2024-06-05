# Define tests.  These are run via `CTest', or by hand.  The "make
# test" target will also run ctest so the tests can be easily added to the build
#
#  This places the output of the test script in
#  ${CMAKE_BINARY_DIR}/Testing/Temporary/LastTest.log and the output
#  of the individual gundam tests in to
#  ${CMAKE_BINARY_DIR}/Testing/gundam-tests.<date>
#



if( WITH_GTEST )

  cmessage( STATUS "Setting up tests..." )

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

  ## Compiled unit tests using GoogleTest.  These are only run if GTest
  ## is found.
  find_package(GTest)
  if(GTEST_FOUND)
    enable_testing()

    # Setup the hemi test suite for the host
    add_executable(hemiTest_host.exe
      GTests/hemiArrayTest.cpp
      GTests/hemiExecutionPolicyTest.cpp
      GTests/hemiExternals.cpp)
    target_link_libraries(hemiTest_host.exe GTest::gtest_main)
    target_link_libraries(hemiTest_host.exe GundamCache)
    target_compile_definitions(hemiTest_host.exe PUBLIC HEMI_CUDA_DISABLE)
    gtest_discover_tests(hemiTest_host.exe)

    if(CMAKE_CUDA_COMPILER)
      # Setup the hemi test suite for the device.
      add_executable(hemiTest_device.exe
        GTests/hemiArrayTest.cu
        GTests/hemiExecutionPolicyTest.cu
        GTests/hemiLaunch.cu
        GTests/hemiExternals.cpp)
      target_link_libraries(hemiTest_device.exe GTest::gtest_main)
      target_link_libraries(hemiTest_device.exe GundamCache)
      gtest_discover_tests(hemiTest_device.exe)
    endif(CMAKE_CUDA_COMPILER)

  endif(GTEST_FOUND)
endif( WITH_GTEST )