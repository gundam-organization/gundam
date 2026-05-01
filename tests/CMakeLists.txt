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

set(DEV_WITHOUT_GUNDAM_TESTS OFF) # Should be off by default.
# Run the main executable test scripts.  These are relatively slow, so they
if( NOT DEV_WITHOUT_GUNDAM_TESTS )
  add_test(
    NAME gundam-tests
    COMMAND "sh" "-c" "(cd ${CMAKE_SOURCE_DIR}/tests && ./gundam-tests.sh ${GUNDAM_TESTS_ARGUMENT} -a ${CMAKE_BINARY_DIR}/Testing/gundam-tests.$(date +%Y-%m-%d-%H%M%S) )"
  )
else()
  cmessage( WARN "GUNDAM executable tests not being run.")
endif( NOT DEV_WITHOUT_GUNDAM_TESTS )

if( WITH_GOOGLE_TEST )
  ## Compiled unit tests using GoogleTest.
  include(GoogleTest)

  cmessage( STATUS "Compiling google tests" )
  add_executable(gundamGTest_unitTest.exe
    GTests/unitTest_JointProbability.cpp
    GTests/unitTest_CalculateCompactSpline.cpp
    GTests/unitTest_CalculateMonotonicSpline.cpp
    GTests/unitTest_CalculateGraph.cpp
    GTests/unitTest_CalculateGeneralSpline.cpp
    GTests/unitTest_CalculateUniformSpline.cpp
    GTests/unitTest_CalculateBilinearInterpolation.cpp
    GTests/unitTest_CalculateBicubicSpline.cpp
    GTests/unitTest_ApplyMonotonicCondition.cpp
  )
  target_link_libraries(gundamGTest_unitTest.exe GTest::gtest_main)
  target_link_libraries(gundamGTest_unitTest.exe GundamFitter)
  gtest_discover_tests(gundamGTest_unitTest.exe)

  if( WITH_CACHE_MANAGER )
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

    cmessage( WARNING "WITH_CACHE_MANAGER is set to false. Skipping Cache::Manager test executables." )

  endif( WITH_CACHE_MANAGER )

else(WITH_GOOGLE_TEST)
  cmessage( WARNING "GoogleTests are not used and some unit tests are skipped")
endif()
