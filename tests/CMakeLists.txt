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
  if(WITH_SLOW_TEST)
    set(GUNDAM_TESTS_ARGUMENT "-s")
  elseif(WITH_EXTENDED_TEST)
    set(GUNDAM_TESTS_ARGUMENT "-e")
  elseif(WITH_REGULAR_TEST)
    set(GUNDAM_TESTS_ARGUMENT "-r")
  else()
    set(GUNDAM_TESTS_ARGUMENT "-f")
  endif()
endif(NOT DEFINED GUNDAM_TESTS_ARGUMENT OR GUNDAM_TESTS_ARGUMENT STREQUAL "")

# Run the main executable test scripts.  These are relatively slow, so they
if(WITH_INTERFACE_TEST)
  cmessage( STATUS "gundam-tests.sh run using: ${GUNDAM_TESTS_ARGUMENT}")
  add_test(
    NAME gundam-tests
    COMMAND "sh" "-c" "(cd ${CMAKE_SOURCE_DIR}/tests && ./gundam-tests.sh ${GUNDAM_TESTS_ARGUMENT} -a ${CMAKE_BINARY_DIR}/Testing/gundam-tests.$(date +%Y-%m-%d-%H%M%S) )"
  )
else(WITH_INTERFACE_TEST)
  cmessage( WARNING "WITH_INTERFACE_TEST is OFF: gundam-tests.sh will be skipped")
endif(WITH_INTERFACE_TEST)

if( WITH_GOOGLE_TEST )
  ## Compiled unit tests using GoogleTest.
  include(GoogleTest)
  cmessage( STATUS "Compiling google tests" )

  ## Create a list of all of the fast tests.  These are always
  ## compiled, and are run when WITH_FAST_TEST is on.  The fast tests
  ## are run during CI, so all of them should finish quickly, and we
  ## should limit the total number.
  list(APPEND FAST_TESTS "")
  list(APPEND FAST_TESTS GTests/unitTest_JointProbability.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_CalculateCompactSpline.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_CalculateMonotonicSpline.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_CalculateGraph.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_CalculateGeneralSpline.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_CalculateUniformSpline.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_CalculateBilinearInterpolation.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_CalculateBicubicSpline.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_ApplyMonotonicCondition.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_DialUtils.cpp)

  # Compile the fast test executable, but only add them to ctest if
  # WITH_FAST_TEST is on. This should NOT be "optimized" to skip compilation
  # when WITH_FAST_TEST is off.
  if(FAST_TESTS)
    add_executable(gundamGTest_fastTest.exe ${FAST_TESTS})
    target_link_libraries(gundamGTest_fastTest.exe GTest::gtest_main)
    target_link_libraries(gundamGTest_fastTest.exe GundamFitter)
    if(WITH_FAST_TEST)
      gtest_discover_tests(gundamGTest_fastTest.exe PROPERTIES LABELS unit)
    endif(WITH_FAST_TEST)
  endif(FAST_TESTS)

  ## Create a list of all of the regular tests.  These are always
  ## compiled and are run when WITH_REGULAR_TEST is on.
  list(APPEND REGULAR_TESTS "")

  # Compile the regular test executable, but only add them to ctest if
  # WITH_REGULAR_TEST is on. This should NOT be "optimized" to skip
  # compilation when WITH_REGULAR_TEST is off.
  if(REGULAR_TESTS)
    add_executable(gundamGTest_regularTest.exe ${REGULAR_TESTS})
    target_link_libraries(gundamGTest_regularTest.exe GTest::gtest_main)
    target_link_libraries(gundamGTest_regularTest.exe GundamFitter)
    if(WITH_REGULAR_TEST)
      gtest_discover_tests(gundamGTest_regularTest.exe PROPERTIES LABELS unit)
    endif(WITH_REGULAR_TEST)
  endif(REGULAR_TESTS)

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
    if (WITH_REGULAR_TEST)
      gtest_discover_tests(gundamGTest_host.exe PROPERTIES LABELS host)
    endif(WITH_REGULAR_TEST)

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
      if(WITH_REGULAR_TEST)
        gtest_discover_tests(gundamGTest_device.exe PROPERTIES LABELS device)
      endif(WITH_REGULAR_TEST)
    endif(CMAKE_CUDA_COMPILER)

  else( WITH_CACHE_MANAGER )

    cmessage( WARNING "WITH_CACHE_MANAGER is OFF. Skip Cache::Manager tests." )

  endif( WITH_CACHE_MANAGER )

else( WITH_GOOGLE_TEST )
  cmessage( WARNING "GoogleTest disabled so some unit tests will be skipped")
endif( WITH_GOOGLE_TEST )
