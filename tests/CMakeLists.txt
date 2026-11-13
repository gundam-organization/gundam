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
  # Label the shell suite separately from the Python suites. The shell runner
  # defaults to fast; -r/-e/-s select increasingly inclusive suites.
  set(shellTestLabel fast)
  separate_arguments(shellTestArguments UNIX_COMMAND "${GUNDAM_TESTS_ARGUMENT}")
  foreach(shellTestArgument IN LISTS shellTestArguments)
    if(shellTestArgument STREQUAL "-f")
      set(shellTestLabel fast)
    elseif(shellTestArgument STREQUAL "-r")
      set(shellTestLabel regular)
    elseif(shellTestArgument STREQUAL "-e")
      set(shellTestLabel extended)
    elseif(shellTestArgument STREQUAL "-s")
      set(shellTestLabel slow)
    endif()
  endforeach()
  set_tests_properties(gundam-tests PROPERTIES LABELS "${shellTestLabel}")
else(WITH_INTERFACE_TEST)
  cmessage( WARNING "WITH_INTERFACE_TEST is OFF: gundam-tests.sh will be skipped")
endif(WITH_INTERFACE_TEST)

# Resolve a usable interpreter on the test node, not only the configure host.
if(WITH_PYTHON_INTERFACE)
  set(pythonTestEnvironment "${CMAKE_COMMAND}" -E env)
  if(APPLE)
    list(APPEND pythonTestEnvironment --unset=DYLD_LIBRARY_PATH)
  endif()
  set(pythonModuleDirectory "$<TARGET_FILE_DIR:${gundamTestPythonTarget}>")
  list(APPEND pythonTestEnvironment "PYTHONPATH=${pythonModuleDirectory}")
  set(pythonTestConfig "${CMAKE_CURRENT_BINARY_DIR}/python-test-config.cmake")
  set(pythonTestLauncher "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunPythonTest.cmake")
  file(GENERATE OUTPUT "${pythonTestConfig}" CONTENT
"set(testPythonCandidates [==[${gundamTestPython};${gundamTestPythonCandidates}]==])
set(testPythonVersions [==[${gundamTestPythonVersions}]==])
set(testPythonModuleDirectory [==[${pythonModuleDirectory}]==])
set(testPythonVenv [==[${CMAKE_CURRENT_BINARY_DIR}/python-venv]==])
set(testPythonUseVenv ${WITH_PYTHON_TEST_VENV})
set(testPythonPrepareScript [==[${CMAKE_CURRENT_SOURCE_DIR}/python/PrepareTestEnvironment.py]==])
set(testPythonRunner [==[${CMAKE_CURRENT_SOURCE_DIR}/python/run.py]==])
set(testPythonSelection [==[${CMAKE_CURRENT_BINARY_DIR}/python-test-selected.cmake]==])
")
  if(WITH_FAST_TEST OR WITH_REGULAR_TEST)
    add_test(NAME gundam-python-prepare
      COMMAND ${pythonTestEnvironment} "${CMAKE_COMMAND}"
        "-DTEST_CONFIG=${pythonTestConfig}" -DTEST_MODE=PREPARE -P "${pythonTestLauncher}")
    set_tests_properties(gundam-python-prepare PROPERTIES
      FIXTURES_SETUP gundamPythonEnvironment LABELS python)
  endif()
  foreach(pythonSuite fast regular)
    string(TOUPPER "${pythonSuite}" pythonSuiteOption)
    if(WITH_${pythonSuiteOption}_TEST)
      # Match the scripts selected by python/run.py and register each separately.
      file(GLOB pythonSuiteScripts CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/python/${pythonSuite}/*.py")
      foreach(pythonScript IN LISTS pythonSuiteScripts)
        get_filename_component(pythonScriptName "${pythonScript}" NAME)
        if(NOT pythonScriptName MATCHES "^([0-9]|test)")
          continue()
        endif()
        get_filename_component(pythonTestStem "${pythonScript}" NAME_WE)
        set(pythonTestName "gundam-python-${pythonSuite}-${pythonTestStem}")
        add_test(
          NAME "${pythonTestName}"
          COMMAND ${pythonTestEnvironment} "${CMAKE_COMMAND}"
            "-DTEST_CONFIG=${pythonTestConfig}" -DTEST_MODE=RUN
            "-DTEST_SUITE=${pythonSuite}" "-DTEST_NAME=${pythonScriptName}"
            -P "${pythonTestLauncher}"
        )
        set_tests_properties("${pythonTestName}" PROPERTIES LABELS "python;python-${pythonSuite}")
        set_tests_properties("${pythonTestName}" PROPERTIES
          FIXTURES_REQUIRED gundamPythonEnvironment)
      endforeach()
    endif()
  endforeach()
endif()

if( WITH_GOOGLE_TEST )
  ## Compiled unit tests using GoogleTest.
  include(GoogleTest)
  cmessage( STATUS "Compiling google tests" )

  function(gundam_discover_tests target_ category_)
    if(APPLE)
      # DYLD_LIBRARY_PATH overrides even the build RPATHs and can load an older
      # installed GUNDAM. Sanitize both discovery (which runs during the build)
      # and CTest execution, without changing the user's shell environment.
      # TEST_LAUNCHER was introduced in CMake 3.29. Older GoogleTest modules
      # use CROSSCOMPILING_EMULATOR as their executable wrapper on native builds.
      if(CMAKE_VERSION VERSION_LESS 3.29)
        set(launcher_property CROSSCOMPILING_EMULATOR)
      else()
        set(launcher_property TEST_LAUNCHER)
      endif()
      get_property(existing_launcher TARGET ${target_} PROPERTY ${launcher_property})
      set(launcher "${CMAKE_COMMAND}" -E env --unset=DYLD_LIBRARY_PATH)
      if(existing_launcher)
        list(APPEND launcher ${existing_launcher})
      endif()
      set_property(TARGET ${target_} PROPERTY ${launcher_property} "${launcher}")
    endif()
    gtest_discover_tests(${target_} ${ARGN})
    # Apply list-valued labels after discovery; older GoogleTest modules flatten
    # semicolon-separated PROPERTIES values when generating CTest commands.
    string(TOLOWER "${category_}" categoryLabel)
    set(testLabels unit "unit-${categoryLabel}")
    set(discoveryArguments ${ARGN})
    list(FIND discoveryArguments LABELS labelIndex)
    if(NOT labelIndex EQUAL -1)
      math(EXPR labelIndex "${labelIndex} + 1")
      list(GET discoveryArguments ${labelIndex} existingLabel)
      list(APPEND testLabels "${existingLabel}")
    endif()
    list(REMOVE_DUPLICATES testLabels)
    set(labelFile "${CMAKE_CURRENT_BINARY_DIR}/${target_}_labels.cmake")
    file(WRITE "${labelFile}"
      "if(DEFINED ${target_}_TESTS AND NOT \"\${${target_}_TESTS}\" STREQUAL \"\")\n"
      "  set_tests_properties(\${${target_}_TESTS} PROPERTIES LABELS \"${testLabels}\")\n"
      "endif()\n")
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${labelFile}")
  endfunction()

  # Group sources by the functionality being tested, independently of backend
  # or suite speed. Separate executables let discovery assign names and labels
  # without maintaining a second mapping of GoogleTest suite names.
  set(unitNumericsSources
    GTests/unitTest_CalculateCompactSpline.cpp
    GTests/unitTest_CalculateMonotonicSpline.cpp
    GTests/unitTest_CalculateGraph.cpp
    GTests/unitTest_CalculateGeneralSpline.cpp
    GTests/unitTest_CalculateUniformSpline.cpp
    GTests/unitTest_CalculateBilinearInterpolation.cpp
    GTests/unitTest_CalculateBicubicSpline.cpp
    GTests/unitTest_ApplyMonotonicCondition.cpp)
  set(unitDialsSources
    GTests/unitTest_DialUtils.cpp
    GTests/unitTest_DialDefinitions.cpp
    GTests/unitTest_FormulaEvaluator.cpp)
  set(unitDataLoadingSources GTests/unitTest_DataDispenser.cpp)
  set(unitLikelihoodSources GTests/unitTest_JointProbability.cpp)
  set(unitMinimizerSources GTests/unitTest_RootMinimizerSimplex.cpp)
  set(unitBackendsSources)

  # Always compile the tests; WITH_FAST_TEST controls CTest registration only.
  foreach(unitCategory Numerics Dials DataLoading Likelihood Minimizer Backends)
    if(unit${unitCategory}Sources)
      set(unitTarget "gundamGTest_${unitCategory}.exe")
      add_executable(${unitTarget} ${unit${unitCategory}Sources})
      target_link_libraries(${unitTarget} PRIVATE GTest::gtest_main GundamFitter)
      if(WITH_FAST_TEST)
        gundam_discover_tests(${unitTarget} ${unitCategory}
          TEST_PREFIX "${unitCategory}/"
          PROPERTIES LABELS unit)
      endif()
    endif()
  endforeach()

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
      gundam_discover_tests(gundamGTest_regularTest.exe Backends
          TEST_PREFIX "Backends/"
          PROPERTIES LABELS regular TIMEOUT 120)
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
      gundam_discover_tests(gundamGTest_host.exe Backends
          TEST_PREFIX "Backends/" PROPERTIES LABELS host)
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
        gundam_discover_tests(gundamGTest_device.exe Backends DISCOVERY_MODE PRE_TEST
            TEST_PREFIX "Backends/" PROPERTIES LABELS device)
      endif(WITH_REGULAR_TEST)
    endif(CMAKE_CUDA_COMPILER)

  else( WITH_CACHE_MANAGER )

    cmessage( WARNING "WITH_CACHE_MANAGER is OFF. Skip Cache::Manager tests." )

  endif( WITH_CACHE_MANAGER )

else( WITH_GOOGLE_TEST )
  cmessage( WARNING "GoogleTest disabled so some unit tests will be skipped")
endif( WITH_GOOGLE_TEST )
