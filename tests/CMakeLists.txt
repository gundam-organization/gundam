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

# Explicit CTest registration: configuration/build must not execute a GPU test.
if(GUNDAM_CUDA_ENABLED)
  add_subdirectory(cuda)
endif()

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

  function(gundam_discover_tests target_)
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
  endfunction()

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
  list(APPEND FAST_TESTS GTests/unitTest_FormulaEvaluator.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_DialDefinitions.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_DialResponseSupervisor.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_DataDispenser.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_RootMinimizerSimplex.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_DataDispenser.cpp)
  list(APPEND FAST_TESTS GTests/unitTest_DialResponseSupervisor.cpp)

  if(WITH_BACKENDS)
    list(APPEND FAST_TESTS GTests/unitTest_BackendManager.cpp)
    if(GUNDAM_CUDA_ENABLED)
      list(APPEND FAST_TESTS GTests/unitTest_CudaBackend.cpp)
      list(APPEND FAST_TESTS GTests/unitTest_CudaBackendPrograms.cpp)
    endif()
    list(APPEND FAST_TESTS GTests/unitTest_EngineLayout.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_CpuComposite.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_BackendFormula.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_MpsFormula.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_CpuWeightExecution.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_ExternalWeightBackend.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_BackendBinning.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_BackendShared.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_BackendLikelihood.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_CpuBackendGradient.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_BackendResponseSmoothing.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_BackendSurface.cpp)
    list(APPEND FAST_TESTS GTests/unitTest_BackendTable.cpp)
  endif()

  # Compile the fast test executable, but only add them to ctest if
  # WITH_FAST_TEST is on. This should NOT be "optimized" to skip compilation
  # when WITH_FAST_TEST is off.
  if(FAST_TESTS)
    add_executable(gundamGTest_fastTest.exe ${FAST_TESTS})
    target_link_libraries(gundamGTest_fastTest.exe GTest::gtest_main)
    target_link_libraries(gundamGTest_fastTest.exe GundamFitter)
    if(WITH_FAST_TEST)
      gundam_discover_tests(gundamGTest_fastTest.exe PROPERTIES LABELS unit)
    endif(WITH_FAST_TEST)
  endif(FAST_TESTS)

  if(WITH_BACKENDS AND GUNDAM_CUDA_ENABLED)
    add_executable(gundamGTest_cudaBackend.exe
        GTests/unitTest_CpuBackendGradient.cpp
        GTests/unitTest_CudaBackendPrograms.cpp
        GTests/unitTest_BackendLikelihood.cpp
        GTests/unitTest_CudaBackendDevice.cpp
        GTests/unitTest_CudaBackendPropagation.cpp
        GTests/unitTest_CudaBackendSurface.cpp
        GTests/unitTest_CudaBackendExternal.cpp
        GTests/unitTest_CudaBackendObservable.cpp
        GTests/unitTest_CudaBackendSmoothed.cpp)
    target_compile_definitions(gundamGTest_cudaBackend.exe PRIVATE GUNDAM_TEST_CUDA_DEVICE)
    target_link_libraries(gundamGTest_cudaBackend.exe PRIVATE
        GTest::gtest_main GundamBackends CUDA::cudart)
    set_property(TARGET gundamGTest_cudaBackend.exe APPEND PROPERTY
        BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
    # Register without discovery: building must work without an allocated GPU.
    add_test(NAME cuda-backend-programs COMMAND gundamGTest_cudaBackend.exe --gtest_filter=CudaBackendPrograms.*)
    add_test(NAME cuda-backend-gradient COMMAND gundamGTest_cudaBackend.exe --gtest_filter=DeviceGradient.Cuda*:DeviceGradient.Portable*:CpuBackendGradient.*)
    add_test(NAME cuda-backend-likelihood COMMAND gundamGTest_cudaBackend.exe --gtest_filter=BackendLikelihood.*)
    add_test(NAME cuda-backend-skeleton COMMAND gundamGTest_cudaBackend.exe --gtest_filter=CudaBackendDevice.*)
    add_test(NAME cuda-backend-propagation COMMAND gundamGTest_cudaBackend.exe --gtest_filter=CudaBackendPropagation.*)
    add_test(NAME cuda-backend-surfaces COMMAND gundamGTest_cudaBackend.exe --gtest_filter=CudaBackendSurface.*)
    add_test(NAME cuda-backend-external COMMAND gundamGTest_cudaBackend.exe --gtest_filter=CudaBackendExternal.*)
    add_test(NAME cuda-backend-observables COMMAND gundamGTest_cudaBackend.exe --gtest_filter=CudaBackendObservable.*)
    add_test(NAME cuda-backend-smoothed COMMAND gundamGTest_cudaBackend.exe --gtest_filter=CudaBackendSmoothed.*)
    set_tests_properties(cuda-backend-programs cuda-backend-likelihood cuda-backend-skeleton cuda-backend-propagation cuda-backend-surfaces cuda-backend-external cuda-backend-observables cuda-backend-smoothed PROPERTIES LABELS "cuda;device" TIMEOUT 120)
  endif()

  ## Create a list of all of the regular tests.  These are always
  ## compiled and are run when WITH_REGULAR_TEST is on.
  list(APPEND REGULAR_TESTS "")
  list(APPEND REGULAR_TESTS GTests/unitTest_BackendFitCovariance.cpp)

  # Compile the regular test executable, but only add them to ctest if
  # WITH_REGULAR_TEST is on. This should NOT be "optimized" to skip
  # compilation when WITH_REGULAR_TEST is off.
  if(REGULAR_TESTS)
    add_executable(gundamGTest_regularTest.exe ${REGULAR_TESTS})
    target_link_libraries(gundamGTest_regularTest.exe GTest::gtest_main)
    target_link_libraries(gundamGTest_regularTest.exe GundamFitter)
    if(WITH_BACKENDS AND GUNDAM_CUDA_ENABLED)
      target_compile_definitions(gundamGTest_regularTest.exe PRIVATE GUNDAM_TEST_HAS_CUDA)
      target_link_libraries(gundamGTest_regularTest.exe CUDA::cudart)
      set_property(TARGET gundamGTest_regularTest.exe APPEND PROPERTY
          BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
    endif()
    if(WITH_REGULAR_TEST)
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/Testing/BackendFitCovariance")
      gundam_discover_tests(gundamGTest_regularTest.exe
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/Testing/BackendFitCovariance"
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
      gundam_discover_tests(gundamGTest_host.exe PROPERTIES LABELS host)
    endif(WITH_REGULAR_TEST)

    if(GUNDAM_CUDA_ENABLED)
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
        gundam_discover_tests(gundamGTest_device.exe DISCOVERY_MODE PRE_TEST PROPERTIES LABELS device)
      endif(WITH_REGULAR_TEST)
    endif(GUNDAM_CUDA_ENABLED)

  else( WITH_CACHE_MANAGER )

    cmessage( WARNING "WITH_CACHE_MANAGER is OFF. Skip Cache::Manager tests." )

  endif( WITH_CACHE_MANAGER )

else( WITH_GOOGLE_TEST )
  cmessage( WARNING "GoogleTest disabled so some unit tests will be skipped")
endif( WITH_GOOGLE_TEST )
