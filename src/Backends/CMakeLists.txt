set(LIB_NAME GundamBackends)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BinningGeometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/DevicePropagationModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/EventStateCache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/DeviceWeightModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/EngineBindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/EngineLayout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendTypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackendProfiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuWeightProbe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackendBinning.cpp
)

if(APPLE)
  enable_language(OBJCXX)
  list(APPEND SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsEventPropagation.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendImpl.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsWeightModel.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendExecution.mm
  )
  set_property(
      DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DeviceMath/DoubleFloat.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendVariableResponse.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendResponseSmoothing.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedBinningMath.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsWeightTypes.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DevicePropagationTypes.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsEventPropagationKernels.metal
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCommon.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCompact.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineUniform.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineMonotonic.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineGeneral.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSurface.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedGraph.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernels.metal
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernelSource.h.in
  )
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedBinningMath.h MPS_SHARED_BINNING_MATH)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendVariableResponse.h MPS_SHARED_VARIABLE_RESPONSE)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendResponseSmoothing.h MPS_RESPONSE_SMOOTHING)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DevicePropagationTypes.h MPS_PROPAGATION_TYPES)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsEventPropagationKernels.metal MPS_EVENT_PROPAGATION_KERNELS)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCommon.h MPS_SHARED_SPLINE_COMMON)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCompact.h MPS_SHARED_SPLINE_COMPACT)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineUniform.h MPS_SHARED_SPLINE_UNIFORM)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineMonotonic.h MPS_SHARED_SPLINE_MONOTONIC)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineGeneral.h MPS_SHARED_SPLINE_GENERAL)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSurface.h MPS_SHARED_SURFACE)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedGraph.h MPS_SHARED_GRAPH)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsWeightTypes.h MPS_WEIGHT_TYPES)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernels.metal MPS_BACKEND_KERNEL_SOURCE)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DeviceMath/DoubleFloat.h MPS_DOUBLE_FLOAT)
  set(MPS_OBSERVABLE_SPLINES "${MPS_SHARED_SPLINE_COMMON}\n${MPS_SHARED_SPLINE_COMPACT}\n${MPS_SHARED_SPLINE_UNIFORM}\n${MPS_SHARED_SPLINE_MONOTONIC}\n${MPS_SHARED_SPLINE_GENERAL}\n${MPS_SHARED_GRAPH}\n${MPS_SHARED_SURFACE}")
  string(REPLACE "GundamDeviceMath" "GundamObservableMath" MPS_OBSERVABLE_SPLINES "${MPS_OBSERVABLE_SPLINES}")
  string(REPLACE "SHARED_" "OBSERVABLE_SHARED_" MPS_OBSERVABLE_SPLINES "${MPS_OBSERVABLE_SPLINES}")
  configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernelSource.h.in
      ${CMAKE_CURRENT_BINARY_DIR}/generated/MpsBackendKernelSource.h
      @ONLY
  )
  list(APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackend.mm)
endif()

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendResponseSmoothing.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/ResponseSmoothingPacking.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendVariableResponse.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DeviceMath/DoubleFloat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BinningGeometry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/EventStateCache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/PropagationResult.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DeviceWeightModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DevicePropagationModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/DevicePropagationTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/EngineBindings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/EngineDescriptors.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/EngineLayout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/EngineView.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Backend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/PropagationInputs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/ParameterSnapshot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendSemanticsQualifiers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendDialSemanticsCore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendDialSemantics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendHostPropagation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/include/CpuWeightKernels.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/include/CpuBackend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/include/CpuBackendProfiler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/include/CpuWeightProbe.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsBackend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsSplinePayloadLayout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsWeightTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsBackendInternal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsEventPropagation.h
)

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME}
    BEFORE
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(
    ${LIB_NAME}
    PUBLIC
    GundamDialDictionary
    GundamStatisticalInference
)

if(APPLE)
  target_link_libraries(${LIB_NAME} PUBLIC "-framework Foundation" "-framework Metal")
endif()

target_compile_definitions(${LIB_NAME} PRIVATE GUNDAM_CPU_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

install(TARGETS ${LIB_NAME} DESTINATION lib)
