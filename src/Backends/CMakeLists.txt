set(LIB_NAME GundamBackendsCommonManager)

add_subdirectory(Common/Formula)
add_subdirectory(Common/Binning)
add_subdirectory(Common/Model)
add_subdirectory(Common/Execution)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackendProfiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuWeightProbe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackendBinning.cpp
)

if(GUNDAM_CUDA_ENABLED)
  list(APPEND SRCFILES
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CUDA/src/CudaBackend.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CUDA/src/CudaWeightModel.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CUDA/src/CudaDeviceContext.cu
  )
endif()

if(APPLE)
  list(APPEND SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsEventPropagation.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendImpl.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsWeightModel.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendExecution.mm
  )
  set_property(
      DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Formula/include/DeviceMath/SharedFormulaOperations.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Formula/include/FormulaTypes.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Formula/include/DeviceMath/FormulaMath.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsDialPrograms.metal
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Model/include/DeviceMath/DoubleFloat.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Binning/include/Semantics/BackendBinningSemantics.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Execution/include/Semantics/BackendVariableResponse.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Model/include/Semantics/BackendResponseSmoothing.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedBinningMath.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsWeightTypes.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Common/Model/include/DevicePropagationTypes.h
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
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Binning/include/Semantics/BackendBinningSemantics.h MPS_BINNING_SEMANTICS)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Execution/include/Semantics/BackendVariableResponse.h MPS_SHARED_VARIABLE_RESPONSE)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Model/include/Semantics/BackendResponseSmoothing.h MPS_RESPONSE_SMOOTHING)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Formula/include/FormulaTypes.h MPS_FORMULA_TYPES)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Formula/include/DeviceMath/FormulaMath.h MPS_FORMULA_MATH)
  foreach(LLH_FILE Common/Model/include/LikelihoodTypes.h Common/Execution/include/Semantics/BackendLikelihood.h Implementations/MPS/src/MpsLikelihood.metal)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${LLH_FILE})
  endforeach()
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Model/include/LikelihoodTypes.h MPS_LIKELIHOOD_TYPES)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Execution/include/Semantics/BackendLikelihood.h MPS_LIKELIHOOD_SEMANTICS)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsLikelihood.metal MPS_LIKELIHOOD_KERNELS)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsDialPrograms.metal MPS_DIAL_PROGRAMS)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Formula/include/DeviceMath/SharedFormulaOperations.h MPS_FORMULA_OPERATIONS)
  string(REPLACE "#include \"DeviceMath/SharedFormulaOperations.h\"" "${MPS_FORMULA_OPERATIONS}" MPS_DIAL_PROGRAMS "${MPS_DIAL_PROGRAMS}")
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Model/include/DevicePropagationTypes.h MPS_PROPAGATION_TYPES)
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
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Common/Model/include/DeviceMath/DoubleFloat.h MPS_DOUBLE_FLOAT)
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

# Manager owns the public entry point and the concrete implementations selected by its factory.
set(BACKENDS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BACKENDS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(Common/Manager)
add_library(GundamBackends ALIAS ${LIB_NAME})
