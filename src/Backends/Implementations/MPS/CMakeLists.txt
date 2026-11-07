set_property(
    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${BACKENDS_SOURCE_DIR}/Common/Formula/include/DeviceMath/SharedFormulaOperations.h
    ${BACKENDS_SOURCE_DIR}/Common/Formula/include/FormulaTypes.h
    ${BACKENDS_SOURCE_DIR}/Common/Formula/include/DeviceMath/FormulaMath.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsDialPrograms.metal
    ${BACKENDS_SOURCE_DIR}/Common/Model/include/DeviceMath/DoubleFloat.h
    ${BACKENDS_SOURCE_DIR}/Common/Binning/include/Semantics/BackendBinningSemantics.h
    ${BACKENDS_SOURCE_DIR}/Common/Execution/include/Semantics/BackendVariableResponse.h
    ${BACKENDS_SOURCE_DIR}/Common/Model/include/Semantics/BackendResponseSmoothing.h
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedBinningMath.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MpsWeightTypes.h
    ${BACKENDS_SOURCE_DIR}/Common/Model/include/DevicePropagationTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsEventPropagationKernels.metal
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCommon.h
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCompact.h
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineUniform.h
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineMonotonic.h
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineGeneral.h
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSurface.h
    ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedGraph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsBackendKernels.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsBackendKernelSource.h.in
)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedBinningMath.h MPS_SHARED_BINNING_MATH)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Binning/include/Semantics/BackendBinningSemantics.h MPS_BINNING_SEMANTICS)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Execution/include/Semantics/BackendVariableResponse.h MPS_SHARED_VARIABLE_RESPONSE)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Model/include/Semantics/BackendResponseSmoothing.h MPS_RESPONSE_SMOOTHING)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Formula/include/FormulaTypes.h MPS_FORMULA_TYPES)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Formula/include/DeviceMath/FormulaMath.h MPS_FORMULA_MATH)
foreach(LLH_FILE
    ${BACKENDS_SOURCE_DIR}/Common/Model/include/LikelihoodTypes.h
    ${BACKENDS_SOURCE_DIR}/Common/Execution/include/Semantics/BackendLikelihood.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsLikelihood.metal
)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LLH_FILE})
endforeach()
file(READ ${BACKENDS_SOURCE_DIR}/Common/Model/include/LikelihoodTypes.h MPS_LIKELIHOOD_TYPES)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Execution/include/Semantics/BackendLikelihood.h MPS_LIKELIHOOD_SEMANTICS)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsLikelihood.metal MPS_LIKELIHOOD_KERNELS)
foreach(GRADIENT_FILE
    ${BACKENDS_SOURCE_DIR}/Common/Model/include/GradientTypes.h
    ${BACKENDS_SOURCE_DIR}/Common/Execution/include/Semantics/BackendDeviceGradient.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsGradient.metal)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GRADIENT_FILE})
endforeach()
file(READ ${BACKENDS_SOURCE_DIR}/Common/Model/include/GradientTypes.h MPS_GRADIENT_TYPES)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Execution/include/Semantics/BackendDeviceGradient.h MPS_GRADIENT_SEMANTICS)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsGradient.metal MPS_GRADIENT_KERNELS)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsDialPrograms.metal MPS_DIAL_PROGRAMS)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Formula/include/DeviceMath/SharedFormulaOperations.h MPS_FORMULA_OPERATIONS)
string(REPLACE "#include \"DeviceMath/SharedFormulaOperations.h\"" "${MPS_FORMULA_OPERATIONS}" MPS_DIAL_PROGRAMS "${MPS_DIAL_PROGRAMS}")
file(READ ${BACKENDS_SOURCE_DIR}/Common/Model/include/DevicePropagationTypes.h MPS_PROPAGATION_TYPES)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsEventPropagationKernels.metal MPS_EVENT_PROPAGATION_KERNELS)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCommon.h MPS_SHARED_SPLINE_COMMON)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCompact.h MPS_SHARED_SPLINE_COMPACT)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineUniform.h MPS_SHARED_SPLINE_UNIFORM)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineMonotonic.h MPS_SHARED_SPLINE_MONOTONIC)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineGeneral.h MPS_SHARED_SPLINE_GENERAL)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSurface.h MPS_SHARED_SURFACE)
file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedGraph.h MPS_SHARED_GRAPH)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/MpsWeightTypes.h MPS_WEIGHT_TYPES)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsBackendKernels.metal MPS_BACKEND_KERNEL_SOURCE)
file(READ ${BACKENDS_SOURCE_DIR}/Common/Model/include/DeviceMath/DoubleFloat.h MPS_DOUBLE_FLOAT)
set(MPS_OBSERVABLE_SPLINES "${MPS_SHARED_SPLINE_COMMON}\n${MPS_SHARED_SPLINE_COMPACT}\n${MPS_SHARED_SPLINE_UNIFORM}\n${MPS_SHARED_SPLINE_MONOTONIC}\n${MPS_SHARED_SPLINE_GENERAL}\n${MPS_SHARED_GRAPH}\n${MPS_SHARED_SURFACE}")
string(REPLACE "GundamDeviceMath" "GundamObservableMath" MPS_OBSERVABLE_SPLINES "${MPS_OBSERVABLE_SPLINES}")
string(REPLACE "SHARED_" "OBSERVABLE_SHARED_" MPS_OBSERVABLE_SPLINES "${MPS_OBSERVABLE_SPLINES}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsBackendKernelSource.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/MpsBackendKernelSource.h
    @ONLY
)

set(LIB_NAME GundamBackendsMPS)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsBackend.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsEventPropagation.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsBackendImpl.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsWeightModel.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MpsBackendExecution.mm
)

if(USE_STATIC_LINKS)
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PUBLIC GundamBackendsCommonExecution)

target_include_directories(${LIB_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(${LIB_NAME} PUBLIC "-framework Foundation" "-framework Metal")

install(TARGETS ${LIB_NAME} DESTINATION lib)
