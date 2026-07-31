set(LIB_NAME GundamBackends)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/EngineBindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/EngineLayout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendRuntimeManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendTypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendImpl.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendDeviceModel.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendExecution.mm
)

if(APPLE)
  enable_language(OBJCXX)
  set_property(
      DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCommon.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCompact.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineUniform.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineMonotonic.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineGeneral.h
      ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedGraph.h
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernels.metal
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernelSource.h.in
  )
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCommon.h MPS_SHARED_SPLINE_COMMON)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineCompact.h MPS_SHARED_SPLINE_COMPACT)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineUniform.h MPS_SHARED_SPLINE_UNIFORM)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineMonotonic.h MPS_SHARED_SPLINE_MONOTONIC)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedSplineGeneral.h MPS_SHARED_SPLINE_GENERAL)
  file(READ ${CMAKE_SOURCE_DIR}/src/Utils/include/DeviceMath/SharedGraph.h MPS_SHARED_GRAPH)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernels.metal MPS_BACKEND_KERNEL_SOURCE)
  configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackendKernelSource.h.in
      ${CMAKE_CURRENT_BINARY_DIR}/generated/MpsBackendKernelSource.h
      @ONLY
  )
  list(APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackend.mm)
else()
  list(APPEND SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackend.cpp)
endif()

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendsManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendEngineBindings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendEngineDescriptors.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendEngineLayout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/EngineView.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendRuntimeManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/IPropagationBackend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/ParameterSnapshot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendSemanticsQualifiers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendDialSemanticsCore.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendDialSemantics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/Semantics/BackendHostPropagation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/include/CpuBackend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsBackend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsBackendInternal.h
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

install(TARGETS ${LIB_NAME} DESTINATION lib)
