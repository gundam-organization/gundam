set(LIB_NAME GundamBackendsCPU)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CpuBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CpuBackendProfiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CpuWeightProbe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CpuBackendBinning.cpp
)

if(USE_STATIC_LINKS)
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PUBLIC GundamBackendsCommonExecution)

target_compile_definitions(${LIB_NAME} PRIVATE GUNDAM_CPU_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

install(TARGETS ${LIB_NAME} DESTINATION lib)
