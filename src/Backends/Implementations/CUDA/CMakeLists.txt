set(LIB_NAME GundamBackendsCUDA)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaWeightModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDeviceContext.cu
)

if(USE_STATIC_LINKS)
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PUBLIC GundamBackendsCommonExecution)

target_link_libraries(${LIB_NAME} PRIVATE CUDA::cudart)
target_compile_definitions(${LIB_NAME} PUBLIC GUNDAM_USING_CUDA)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET ${LIB_NAME} APPEND PROPERTY
    BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")

install(TARGETS ${LIB_NAME} DESTINATION lib)
