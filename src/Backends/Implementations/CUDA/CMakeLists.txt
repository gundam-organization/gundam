set(LIB_NAME GundamBackendsCUDA)

# Independent workers load this C ABI even when GUNDAM is linked statically.
add_library(GundamCudaWorkerBuffers SHARED src/CudaWorkerBuffer.cpp)
target_include_directories(GundamCudaWorkerBuffers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(GundamCudaWorkerBuffers PRIVATE ${CMAKE_SOURCE_DIR}/src/Utils/include)
target_link_libraries(GundamCudaWorkerBuffers PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
set_target_properties(GundamCudaWorkerBuffers PROPERTIES CXX_VISIBILITY_PRESET default)
set_property(TARGET GundamCudaWorkerBuffers APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
install(TARGETS GundamCudaWorkerBuffers DESTINATION lib)

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

target_link_libraries(${LIB_NAME} PRIVATE CUDA::cudart GundamCudaWorkerBuffers)
target_compile_definitions(${LIB_NAME} PUBLIC GUNDAM_USING_CUDA)
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
set_property(TARGET ${LIB_NAME} APPEND PROPERTY
    BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")

install(TARGETS ${LIB_NAME} DESTINATION lib)
