set(LIB_NAME GundamBackendsCommonManager)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BackendFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BackendManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BinningGeometryBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EngineBindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EngineLayout.cpp
    ${SRCFILES}
)

if(USE_STATIC_LINKS)
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PUBLIC GundamBackendsCommonExecution GundamStatisticalInference)

install(TARGETS ${LIB_NAME} DESTINATION lib)

target_include_directories(${LIB_NAME} BEFORE
    PUBLIC ${BACKENDS_SOURCE_DIR}/Implementations/CPU/include
    PRIVATE ${BACKENDS_BINARY_DIR}/generated
)

if(GUNDAM_CUDA_ENABLED)
  target_link_libraries(${LIB_NAME} PRIVATE CUDA::cudart)
  target_compile_definitions(${LIB_NAME} PUBLIC GUNDAM_USING_CUDA)
  target_include_directories(${LIB_NAME} BEFORE PUBLIC ${BACKENDS_SOURCE_DIR}/Implementations/CUDA/include)
  set_target_properties(${LIB_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  set_property(TARGET ${LIB_NAME} APPEND PROPERTY
      BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
endif()

if(APPLE)
  target_include_directories(${LIB_NAME} BEFORE PUBLIC ${BACKENDS_SOURCE_DIR}/Implementations/MPS/include)
  target_link_libraries(${LIB_NAME} PUBLIC "-framework Foundation" "-framework Metal")
endif()

target_compile_definitions(${LIB_NAME} PRIVATE GUNDAM_CPU_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

