set(LIB_NAME GundamBackendsCommonManager)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BackendFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BackendManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BinningGeometryBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EngineBindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EngineLayout.cpp
)

if(USE_STATIC_LINKS)
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PUBLIC
    GundamBackendsCommonModel
    GundamStatisticalInference
    GundamBackendsCPU
)
if(GUNDAM_CUDA_ENABLED)
  target_link_libraries(${LIB_NAME} PUBLIC GundamBackendsCUDA)
endif()
if(APPLE)
  target_link_libraries(${LIB_NAME} PUBLIC GundamBackendsMPS)
endif()

install(TARGETS ${LIB_NAME} DESTINATION lib)
