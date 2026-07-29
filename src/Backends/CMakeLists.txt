set(LIB_NAME GundamBackends)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendModelBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/src/BackendTypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/src/CpuBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/src/MpsBackend.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendModel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendModelBuilder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/BackendTypes.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/IPropagationBackend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Common/include/ParameterSnapshot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/CPU/include/CpuBackend.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Implementations/MPS/include/MpsBackend.h
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
)

target_link_libraries(
    ${LIB_NAME}
    PUBLIC
    GundamDialDictionary
)

install(TARGETS ${LIB_NAME} DESTINATION lib)
