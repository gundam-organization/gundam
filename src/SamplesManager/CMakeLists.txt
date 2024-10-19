# Samples
set( LIB_NAME GundamSamplesManager )

set( SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/VariableCollection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/EventUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Histogram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Sample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SampleSet.cpp
)

set( HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/EventUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Histogram.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Event.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Sample.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/SampleSet.h
)

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    ${LIB_NAME}
    PUBLIC GundamParametersManager
)

#set_target_properties( ${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)

