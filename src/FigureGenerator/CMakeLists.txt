# Samples
set( LIB_NAME GundamFigureGenerator )

set( SRCFILES
    src/FigureGenerator.cpp
    src/SampleHistograms1D.cpp
    src/FigureUtils.cpp
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
    PUBLIC GundamSamplesManager
)

#set_target_properties( ${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)

