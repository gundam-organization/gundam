set( LIB_NAME GundamCrossSection )

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CrossSectionCalculator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ParameterSetRenorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BinNormaliser.cpp
  )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    ${LIB_NAME}
    PUBLIC
    GundamFitter # need the fitter library to handle old configs (when the StatisticalInference lib didn't exist)
)

install(TARGETS ${LIB_NAME} DESTINATION lib)
