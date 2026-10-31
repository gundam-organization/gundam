set(LIB_NAME GundamBackendsCommonFormula)

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FormulaCompiler.cpp
)

if(USE_STATIC_LINKS)
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS ${LIB_NAME} DESTINATION lib)
