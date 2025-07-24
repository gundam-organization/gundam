set( LIB_NAME GundamFitter )

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/FitterEngine.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/MinimizerBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/SimpleMcmc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/RootMinimizer.cpp
  )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/include
)

target_link_libraries(
    ${LIB_NAME}
    PUBLIC
    GundamStatisticalInference
    SimpleMcmc_Dict
)

set(CMAKE_CXX_FLAGS_DICT "-Wno-nonportable-include-path")
ROOT_GENERATE_DICTIONARY(
  SimpleMcmc_Dict
  ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/include/SimpleMcmc.h
  MODULE ${LIB_NAME}
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/SimpleMcmc_LinkDef.h
  OPTIONS "${CMAKE_CXX_FLAGS_DICT}"
)

install(TARGETS ${LIB_NAME} DESTINATION lib)

# Install the ROOT pcm and rootmap files
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libGundamFitter_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/libGundamFitter.rootmap
  DESTINATION
  ${CMAKE_INSTALL_PREFIX}/lib
  COMPONENT
  libraries
)
