set( LIB_NAME GundamFitter )

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/FitterEngine.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/MinimizerBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/AdaptiveMcmc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/src/RootMinimizer.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Likelihood/src/LikelihoodInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Likelihood/src/JointProbability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Likelihood/src/ParameterScanner.cpp
  )

if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(
    ${LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Likelihood/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Minimizer/include
)

target_link_libraries(
    ${LIB_NAME}
    PUBLIC GundamPropagator
)

#set_target_properties( ${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}" )

install(TARGETS ${LIB_NAME} DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)
