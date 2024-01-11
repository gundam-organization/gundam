
set( LIB_NAME GundamFitter )

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FitterEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LikelihoodInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MinimizerBase.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MinimizerInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MCMCInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JointProbability.cpp
  )

if( USE_STATIC_LINKS )
  add_library( ${LIB_NAME} STATIC ${SRCFILES} )
else()
  add_library( ${LIB_NAME} SHARED ${SRCFILES} )
endif()

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    ${LIB_NAME}
    PUBLIC GundamPropagator
)

#set_target_properties( ${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}" )

install( TARGETS ${LIB_NAME} DESTINATION lib )
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)
