
set( LIB_NAME GundamParametersManager )

set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ParameterSet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Parameter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ParametersManager.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Parameter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ParameterSet.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ParametersManager.h
)

#ROOT_GENERATE_DICTIONARY(
#        GundamFitParametersDict
#        ${HEADERS}
##        ${MY_PUBLIC_HEADERS}
#        LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/FitParametersLinkDef.h
##        MODULE GundamUtils
#)
#
#list(APPEND SRCFILES GundamParametersDict.cxx)


if( USE_STATIC_LINKS )
  add_library(${LIB_NAME} STATIC ${SRCFILES})
else()
  add_library(${LIB_NAME} SHARED ${SRCFILES})
endif()

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Dependencies
target_link_libraries( ${LIB_NAME} PUBLIC
    GundamUtils
    ${ROOT_LIBRARIES}
)

#set_target_properties(${LIB_NAME} PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS ${LIB_NAME} DESTINATION lib)
#
#install(FILES ${HEADERS} DESTINATION include)
#install(
#        FILES
#        ${CMAKE_CURRENT_BINARY_DIR}/libGundamParametersDict_rdict.pcm
#        ${CMAKE_CURRENT_BINARY_DIR}/libGundamParametersDict.rootmap
#        DESTINATION
#        ${CMAKE_INSTALL_PREFIX}/lib
#        COMPONENT
#        libraries
#)

