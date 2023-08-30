set(SRCFILES
        src/FitParameterSet.cpp
        src/FitParameter.cpp
        )

set(HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/FitParameter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/FitParameterSet.h
        )


#ROOT_GENERATE_DICTIONARY(
#        GundamFitParametersDict
#        ${HEADERS}
##        ${MY_PUBLIC_HEADERS}
#        LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/FitParametersLinkDef.h
##        MODULE GundamUtils
#)
#
#list(APPEND SRCFILES GundamFitParametersDict.cxx)


if( USE_STATIC_LINKS )
  add_library(GundamFitParameters STATIC ${SRCFILES})
else()
  add_library(GundamFitParameters SHARED ${SRCFILES})
endif()

target_include_directories(GundamFitParameters PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Dependencies
target_link_libraries( GundamFitParameters PUBLIC GundamUtils ${ROOT_LIBRARIES} )

#set_target_properties(GundamFitParameters PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS GundamFitParameters DESTINATION lib)
#
#install(FILES ${HEADERS} DESTINATION include)
#install(
#        FILES
#        ${CMAKE_CURRENT_BINARY_DIR}/libGundamFitParametersDict_rdict.pcm
#        ${CMAKE_CURRENT_BINARY_DIR}/libGundamFitParametersDict.rootmap
#        DESTINATION
#        ${CMAKE_INSTALL_PREFIX}/lib
#        COMPONENT
#        libraries
#)

