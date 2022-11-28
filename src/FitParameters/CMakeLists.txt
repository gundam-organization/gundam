set(SRCFILES
        src/FitParameterSet.cpp
        src/FitParameter.cpp
        src/DialSet.cpp
        src/Dial.cpp
        src/NormDial.cpp
        src/SplineDial.cpp
        src/GraphDial.cpp
        src/NestedDialTest.cpp
        )

#file( GLOB MY_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../Utils/include/*.h" )

set(HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Dial.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/DialSet.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/DialWrapper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/FitParameter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/FitParameterSet.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/GraphDial.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/NestedDialTest.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/NormDial.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SplineDial.h
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

