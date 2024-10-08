set(SRCFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GundamGlobals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BinSet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Bin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GundamGreetings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ParameterThrowerMarkHarz.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ConfigUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GundamUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RootUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GundamApp.cpp
    )

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CalculateGeneralSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CalculateMonotonicSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/CalculateUniformSpline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Bin.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/BinSet.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GundamGlobals.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GundamGreetings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/VariableDictionary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ParameterThrowerMarkHarz.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ConfigUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GundamUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GundamApp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/GundamBacktrace.h
    )


set( UTILS_HEADERS ${HEADERS} )

#ROOT_GENERATE_DICTIONARY(
#        GundamUtilsDict
#        ${HEADERS}
#        LINKDEF
#        ${CMAKE_CURRENT_SOURCE_DIR}/UtilsLinkDef.h
#)

# TODO: dont do that -> shared lib only
#list(APPEND SRCFILES GundamUtilsDict.cxx)

if( USE_STATIC_LINKS )
  add_library( GundamUtils STATIC ${SRCFILES})
else()
  add_library( GundamUtils SHARED ${SRCFILES})
endif()

add_dependencies( GundamUtils preBuildVersionCheck )
include_directories( ${CMAKE_BINARY_DIR}/generated/ ) # version header

# Make sure the current directories are available for the later
# compilation.
target_include_directories( GundamUtils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries( GundamUtils PUBLIC ${ROOT_LIBRARIES} )
install( TARGETS GundamUtils DESTINATION lib )

#Can uncomment this to install the headers... but is it really neccessary?
#install(FILES ${HEADERS} DESTINATION include)


#install(
#        FILES
#        ${CMAKE_CURRENT_BINARY_DIR}/libGundamUtilsDict_rdict.pcm
#        ${CMAKE_CURRENT_BINARY_DIR}/libGundamUtilsDict.rootmap
#        DESTINATION
#        ${CMAKE_INSTALL_PREFIX}/lib
#        COMPONENT
#        libraries
#)
