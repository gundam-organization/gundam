# Applications

set( APPLICATION_LIST
        # GUNDAM apps
        gundamFitter
        gundamCalcXsec
        gundamConfigUnfolder
        gundamPlotExtractor
        gundamConfigCompare
        gundamFitCompare
        gundamRoot
)


if( ENABLE_DEV_MODE )
    list(APPEND APPLICATION_LIST Sandbox)
endif()


configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.in.h
        ${CMAKE_CURRENT_BINARY_DIR}/LinkDef.h )


ROOT_GENERATE_DICTIONARY(
        G__GundamRootDict
        ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/GenericToolbox.Root.h
        LINKDEF
        ${CMAKE_CURRENT_BINARY_DIR}/LinkDef.h
)

if( USE_STATIC_LINKS )
    add_library( GundamRootDict STATIC G__GundamRootDict.cxx )
else()
    add_library( GundamRootDict SHARED G__GundamRootDict.cxx )
endif()

target_link_libraries( GundamRootDict ${ROOT_LIBRARIES} )
install(TARGETS GundamRootDict DESTINATION lib)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/libGundamRootDict_rdict.pcm
        ${CMAKE_CURRENT_BINARY_DIR}/libGundamRootDict.rootmap
    DESTINATION
        ${CMAKE_INSTALL_PREFIX}/lib
    COMPONENT
        libraries
)

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx )
    install( TARGETS ${app} DESTINATION bin )
endforeach()

# Dependencies
target_link_libraries( gundamFitter GundamFitter )
target_link_libraries( gundamFitCompare GundamUtils )
target_link_libraries( gundamConfigUnfolder GundamUtils )
target_link_libraries( gundamConfigCompare GundamUtils )
target_link_libraries( gundamPlotExtractor GundamUtils )
target_link_libraries( gundamCalcXsec GundamPropagator )
target_link_libraries( gundamRoot GundamPropagator GundamRootDict )

