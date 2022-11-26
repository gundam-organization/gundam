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


# Need a full unfold of the source path
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.in.h
    ${CMAKE_CURRENT_BINARY_DIR}/LinkDef.h
)

# Generate .cxx file
# Explicit include of the GenericToolbox.Root.h
ROOT_GENERATE_DICTIONARY(
    G__GundamRootDict
    ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/GenericToolbox.Root.h
    ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/GenericToolbox.RawDataArray.h
    ${CMAKE_SOURCE_DIR}/submodules/simple-cpp-logger/include/Logger.h
    LINKDEF
    ${CMAKE_CURRENT_BINARY_DIR}/LinkDef.h
)

# Needs to be a shared library
add_library(
    GundamRootDict SHARED
    G__GundamRootDict.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.cpp
)
#target_compile_features(GundamRootDict PRIVATE cxx_std_17)

target_link_libraries( GundamRootDict ${ROOT_LIBRARIES} )
install(TARGETS GundamRootDict DESTINATION lib)

# Also need to install dict files
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libGundamRootDict_rdict.pcm
    ${CMAKE_CURRENT_BINARY_DIR}/libGundamRootDict.rootmap
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib
    COMPONENT
    libraries
)

# Now define the frontends
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

#target_sources( gundamRoot PRIVATE G__GundamRootDict.cxx )
#target_link_libraries( gundamRoot GundamPropagator )
target_link_libraries( gundamRoot
    GundamRootDict
#    GundamPropagator
    )

