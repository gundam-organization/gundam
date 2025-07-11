# Applications

set(
    APPLICATION_LIST
    # GUNDAM C++ apps
    gundamFitter
    gundamFitReader
    gundamCalcXsec
    gundamFitCompare
    gundamFitPlot
    gundamInputZipper
    gundamConfigCompare
    gundamPlotExtractor
    gundamConfigUnfolder
)

set(SCRIPT_LIST
  # GUNDAM scripts (not compiled)
  gundamContinue
)

if( WITH_GUNDAM_SANDBOX_APP )
    list(APPEND APPLICATION_LIST Sandbox)
endif()

if( WITH_GUNDAM_ROOT_APP )
    list(APPEND APPLICATION_LIST gundamRoot)

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
        ${CMAKE_SOURCE_DIR}/submodules/cpp-generic-toolbox/include/GenericToolbox.Utils.h
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

    target_link_libraries( GundamRootDict PRIVATE ${ROOT_LIBRARIES} )
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
endif()



# Now define the frontends
foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx )
    install( TARGETS ${app} DESTINATION bin )
endforeach()

# Dependencies
target_link_libraries( gundamFitter GundamFitter )
target_link_libraries( gundamCalcXsec GundamFitter ) # using the fitter engine to parse back the config file
target_link_libraries( gundamFitReader GundamUtils )
target_link_libraries( gundamInputZipper GundamUtils )
target_link_libraries( gundamFitCompare GundamUtils )
target_link_libraries( gundamFitPlot GundamUtils )
target_link_libraries( gundamConfigUnfolder GundamUtils )
target_link_libraries( gundamConfigCompare GundamUtils )
target_link_libraries( gundamPlotExtractor GundamUtils )

if( WITH_GUNDAM_ROOT_APP )
#target_sources( gundamRoot PRIVATE G__GundamRootDict.cxx )
#target_link_libraries( gundamRoot GundamPropagator )
target_link_libraries( gundamRoot
    GundamRootDict GundamUtils
    #    GundamPropagator
    )
endif ()

foreach( script ${SCRIPT_LIST} )
  cmessage( STATUS "Adding script: ${script}" )
  install(PROGRAMS src/${script} DESTINATION bin)
endforeach()
