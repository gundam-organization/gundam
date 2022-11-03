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

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    install( TARGETS ${app} DESTINATION bin )
endforeach()

# Dependencies
target_link_libraries( gundamFitter GundamFitter )
target_link_libraries( gundamFitCompare Utils )
target_link_libraries( gundamConfigUnfolder Utils )
target_link_libraries( gundamPlotExtractor Utils )
target_link_libraries( gundamCalcXsec Propagator )

