# Applications

set( APPLICATION_LIST
        # GUNDAM apps
        gundamFitter
        gundamConfigUnfolder
        gundamPlotExtractor
        gundamConfigCompare
        gundamFitCompare
)

if( ENABLE_DEV_MODE )
    list(APPEND APPLICATION_LIST Sandbox)
endif()

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    target_link_libraries( ${app} GundamFitter )
    install( TARGETS ${app} DESTINATION bin )
endforeach()
