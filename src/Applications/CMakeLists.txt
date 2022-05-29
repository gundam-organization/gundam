# Applications

set( APPLICATION_LIST
        # GUNDAM apps
        gundamFitter
        gundamConfigUnfolder
        gundamPlotExtractor
        gundamConfigCompare
        gundamFitCompare
)

if( ENABLED_DEV_MODE )
    list(APPEND APPLICATION_LIST Sandbox)
endif()

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    target_link_libraries( ${app} GundamFitter )
#    if( USE_STATIC_LINKS )
#        target_link_libraries(${app} STATIC GundamFitter)
#    else()
#        target_link_libraries(${app} SHARED GundamFitter)
#    endif()
    install( TARGETS ${app} DESTINATION bin )
endforeach()
