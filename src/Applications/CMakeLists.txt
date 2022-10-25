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

#set (DICT_HEADERS FitterEngine.h)
#
#ROOT_GENERATE_DICTIONARY(FitterEngineDict ${DICT_HEADERS} LINKDEF ./GundamLinkDef.h)
#list(APPEND SRCFILES FitterEngineDict.cxx)


if( ENABLE_DEV_MODE )
    list(APPEND APPLICATION_LIST Sandbox)
endif()

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    target_link_libraries( ${app} GundamFitter )
    install( TARGETS ${app} DESTINATION bin )
endforeach()
