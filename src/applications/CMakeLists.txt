# Applications

set( APPLICATION_LIST
        xsllhND280UpFit
        xsllhFit
        xsllhTreeConvert
        xsllhGenWeightsFormater
        xsllhDetCovGenerator
        xsllhGetDetParCov
        xsllhCalcXsec
        xsllhDetVariations
        xsllhPlot
        xsllhUnitTests
        xsllhXsecCov
        xsllhFitter
        YamlSandbox
)

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    cmessage(STATUS "${app} external links: ${LINK_LIBARY_LIST}")
    target_link_libraries(${app} ${LINK_LIBARY_LIST})
    set_target_properties( ${app} PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS} )
    foreach(mod ${MODULETargets})
        cmessage(STATUS "${app}: adding link with ${mod}")
        target_link_libraries(${app} ${mod})
    endforeach()
    install( TARGETS ${app} DESTINATION bin )
endforeach()
