# Applications

set( APPLICATION_LIST
        # Old xsllhFitter tools
        xsllhND280UpFit
        xsllhFit
        xsllhTreeConvert
        xsllhGenWeightsFormater
        xsllhGetDetParCov
        xsllhCalcXsec
        xsllhDetVariations
        xsllhPlot
        xsllhUnitTests
        xsllhXsecCov
)

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    target_link_libraries(${app}  XsllhFitter)
    install( TARGETS ${app} DESTINATION bin )
endforeach()
