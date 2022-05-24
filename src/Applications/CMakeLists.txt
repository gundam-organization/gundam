# Applications

set( APPLICATION_LIST
        # GUNDAM apps
        gundamFitter
        gundamConfigUnfolder
        gundamPlotExtractor
        gundamConfigCompare
        gundamFitCompare

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

        # Others
        Sandbox
)

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    target_link_libraries(${app}  GundamFitter)
    install( TARGETS ${app} DESTINATION bin )
endforeach()
