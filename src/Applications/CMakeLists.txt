# Applications

set( APPLICATION_LIST
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
        gundamFitter
        gundamConfigUnfolder
        gundamPlotExtractor
        gundamConfigCompare
        YamlSandbox
)

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    target_link_libraries(${app}  GundamFitter)
    install( TARGETS ${app} DESTINATION bin )
endforeach()
