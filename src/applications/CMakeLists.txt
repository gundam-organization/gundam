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
)

foreach( app ${APPLICATION_LIST} )
    cmessage( STATUS "Defining app: ${app}" )
    add_executable( ${app} src/${app}.cxx)
    set_target_properties( ${app} PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS} )
    foreach(mod ${MODULETargets})
        cmessage(STATUS "Linking ${app} with: ${mod}")
        target_link_libraries(${app} ${mod})
    endforeach()
    install( TARGETS ${app} DESTINATION bin )
endforeach()
