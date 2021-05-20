#xsllhDetCovGenerator
add_executable(xsllhDetCovGenerator src/xsllhDetCovGenerator.cxx)

set_target_properties(xsllhDetCovGenerator PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
    target_include_directories(xsllhDetCovGenerator PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
    target_include_directories(xsllhDetCovGenerator PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()


#xsllhCalcXsec
add_executable(xsllhCalcXsec src/xsllhCalcXsec.cxx)

set_target_properties(xsllhCalcXsec PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhCalcXsec PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhCalcXsec PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhCalcXsec ${mod})
endforeach()


# xsllhND280UpFit.cxx
add_executable(xsllhND280UpFit src/xsllhND280UpFit.cxx)

set_target_properties(xsllhND280UpFit PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
    target_include_directories(xsllhND280UpFit PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
    target_include_directories(xsllhND280UpFit PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
    cmessage(STATUS "Linking with: ${mod}")
    target_link_libraries(xsllhND280UpFit ${mod})
endforeach()


#xsllhDetVariations
add_executable(xsllhDetVar src/xsllhDetVariations.cxx)
add_executable(xsllhGetDetParCov src/xsllhGetDetParCov.cxx)

set_target_properties(xsllhDetVar PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})
set_target_properties(xsllhGetDetParCov PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhDetVar PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhDetVar PUBLIC ${CMAKE_SOURCE_DIR}/include)
 target_include_directories(xsllhGetDetParCov PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhGetDetParCov PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhDetVar ${mod})
  target_link_libraries(xsllhGetDetParCov ${mod})
endforeach()

#xsllhFit
add_executable(xsllhFit src/xsllhFit.cxx)

set_target_properties(xsllhFit PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhFit PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhFit PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhFit ${mod})
endforeach()

#xsllhPlot
add_executable(xsllhPlot src/xsllhPlot.cxx)

set_target_properties(xsllhPlot PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhPlot PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhPlot PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhPlot ${mod})
endforeach()

#xsllhTreeConvert
add_executable(xsllhTreeConvert src/xsllhTreeConvert.cxx)

set_target_properties(xsllhTreeConvert PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhTreeConvert PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhTreeConvert PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhTreeConvert ${mod})
endforeach()

#xsllhGenWeightsBinner
add_executable(xsllhGenWeightsBinner src/xsllhGenWeightsBinner.cxx)

set_target_properties(xsllhGenWeightsBinner PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
    target_include_directories(xsllhGenWeightsBinner PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
    target_include_directories(xsllhGenWeightsBinner PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
    cmessage(STATUS "Linking with: ${mod}")
    target_link_libraries(xsllhGenWeightsBinner ${mod})
endforeach()

#xsllhGenWeightsFormater
add_executable(xsllhGenWeightsFormater src/xsllhGenWeightsFormater.cxx)

set_target_properties(xsllhGenWeightsFormater PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
    target_include_directories(xsllhGenWeightsFormater PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
    target_include_directories(xsllhGenWeightsFormater PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
    cmessage(STATUS "Linking with: ${mod}")
    target_link_libraries(xsllhGenWeightsFormater ${mod})
endforeach()

#xsllhUnitTests
add_executable(xsllhUnitTest src/xsllhUnitTests.cxx)

set_target_properties(xsllhUnitTest PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhUnitTest PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhUnitTest PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhUnitTest ${mod})
endforeach()

#xsllhXseCov
add_executable(xsllhXsecCov src/xsllhXsecCov.cxx)

set_target_properties(xsllhXsecCov PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhXsecCov PUBLIC ${CMAKE_SOURCE_DIR}/src/${mod}/src)
 target_include_directories(xsllhXsecCov PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhXsecCov ${mod})
endforeach()

#install targets
install(TARGETS xsllhCalcXsec xsllhND280UpFit xsllhFit xsllhDetVar xsllhPlot xsllhTreeConvert xsllhGenWeightsBinner xsllhGenWeightsFormater xsllhUnitTest xsllhXsecCov DESTINATION bin)
