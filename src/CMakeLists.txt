#xsllhCalcXsec
add_executable(xsllhCalcXsec xsllhCalcXsec.cxx)

set_target_properties(xsllhCalcXsec PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhCalcXsec PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
 target_include_directories(xsllhCalcXsec PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhCalcXsec ${mod})
endforeach()

#xsllhDetVariations
add_executable(xsllhDetVar xsllhDetVariations.cxx)

set_target_properties(xsllhDetVar PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhDetVar PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
 target_include_directories(xsllhDetVar PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhDetVar ${mod})
endforeach()

#xsllhFit
add_executable(xsllhFit xsllhFit.cxx)

set_target_properties(xsllhFit PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhFit PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
 target_include_directories(xsllhFit PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhFit ${mod})
endforeach()

#xsllhUnitTests
add_executable(xsllhUnitTest xsllhUnitTests.cxx)

set_target_properties(xsllhUnitTest PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(xsllhUnitTest PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
 target_include_directories(xsllhUnitTest PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(xsllhUnitTest ${mod})
endforeach()

#install targets
install(TARGETS xsllhCalcXsec xsllhFit xsllhDetVar xsllhUnitTest DESTINATION bin)
