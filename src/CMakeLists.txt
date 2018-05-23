
add_executable(CCQEFit EXCLUDE_FROM_ALL ccqefit.cc)

set_target_properties(CCQEFit PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(CCQEFit PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(CCQEFit ${mod})
endforeach()

add_executable(FitBase EXCLUDE_FROM_ALL fitBase.cc)

set_target_properties(FitBase PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(FitBase PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(FitBase ${mod})
endforeach()

add_executable(ingridFit ingridFit.cc)

set_target_properties(ingridFit PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(ingridFit PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
 target_include_directories(ingridFit PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(ingridFit ${mod})
endforeach()

add_executable(calcXsec calcXsec.cc)

set_target_properties(calcXsec PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(calcXsec PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
 target_include_directories(calcXsec PUBLIC ${CMAKE_SOURCE_DIR}/include)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(calcXsec ${mod})
endforeach()

add_executable(PropError EXCLUDE_FROM_ALL propError.C)

set_target_properties(PropError PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(PropError PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(PropError ${mod})
endforeach()

#install(TARGETS CCQEFit CCQEFit_daT CCQEFit_dphiT FitBase ingridFit PropError PropError_daT PropError_dphiT DESTINATION bin)
install(TARGETS calcXsec ingridFit DESTINATION bin)
