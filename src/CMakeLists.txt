
add_executable(CCQEFit EXCLUDE_FROM_ALL ccqefit.cc)

set_target_properties(CCQEFit PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(CCQEFit PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(CCQEFit ${mod})
endforeach()



add_executable(CCQEFit_daT EXCLUDE_FROM_ALL ccqefit_dat.cc)

set_target_properties(CCQEFit_daT PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(CCQEFit_daT PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(CCQEFit_daT ${mod})
endforeach()



add_executable(CCQEFit_dphiT EXCLUDE_FROM_ALL ccqefit_dphit.cc)

set_target_properties(CCQEFit_dphiT PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(CCQEFit_dphiT PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(CCQEFit_dphiT ${mod})
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



add_executable(PropError EXCLUDE_FROM_ALL propError.C)

set_target_properties(PropError PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(PropError PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(PropError ${mod})
endforeach()



add_executable(PropError_daT EXCLUDE_FROM_ALL propError_dat.C)

set_target_properties(PropError_daT PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(PropError_daT PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(PropError_daT ${mod})
endforeach()



add_executable(PropError_dphiT EXCLUDE_FROM_ALL propError_dphit.C)

set_target_properties(PropError_dphiT PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

foreach(mod ${MODULES})
 target_include_directories(PropError_dphiT PUBLIC ${CMAKE_SOURCE_DIR}/${mod}/src)
endforeach()

foreach(mod ${MODULETargets})
  cmessage(STATUS "Linking with: ${mod}")
  target_link_libraries(PropError_dphiT ${mod})
endforeach()

#install(TARGETS CCQEFit CCQEFit_daT CCQEFit_dphiT FitBase ingridFit PropError PropError_daT PropError_dphiT DESTINATION bin)
install(TARGETS ingridFit DESTINATION bin)
