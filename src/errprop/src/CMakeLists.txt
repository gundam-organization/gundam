set(SRCFILES FitObj.cc XsecCalc.cc)
set(HEADERS FitObj.hh XsecCalc.hh)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(errprop STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(errprop SHARED ${SRCFILES})
endif()

set_target_properties(errprop PROPERTIES VERSION
  "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(errprop PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})
target_include_directories(errprop PUBLIC ${CMAKE_SOURCE_DIR}/src/anaevents/src
    ${CMAKE_SOURCE_DIR}/src/fitparam/src ${CMAKE_SOURCE_DIR}/src/utils/src ${CMAKE_SOURCE_DIR}/include)

install(TARGETS errprop DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} errprop PARENT_SCOPE)
