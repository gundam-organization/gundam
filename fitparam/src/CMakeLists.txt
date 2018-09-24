set(SRCFILES AnaFitParameters.cc DetParameters.cc FitParameters.cc
  FluxParameters.cc FluxParameters_norm.cc XsecParameters.cc)
set(HEADERS AnaFitParameters.hh DetParameters.hh FitParameters.hh
  FluxParameters.hh FluxParameters_norm.hh XsecParameters.hh)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(fitparam STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(fitparam SHARED ${SRCFILES})
endif()

set_target_properties(fitparam PROPERTIES VERSION
  "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(fitparam PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})
target_include_directories(fitparam PUBLIC ${CMAKE_SOURCE_DIR}/anaevents/src
    ${CMAKE_SOURCE_DIR}/utils/src ${CMAKE_SOURCE_DIR}/include)

install(TARGETS fitparam DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} fitparam PARENT_SCOPE)
