set(SRCFILES
        src/AnaFitParameters.cc
        src/DetParameters.cc
        src/FitParameters.cc
        src/FluxParameters.cc
        src/XsecParameters.cc
        src/FitParameterSet.cpp
        src/FitParameter.cpp
        src/Dial.cpp
        src/DialSet.cpp
        )

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(fitparam STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(fitparam SHARED ${SRCFILES})
endif()

set_target_properties(fitparam PROPERTIES VERSION
  "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(fitparam PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS fitparam DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} fitparam PARENT_SCOPE)
