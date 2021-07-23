set(SRCFILES
  src/FitObj.cc
  src/XsecCalc.cc
  src/Propagator.cpp
)

set(HEADERS
  include/FitObj.hh
  include/XsecCalc.hh
  include/Propagator.h
)

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(errprop STATIC ${SRCFILES})
else()
  add_library(errprop SHARED ${SRCFILES})
endif()

target_link_libraries( errprop
        fitparam
        anaevents
        fitsamples
        )

set_target_properties(errprop PROPERTIES VERSION
  "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(errprop PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS errprop DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} errprop PARENT_SCOPE)
