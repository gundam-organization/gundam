# Samples

set( SRCFILES
        src/FitSampleSet.cpp
        src/FitSample.cpp
        src/DataSet.cpp
        src/PhysicsEvent.cpp
        src/SampleElement.cpp
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(fitsamples STATIC ${SRCFILES})
else()
  add_library(fitsamples SHARED ${SRCFILES})
endif()

#if(CMAKE_BUILD_TYPE MATCHES DEBUG)
#  add_library(samples STATIC ${SRCFILES})
#else(CMAKE_BUILD_TYPE MATCHES RELEASE)
#  add_library(samples SHARED ${SRCFILES})
#endif()

target_link_libraries( fitsamples
        xslfutils
        fitparam
        anaevents
        )

set_target_properties(
        fitsamples PROPERTIES VERSION
        "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(fitsamples PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS fitsamples DESTINATION lib)

set(MODULETargets ${MODULETargets} fitsamples PARENT_SCOPE)
