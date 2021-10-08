# Samples

set( SRCFILES
        src/SampleElement.cpp
        src/FitSampleSet.cpp
        src/FitSample.cpp
        src/DataSet.cpp
        src/PhysicsEvent.cpp
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamFitSamples STATIC ${SRCFILES})
else()
  add_library(GundamFitSamples SHARED ${SRCFILES})
endif()

#if(CMAKE_BUILD_TYPE MATCHES DEBUG)
#  add_library(samples STATIC ${SRCFILES})
#else(CMAKE_BUILD_TYPE MATCHES RELEASE)
#  add_library(samples SHARED ${SRCFILES})
#endif()

target_link_libraries( GundamFitSamples
        GundamUtils
        GundamFitParameters
        anaevents
        )

set_target_properties(
        GundamFitSamples PROPERTIES VERSION
        "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}")
set_target_properties(GundamFitSamples PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS GundamFitSamples DESTINATION lib)

set(MODULETargets ${MODULETargets} GundamFitSamples PARENT_SCOPE)
