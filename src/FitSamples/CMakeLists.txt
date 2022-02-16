# Samples

set( SRCFILES
        src/SampleElement.cpp
        src/FitSampleSet.cpp
        src/FitSample.cpp
        src/PhysicsEvent.cpp
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamFitSamples STATIC ${SRCFILES})
else()
  add_library(GundamFitSamples SHARED ${SRCFILES})
endif()

target_link_libraries( GundamFitSamples PUBLIC
        GundamUtils
        GundamFitParameters
        anaevents
        ${ROOT_LIBRARIES}
        )

set_target_properties(
        GundamFitSamples PROPERTIES VERSION
        "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

install(TARGETS GundamFitSamples DESTINATION lib)

set(MODULETargets ${MODULETargets} GundamFitSamples PARENT_SCOPE)
