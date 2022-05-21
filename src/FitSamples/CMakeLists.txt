# Samples

set( SRCFILES
        src/SampleElement.cpp
        src/FitSampleSet.cpp
        src/FitSample.cpp
        src/PhysicsEvent.cpp
        src/PlotGenerator.cpp
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamFitSamples STATIC ${SRCFILES})
else()
  add_library(GundamFitSamples SHARED ${SRCFILES})
endif()

target_include_directories(GundamFitSamples PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

