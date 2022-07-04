# Samples

set( SRCFILES
        src/SampleElement.cpp
        src/FitSampleSet.cpp
        src/FitSample.cpp
        src/PhysicsEvent.cpp
        src/PlotGenerator.cpp
        src/JointProbability.cpp
        )

if( USE_STATIC_LINKS )
  add_library(GundamFitSamples STATIC ${SRCFILES})
else()
  add_library(GundamFitSamples SHARED ${SRCFILES})
endif()

target_include_directories(GundamFitSamples PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries( GundamFitSamples PUBLIC GundamUtils GundamFitParameters ${ROOT_LIBRARIES} )

#set_target_properties( GundamFitSamples PROPERTIES VERSION "${GUNDAM_VERSION_STRING}")

install(TARGETS GundamFitSamples DESTINATION lib)

