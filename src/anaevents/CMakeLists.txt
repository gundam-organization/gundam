# Samples

set( SRCFILES
        src/AnaSample.cc
        src/AnaTreeMC.cc
        src/AnaEvent.cpp
        src/PlotGenerator.cpp
        )

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  add_library(anaevents STATIC ${SRCFILES})
else(CMAKE_BUILD_TYPE MATCHES RELEASE)
  add_library(anaevents SHARED ${SRCFILES})
endif()

target_link_libraries( anaevents
        xslfutils
        )

set_target_properties(
        anaevents PROPERTIES VERSION
        "${xsLLhFitter_VERSION_MAJOR}.${xsLLhFitter_VERSION_MINOR}.${xsLLhFitter_VERSION_REVISION}")
set_target_properties(anaevents PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS anaevents DESTINATION lib)

set(MODULETargets ${MODULETargets} anaevents PARENT_SCOPE)
