# Samples

set( SRCFILES
        src/AnaSample.cc
        src/AnaTreeMC.cc
        src/AnaEvent.cpp
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(anaevents STATIC ${SRCFILES})
else()
  add_library(anaevents SHARED ${SRCFILES})
endif()

#if(CMAKE_BUILD_TYPE MATCHES DEBUG)
#  add_library(anaevents STATIC ${SRCFILES})
#else(CMAKE_BUILD_TYPE MATCHES RELEASE)
#  add_library(anaevents SHARED ${SRCFILES})
#endif()

target_link_libraries( anaevents
        GundamUtils
        )

set_target_properties(
        anaevents PROPERTIES VERSION
        "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}")
set_target_properties(anaevents PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS anaevents DESTINATION lib)

set(MODULETargets ${MODULETargets} anaevents PARENT_SCOPE)
