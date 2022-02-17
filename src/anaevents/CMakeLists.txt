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

target_link_libraries( anaevents PUBLIC
  GundamUtils
  ${ROOT_LIBRARIES}
  )

set_target_properties(
        anaevents PROPERTIES VERSION
        "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

install(TARGETS anaevents DESTINATION lib)

set(MODULETargets ${MODULETargets} anaevents PARENT_SCOPE)
