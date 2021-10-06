set(SRCFILES
        src/AnaFitParameters.cc
        src/DetParameters.cc
        src/FitParameters.cc
        src/FluxParameters.cc
        src/XsecParameters.cc
        src/XsecDial.cc

        src/FitParameterSet.cpp
        src/FitParameter.cpp
        src/DialSet.cpp
        src/Dial.cpp
        src/NormalizationDial.cpp
        src/SplineDial.cpp
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(fitparam STATIC ${SRCFILES})
else()
  add_library(fitparam SHARED ${SRCFILES})
endif()
#
#if(CMAKE_BUILD_TYPE MATCHES DEBUG)
#  add_library(fitparam STATIC ${SRCFILES})
#else(CMAKE_BUILD_TYPE MATCHES RELEASE)
#  add_library(fitparam SHARED ${SRCFILES})
#endif()

# Dependencies
target_link_libraries( fitparam
        anaevents
        xslfutils
)

set_target_properties(fitparam PROPERTIES VERSION
  "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}")
set_target_properties(fitparam PROPERTIES LINK_FLAGS ${ROOT_LD_FLAGS})

install(TARGETS fitparam DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} fitparam PARENT_SCOPE)
