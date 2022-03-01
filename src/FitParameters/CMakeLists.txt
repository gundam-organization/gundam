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
        src/GraphDial.cpp
        )

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamFitParameters STATIC ${SRCFILES})
else()
  add_library(GundamFitParameters SHARED ${SRCFILES})
endif()

# Dependencies
target_link_libraries( GundamFitParameters PUBLIC
        anaevents
        GundamUtils
        ${ROOT_LIBRARIES}
)

set_target_properties(GundamFitParameters PROPERTIES VERSION
  "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

install(TARGETS GundamFitParameters DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} GundamFitParameters PARENT_SCOPE)
