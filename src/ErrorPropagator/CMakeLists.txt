set(SRCFILES
        src/FitObj.cc
        src/XsecCalc.cc
        src/Propagator.cpp
        src/PlotGenerator.cpp
        src/DataSetLoader.cpp
        src/EventTreeWriter.cpp
)

set(HEADERS
        include/FitObj.hh
        include/XsecCalc.hh
        include/Propagator.h
        include/PlotGenerator.h
        include/DataSetLoader.h
        include/EventTreeWriter.h
)

if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamErrorPropagator STATIC ${SRCFILES})
else()
  add_library(GundamErrorPropagator SHARED ${SRCFILES})
endif()

target_link_libraries( GundamErrorPropagator
        GundamFitParameters
        GundamFitSamples
        GundamCache
        anaevents
        ${ROOT_LIBRARIES}
        )

set_target_properties(GundamErrorPropagator PROPERTIES VERSION
  "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}")

install(TARGETS GundamErrorPropagator DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} GundamErrorPropagator PARENT_SCOPE)
