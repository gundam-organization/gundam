set(SRCFILES
        src/ND280Fitter.cpp
        src/XsecFitter.cc
        src/FitterEngine.cpp
#        src/MinimizerInterface.cpp
        )
set(HEADERS
        include/ND280Fitter.h
        include/XsecFitter.hh
        include/FitterEngine.h
#        include/MinimizerInterface.h
        )
if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(GundamFitter STATIC ${SRCFILES})
else()
  add_library(GundamFitter SHARED ${SRCFILES})
endif()

target_link_libraries( GundamFitter
        GundamErrorPropagator
        )

set_target_properties(
        GundamFitter PROPERTIES VERSION
        "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}${gundam_VERSION_TAG}"
)

set_target_properties(
        GundamFitter PROPERTIES LINK_FLAGS
        ${ROOT_LD_FLAGS}
)

install(TARGETS GundamFitter DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} GundamFitter PARENT_SCOPE)
