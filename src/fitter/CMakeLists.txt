set(SRCFILES
        src/ND280Fitter.cpp
        src/XsecFitter.cc
        src/FitterEngine.cpp
        )
set(HEADERS
        include/ND280Fitter.h
        include/XsecFitter.hh
        include/FitterEngine.h
        )
if( USE_STATIC_LINKS STREQUAL "1" )
  add_library(fitter STATIC ${SRCFILES})
else()
  add_library(fitter SHARED ${SRCFILES})
endif()
#
#if(CMAKE_BUILD_TYPE MATCHES DEBUG)
#  add_library(fitter STATIC ${SRCFILES})
#else(CMAKE_BUILD_TYPE MATCHES RELEASE)
#  add_library(fitter SHARED ${SRCFILES})
#endif()

target_link_libraries( fitter
        errprop
        )

set_target_properties(
        fitter PROPERTIES VERSION
        "${gundam_VERSION_MAJOR}.${gundam_VERSION_MINOR}.${gundam_VERSION_REVISION}"
)

set_target_properties(
        fitter PROPERTIES LINK_FLAGS
        ${ROOT_LD_FLAGS}
)

install(TARGETS fitter DESTINATION lib)
#Can uncomment this to install the headers... but is it really neccessary?
# install(FILES ${HEADERS} DESTINATION include)

set(MODULETargets ${MODULETargets} fitter PARENT_SCOPE)
